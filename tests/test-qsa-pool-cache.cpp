// Dumps the logits of a fixed token schedule so two runs can be compared byte for byte.
// The schedule covers what the pooled indexer-key cache (qwen4exp QSA) has to get right:
// a prompt decoded in ubatches smaller than the block ratio's multiples, single-token
// decodes that complete a block every ratio steps, rollbacks inside the recurrent ring
// that reuse cells, and wider batches that pool several blocks at once.
//
//   LLAMA_QSA_POOL_CACHE=1 test-qsa-pool-cache model.gguf on.bin  [n_ubatch] [n_ctx] [n_prompt] [n_repeat]
//   LLAMA_QSA_POOL_CACHE=0 test-qsa-pool-cache model.gguf off.bin [n_ubatch] [n_ctx] [n_prompt] [n_repeat]
//   cmp on.bin off.bin
//
// n_repeat > 1 runs the schedule again in a fresh context of the same process (out.bin.2,
// ...): a second context inherits whatever the first left in freed device memory, which a
// fresh process never sees.
//
// The two files must be identical: every pooled row is a per-row computation on the same
// cached keys, so caching it cannot change a bit.
//
// A second file, out.bin.topk, lists the selected cells of every QSA layer for every token
// (sorted, one line per token and layer). Two runs that select the same cells, whatever the
// numerics of the attention over them, produce identical .topk files: that is the gate for
// changes to the selection (block-level top-k, pooled keys) on a model whose experts
// amplify any rounding difference into different logits. A prompt longer than the default
// writes only its last logit row, so a deep run stays small.

#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

struct topk_capture {
    FILE * out = nullptr;
    int    step = 0;
    std::vector<int32_t> buf;
};

// the scheduler asks once per tensor whether it should be observed, then hands it over computed
static bool topk_cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * cap = (topk_capture *) ud;

    const bool want = strncmp(t->name, "indexer_top_k", 13) == 0;

    if (ask) {
        return want;
    }

    if (!want || t->type != GGML_TYPE_I32) {
        return true;
    }

    const int64_t width = t->ne[0];
    const int64_t n_row = ggml_nelements(t)/width;

    cap->buf.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, cap->buf.data(), 0, ggml_nbytes(t));

    for (int64_t r = 0; r < n_row; ++r) {
        std::vector<int32_t> row(cap->buf.begin() + r*width, cap->buf.begin() + (r + 1)*width);
        std::sort(row.begin(), row.end());

        fprintf(cap->out, "%d %s %" PRId64 ":", cap->step, t->name, r);
        for (const int32_t c : row) {
            fprintf(cap->out, " %d", c);
        }
        fprintf(cap->out, "\n");
    }

    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf out.bin [n_ubatch]\n", argv[0]);
        return 1;
    }

    const char *   model_path = argv[1];
    const char *   out_path   = argv[2];
    const uint32_t n_ubatch   = argc > 3 ? (uint32_t) atoi(argv[3]) : 8;
    const uint32_t n_ctx      = argc > 4 ? (uint32_t) atoi(argv[4]) : 256;
    const int      n_prompt   = argc > 5 ? atoi(argv[5]) : 61;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();

    // LLAMA_QSA_LAZY=on|auto|off: the lazy tensor mode (the PLE table read through the mmap on
    // demand), which the exactness test runs with on
    if (const char * lz = getenv("LLAMA_QSA_LAZY")) {
        mparams.load_mode = LLAMA_LOAD_MODE_MMAP;
        mparams.lazy_mode = strcmp(lz, "on") == 0 ? LLAMA_LAZY_MODE_ON : strcmp(lz, "auto") == 0 ? LLAMA_LAZY_MODE_AUTO : LLAMA_LAZY_MODE_OFF;
    }

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", model_path);
        return 1;
    }

    topk_capture cap;
    cap.out = fopen((std::string(out_path) + ".topk").c_str(), "w");

    const int n_repeat = argc > 6 ? std::max(1, atoi(argv[6])) : 1;

    const std::string out_base = out_path;

    for (int rep = 1; rep <= n_repeat; ++rep) {
    const std::string out_rep = rep == 1 ? out_base : out_base + "." + std::to_string(rep);
    out_path = out_rep.c_str();
    cap.out = rep == 1 ? cap.out : fopen((std::string(out_path) + ".topk").c_str(), "w");
    cap.step = 0;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = n_ctx;
    cparams.n_batch         = std::max<uint32_t>(64, n_ubatch);
    cparams.n_ubatch        = n_ubatch;
    cparams.n_seq_max       = 1;
    cparams.n_rs_seq        = 4;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.cb_eval         = topk_cb;
    cparams.cb_eval_user_data = &cap;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create the context\n");
        return 1;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    FILE * out = fopen(out_path, "wb");
    if (out == nullptr) {
        fprintf(stderr, "failed to open %s\n", out_path);
        return 1;
    }

    uint32_t rng = 12345;

    // LLAMA_QSA_PROMPT_TEXT=<text>: the prompt is that text tokenized and then continued
    // with period 17, the way test-spec-decode-exactness builds its prompt; a periodic prompt
    // makes whole blocks identical, which random tokens never do
    std::vector<llama_token> text_toks;
    if (const char * txt = getenv("LLAMA_QSA_PROMPT_TEXT")) {
        text_toks.resize(strlen(txt) + 16);
        const int n = llama_tokenize(vocab, txt, (int32_t) strlen(txt), text_toks.data(), (int32_t) text_toks.size(), true, true);
        GGML_ASSERT(n > 0);
        text_toks.resize(n);
        while ((int) text_toks.size() < n_prompt + 200) {
            text_toks.push_back(text_toks[text_toks.size() % 17]);
        }
    }

    size_t n_text_used = 0;

    auto next_tok = [&]() {
        if (!text_toks.empty()) {
            return text_toks[n_text_used++ % text_toks.size()];
        }
        rng = rng*1664525u + 1013904223u;
        return (llama_token) ((rng >> 8) % n_vocab);
    };

    llama_pos pos = 0;

    llama_batch batch = llama_batch_init(std::max(64, n_prompt), 0, 1);

    size_t n_rows = 0;

    // a long prompt keeps only its last logit row; the batch may exceed n_batch and is split
    // LLAMA_QSA_ALL_LOGITS=1 requests a logit row for every prompt token, as
    // test-spec-decode-exactness does (its 9000-token prefill fills a 9 GB output buffer);
    // only the last row is written to the file either way
    static const bool all_logits = getenv("LLAMA_QSA_ALL_LOGITS") != nullptr && atoi(getenv("LLAMA_QSA_ALL_LOGITS")) != 0;

    std::function<void(const std::vector<llama_token> &)> decode_impl;
    decode_impl = [&](const std::vector<llama_token> & toks) {
        const bool last_only = toks.size() > 64;

        for (size_t off = 0; off < toks.size(); off += cparams.n_batch) {
            const size_t n = std::min<size_t>(cparams.n_batch, toks.size() - off);

            batch.n_tokens = 0;

            for (size_t i = 0; i < n; ++i) {
                const int j = batch.n_tokens++;

                batch.token[j]     = toks[off + i];
                batch.pos[j]       = pos + (llama_pos) (off + i);
                batch.n_seq_id[j]  = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j]    = all_logits || !last_only || off + i + 1 == toks.size();
            }

            cap.step++;

            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "decode failed at pos %d\n", pos);
                exit(1);
            }

            for (int j = 0; j < batch.n_tokens; ++j) {
                if (!batch.logits[j] || (last_only && off + j + 1 != toks.size())) {
                    continue;
                }

                const float * logits = llama_get_logits_ith(ctx, j);
                fwrite(logits, sizeof(float), n_vocab, out);
                n_rows++;
            }
        }

        pos += (llama_pos) toks.size();
    };

    // LLAMA_QSA_ONE_AT_A_TIME=1 decodes every step of the schedule after the prompt one token
    // per llama_decode, so the selections of the wide steps can be diffed against the same
    // queries seen one at a time (the .topk rows line up by query order)
    static const bool one_at_a_time = getenv("LLAMA_QSA_ONE_AT_A_TIME") != nullptr && atoi(getenv("LLAMA_QSA_ONE_AT_A_TIME")) != 0;

    auto decode = [&](const std::vector<llama_token> & toks) {
        if (one_at_a_time && toks.size() > 1 && toks.size() <= 64) {
            for (const llama_token t : toks) {
                decode_impl({ t });
            }
            return;
        }
        decode_impl(toks);
    };

    auto rollback = [&](int n) {
        if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, pos - n, -1)) {
            fprintf(stderr, "rollback of %d refused at pos %d\n", n, pos);
            exit(1);
        }

        pos -= n;
    };

    // a prompt that is neither a multiple of the ubatch nor of the block ratio
    {
        std::vector<llama_token> prompt;
        for (int i = 0; i < n_prompt; ++i) {
            prompt.push_back(next_tok());
        }
        decode(prompt);
    }

    // single-token decodes complete a block every `ratio` steps
    for (int i = 0; i < 30; ++i) {
        decode({ next_tok() });
    }

    // a rollback inside the ring, then the freed cells are reused by fresh tokens
    rollback(3);
    for (int i = 0; i < 3; ++i) {
        decode({ next_tok() });
    }

    // a wider batch pools several blocks in one window
    {
        std::vector<llama_token> toks;
        for (int i = 0; i < 5; ++i) {
            toks.push_back(next_tok());
        }
        decode(toks);
    }

    // a rollback that leaves a block incomplete, then a verify-width batch
    rollback(2);
    {
        std::vector<llama_token> toks;
        for (int i = 0; i < 7; ++i) {
            toks.push_back(next_tok());
        }
        decode(toks);
    }

    for (int i = 0; i < 20; ++i) {
        decode({ next_tok() });
    }

    fclose(out);
    fclose(cap.out);

    printf("%s: %zu logit rows of %d, final pos %d, top-k lines in %s.topk\n", out_path, n_rows, n_vocab, pos, out_path);

    llama_batch_free(batch);
    llama_free(ctx);
    } // rep

    llama_model_free(model);
    llama_backend_free();

    return 0;
}
