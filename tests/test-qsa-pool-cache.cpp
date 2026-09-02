// Dumps the logits of a fixed token schedule so two runs can be compared byte for byte.
// The schedule covers what the pooled indexer-key cache (qwen4exp QSA) has to get right:
// a prompt decoded in ubatches smaller than the block ratio's multiples, single-token
// decodes that complete a block every ratio steps, rollbacks inside the recurrent ring
// that reuse cells, and wider batches that pool several blocks at once.
//
//   LLAMA_QSA_POOL_CACHE=1 test-qsa-pool-cache model.gguf on.bin  [n_ubatch] [n_ctx] [n_prompt]
//   LLAMA_QSA_POOL_CACHE=0 test-qsa-pool-cache model.gguf off.bin [n_ubatch] [n_ctx] [n_prompt]
//   cmp on.bin off.bin
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

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", model_path);
        return 1;
    }

    topk_capture cap;
    cap.out = fopen((std::string(out_path) + ".topk").c_str(), "w");

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

    auto next_tok = [&]() {
        rng = rng*1664525u + 1013904223u;
        return (llama_token) ((rng >> 8) % n_vocab);
    };

    llama_pos pos = 0;

    llama_batch batch = llama_batch_init(std::max(64, n_prompt), 0, 1);

    size_t n_rows = 0;

    // a long prompt keeps only its last logit row; the batch may exceed n_batch and is split
    auto decode = [&](const std::vector<llama_token> & toks) {
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
                batch.logits[j]    = !last_only || off + i + 1 == toks.size();
            }

            cap.step++;

            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "decode failed at pos %d\n", pos);
                exit(1);
            }

            for (int j = 0; j < batch.n_tokens; ++j) {
                if (!batch.logits[j]) {
                    continue;
                }

                const float * logits = llama_get_logits_ith(ctx, j);
                fwrite(logits, sizeof(float), n_vocab, out);
                n_rows++;
            }
        }

        pos += (llama_pos) toks.size();
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
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
