// Dumps the logits of a fixed token schedule so two runs can be compared byte for byte.
// The schedule covers what the pooled indexer-key cache (qwen4exp QSA) has to get right:
// a prompt decoded in ubatches smaller than the block ratio's multiples, single-token
// decodes that complete a block every ratio steps, rollbacks inside the recurrent ring
// that reuse cells, and wider batches that pool several blocks at once.
//
//   LLAMA_QSA_POOL_CACHE=1 test-qsa-pool-cache model.gguf on.bin  [n_ubatch]
//   LLAMA_QSA_POOL_CACHE=0 test-qsa-pool-cache model.gguf off.bin [n_ubatch]
//   cmp on.bin off.bin
//
// The two files must be identical: every pooled row is a per-row computation on the same
// cached keys, so caching it cannot change a bit.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf out.bin [n_ubatch]\n", argv[0]);
        return 1;
    }

    const char *   model_path = argv[1];
    const char *   out_path   = argv[2];
    const uint32_t n_ubatch   = argc > 3 ? (uint32_t) atoi(argv[3]) : 8;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", model_path);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 256;
    cparams.n_batch         = 64;
    cparams.n_ubatch        = n_ubatch;
    cparams.n_seq_max       = 1;
    cparams.n_rs_seq        = 4;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

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

    llama_batch batch = llama_batch_init(64, 0, 1);

    size_t n_rows = 0;

    auto decode = [&](const std::vector<llama_token> & toks) {
        batch.n_tokens = 0;

        for (size_t i = 0; i < toks.size(); ++i) {
            const int j = batch.n_tokens++;

            batch.token[j]     = toks[i];
            batch.pos[j]       = pos + (llama_pos) i;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = 1;
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at pos %d\n", pos);
            exit(1);
        }

        pos += (llama_pos) toks.size();

        for (int j = 0; j < batch.n_tokens; ++j) {
            const float * logits = llama_get_logits_ith(ctx, j);
            fwrite(logits, sizeof(float), n_vocab, out);
            n_rows++;
        }
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
        for (int i = 0; i < 61; ++i) {
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

    printf("%s: %zu logit rows of %d, final pos %d\n", out_path, n_rows, n_vocab, pos);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
