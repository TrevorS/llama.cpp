// Probe: does evict+re-decode perturb DSV4 target state?
// Pass A: straight greedy decode of N tokens.
// Pass B: same, but before each accepted token, decode a junk token at the
//         same position, llama_memory_seq_rm it, then decode the real token.
// Lossless rollback => identical token streams.
#include "llama.h"
#include "../src/llama-ext.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static llama_token greedy(llama_context * ctx) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(ctx, -1);
    llama_token best = 0;
    float best_v = -1e30f;
    for (int i = 0; i < n_vocab; ++i) {
        if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    }
    return best;
}

static int decode_one(llama_context * ctx, llama_token tok, llama_pos pos, bool want_logits) {
    llama_batch b = llama_batch_init(1, 0, 1);
    b.n_tokens = 1;
    b.token[0] = tok;
    b.pos[0] = pos;
    b.n_seq_id[0] = 1;
    b.seq_id[0][0] = 0;
    b.logits[0] = want_logits;
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [n_gen]\n", argv[0]); return 1; }
    const char * model_path = argv[1];
    const int n_gen = argc > 2 ? atoi(argv[2]) : 48;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    mparams.use_mmap = false;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 512;
    cparams.n_ubatch = 512;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char * prompt = "Explain, step by step, how a hash map handles collisions and when you would prefer a B-tree instead.";
    std::vector<llama_token> ptoks(512);
    int n_p = llama_tokenize(vocab, prompt, (int) strlen(prompt), ptoks.data(), (int) ptoks.size(), true, true);
    ptoks.resize(n_p);
    fprintf(stderr, "prompt tokens: %d\n", n_p);

    const llama_token junk = 12345; // arbitrary wrong token
    const int junk_depth = argc > 4 ? atoi(argv[4]) : 1; // junk tokens decoded ahead before evicting
    const bool use_ckpt  = argc > 5 && atoi(argv[5]) != 0; // rollback via state save/restore (server ckpt flow)
    const bool batch_junk = argc > 6 && atoi(argv[6]) != 0; // decode the junk tokens as ONE batch (like a verify)

    const bool with_export = argc > 3 && atoi(argv[3]) != 0;
    int n_embd_h = 0;
    if (with_export) {
        llama_set_embeddings_nextn(ctx, true, /*masked*/ false);
        n_embd_h = llama_model_n_embd_nextn(model);
        fprintf(stderr, "h export enabled, n_embd_nextn=%d\n", n_embd_h);
    }
    std::vector<std::vector<float>> h_rows_a, h_rows_b;

    auto run_pass = [&](bool with_rollback) {
        llama_memory_t mem = llama_get_memory(ctx);
        llama_memory_clear(mem, true);

        llama_batch b = llama_batch_init(512, 0, 1);
        b.n_tokens = n_p;
        for (int i = 0; i < n_p; ++i) {
            b.token[i] = ptoks[i];
            b.pos[i] = i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = (i == n_p - 1);
        }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "prefill failed\n"); exit(1); }
        llama_batch_free(b);

        auto & h_rows = with_rollback ? h_rows_b : h_rows_a;
        h_rows.clear();

        std::vector<llama_token> out;
        llama_token cur = greedy(ctx);
        out.push_back(cur);
        llama_pos pos = n_p;
        for (int t = 0; t < n_gen - 1; ++t) {
            if (with_rollback) {
                // decode junk token(s) starting at this position, then evict them
                // (junk_depth > 1 exercises the deep-rollback branch of seq_rm)
                std::vector<uint8_t> ckpt;
                if (use_ckpt) {
                    const size_t sz = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    ckpt.resize(sz);
                    llama_state_seq_get_data_ext(ctx, ckpt.data(), sz, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
                if (batch_junk) {
                    llama_batch b = llama_batch_init(junk_depth, 0, 1);
                    b.n_tokens = junk_depth;
                    for (int j = 0; j < junk_depth; ++j) {
                        b.token[j] = junk;
                        b.pos[j] = pos + j;
                        b.n_seq_id[j] = 1;
                        b.seq_id[j][0] = 0;
                        b.logits[j] = true;
                    }
                    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "junk batch decode failed at %d\n", pos); exit(1); }
                    llama_batch_free(b);
                } else {
                    for (int j = 0; j < junk_depth; ++j) {
                        if (decode_one(ctx, junk, pos + j, true) != 0) { fprintf(stderr, "junk decode failed at %d\n", pos + j); exit(1); }
                    }
                }
                if (use_ckpt) {
                    // server ckpt flow: restore comp state, then trim the raw tail
                    if (llama_state_seq_set_data_ext(ctx, ckpt.data(), ckpt.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
                        fprintf(stderr, "state restore failed at %d\n", pos); exit(1);
                    }
                }
                if (!llama_memory_seq_rm(mem, 0, pos, -1)) { fprintf(stderr, "seq_rm failed at %d\n", pos); exit(1); }
            }
            if (decode_one(ctx, cur, pos, true) != 0) { fprintf(stderr, "decode failed at %d\n", pos); exit(1); }
            if (with_export) {
                const float * h = llama_get_embeddings_nextn(ctx);
                h_rows.emplace_back(h, h + n_embd_h);
            }
            cur = greedy(ctx);
            out.push_back(cur);
            pos++;
        }
        return out;
    };

    std::vector<llama_token> a = run_pass(false);
    std::vector<llama_token> b = run_pass(true);

    int div = -1;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        if (a[i] != b[i]) { div = (int) i; break; }
    }
    if (with_export) {
        double max_rel = 0.0; int worst = -1;
        for (size_t i = 0; i < h_rows_a.size() && i < h_rows_b.size(); ++i) {
            double num = 0.0, den = 0.0, na = 0.0, nb = 0.0;
            for (int e = 0; e < n_embd_h; ++e) {
                const double d = (double) h_rows_a[i][e] - (double) h_rows_b[i][e];
                num += d * d;
                den += (double) h_rows_a[i][e] * (double) h_rows_a[i][e];
                na  += (double) h_rows_a[i][e] * (double) h_rows_a[i][e];
                nb  += (double) h_rows_b[i][e] * (double) h_rows_b[i][e];
            }
            const double rel = den > 0 ? sqrt(num / den) : 0.0;
            if (rel > 1e-6) {
                printf("  step %2zu: h rel diff %.6f (|a|=%.1f |b|=%.1f)\n", i, rel, sqrt(na), sqrt(nb));
            }
            if (rel > max_rel) { max_rel = rel; worst = (int) i; }
        }
        printf("h rows compared: %zu, max relative L2 diff: %.6f (step %d)\n",
               h_rows_a.size(), max_rel, worst);
    }
    if (div < 0) {
        printf("IDENTICAL: %zu tokens\n", a.size());
    } else {
        printf("DIVERGED at token %d/%zu\n", div, a.size());
        for (int i = div; i < div + 5 && i < (int) a.size(); ++i) {
            char sa[64] = {0}, sb[64] = {0};
            llama_token_to_piece(vocab, a[i], sa, sizeof(sa) - 1, 0, true);
            llama_token_to_piece(vocab, b[i], sb, sizeof(sb) - 1, 0, true);
            printf("  [%d] clean=%d '%s'  rollback=%d '%s'\n", i, a[i], sa, b[i], sb);
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return div >= 0 ? 2 : 0;
}
