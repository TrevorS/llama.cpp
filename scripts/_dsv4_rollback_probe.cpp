// Probe: does evict+re-decode perturb DSV4 target state?
// Pass A: straight greedy decode of N tokens.
// Pass B: same, but before each accepted token, decode a junk token at the
//         same position, llama_memory_seq_rm it, then decode the real token.
// Lossless rollback => identical token streams.
#include "llama.h"
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

        std::vector<llama_token> out;
        llama_token cur = greedy(ctx);
        out.push_back(cur);
        llama_pos pos = n_p;
        for (int t = 0; t < n_gen - 1; ++t) {
            if (with_rollback) {
                // decode a junk token at this position, then evict it
                if (decode_one(ctx, junk, pos, true) != 0) { fprintf(stderr, "junk decode failed at %d\n", pos); exit(1); }
                if (!llama_memory_seq_rm(mem, 0, pos, -1)) { fprintf(stderr, "seq_rm failed at %d\n", pos); exit(1); }
            }
            if (decode_one(ctx, cur, pos, true) != 0) { fprintf(stderr, "decode failed at %d\n", pos); exit(1); }
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
