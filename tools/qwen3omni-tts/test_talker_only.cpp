// Minimal Talker test - loads only Talker model with pre-computed embeddings
// Build: g++ -o test_talker_only test_talker_only.cpp -I../../include -L../../build/bin -lllama -lggml

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <cmath>
#include "llama.h"

int main(int argc, char ** argv) {
    if (argc < 3) {
        printf("Usage: %s <talker.gguf> <prefill_embeds.bin>\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    const char * embeds_path = argv[2];

    // Load prefill embeddings from HF
    printf("Loading embeddings from %s...\n", embeds_path);
    std::ifstream fin(embeds_path, std::ios::binary);
    if (!fin.is_open()) {
        printf("Failed to open embeddings file\n");
        return 1;
    }

    // Read header: ndims, shape[0], shape[1]
    uint32_t ndims;
    fin.read(reinterpret_cast<char*>(&ndims), sizeof(uint32_t));
    std::vector<uint32_t> shape(ndims);
    for (uint32_t i = 0; i < ndims; ++i) {
        fin.read(reinterpret_cast<char*>(&shape[i]), sizeof(uint32_t));
    }

    int n_tokens = shape[0];
    int n_embd = shape[1];
    printf("Embeddings shape: [%d, %d]\n", n_tokens, n_embd);

    std::vector<float> embeddings(n_tokens * n_embd);
    fin.read(reinterpret_cast<char*>(embeddings.data()), n_tokens * n_embd * sizeof(float));
    fin.close();

    // Initialize llama
    llama_backend_init();

    // Load model
    printf("Loading Talker model from %s...\n", model_path);
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        printf("Failed to load model\n");
        return 1;
    }

    int model_n_embd = llama_model_n_embd(model);
    printf("Model n_embd: %d\n", model_n_embd);

    if (model_n_embd != n_embd) {
        printf("ERROR: Model n_embd (%d) != embeddings n_embd (%d)\n", model_n_embd, n_embd);
        llama_model_free(model);
        return 1;
    }

    // Create context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 64;
    cparams.embeddings = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        printf("Failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // Create batch with embeddings
    printf("Running prefill with %d tokens...\n", n_tokens);
    llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
    batch.n_tokens = n_tokens;
    memcpy(batch.embd, embeddings.data(), n_tokens * n_embd * sizeof(float));

    for (int i = 0; i < n_tokens; ++i) {
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
    }

    // Run decode
    int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        printf("llama_decode failed: %d\n", ret);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // Get logits
    float * logits = llama_get_logits(ctx);
    int n_vocab = llama_model_n_vocab(model);
    printf("Vocab size: %d\n", n_vocab);

    // Find top-10
    printf("\nTop-10 logits:\n");
    std::vector<std::pair<float, int>> logits_with_idx;
    for (int i = 0; i < n_vocab; ++i) {
        logits_with_idx.push_back({logits[i], i});
    }
    std::partial_sort(logits_with_idx.begin(), logits_with_idx.begin() + 10, logits_with_idx.end(),
                      [](auto a, auto b) { return a.first > b.first; });
    for (int i = 0; i < 10; ++i) {
        printf("  [%d] token=%d, logit=%.4f\n", i, logits_with_idx[i].second, logits_with_idx[i].first);
    }

    // Print logits stats
    float sum = 0, sum_sq = 0;
    for (int i = 0; i < n_vocab; ++i) {
        sum += logits[i];
        sum_sq += logits[i] * logits[i];
    }
    float mean = sum / n_vocab;
    float std_dev = std::sqrt(sum_sq / n_vocab - mean * mean);
    printf("\nLogits stats: mean=%.4f, std=%.4f\n", mean, std_dev);

    // Save logits
    std::ofstream fout("cpp_logits.bin", std::ios::binary);
    uint32_t out_ndims = 1;
    uint32_t out_dim0 = n_vocab;
    fout.write(reinterpret_cast<const char*>(&out_ndims), sizeof(uint32_t));
    fout.write(reinterpret_cast<const char*>(&out_dim0), sizeof(uint32_t));
    fout.write(reinterpret_cast<const char*>(logits), n_vocab * sizeof(float));
    fout.close();
    printf("Saved logits to cpp_logits.bin\n");

    // Compare with expected (HF greedy token = 1049)
    printf("\nExpected HF top token: 1049 (logit ~21.88)\n");
    printf("Actual C++ token 1049 logit: %.4f\n", logits[1049]);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
