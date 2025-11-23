// ABOUTME: FP4 Model inference test with token generation
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    printf("===============================================\n");
    printf("FP4 Model Token Generation Test (Blackwell GB10)\n");
    printf("===============================================\n\n");

    const char * model_path = "/models/llama-cpp/gpt-oss/gpt-oss-120b-mxfp4-00001-of-00003.gguf";
    if (argc > 1) {
        model_path = argv[1];
    }

    printf("Model: %s\n\n", model_path);

    // Load model with all layers on GPU
    printf("[1/3] Loading FP4 model...\n");
    auto t_start = std::chrono::high_resolution_clock::now();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 256;  // Load all layers on GPU
    mparams.main_gpu = 0;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "ERROR: Failed to load model\n");
        return 1;
    }

    auto t_load = std::chrono::high_resolution_clock::now();
    auto load_time = std::chrono::duration<double>(t_load - t_start).count();

    printf("✓ Model loaded successfully\n");
    printf("  Load time: %.2f seconds\n", load_time);
    printf("  Parameters: %lu\n", llama_model_n_params(model));
    printf("  Size: %.2f GB\n", llama_model_size(model) / (1024.0 * 1024.0 * 1024.0));
    printf("\n");

    // Get vocab from model
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!vocab) {
        fprintf(stderr, "ERROR: Failed to get vocab from model\n");
        llama_model_free(model);
        return 1;
    }

    // Create context
    printf("[2/3] Creating inference context...\n");
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;
    cparams.n_threads = 8;

    llama_context * ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "ERROR: Failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    printf("✓ Context created\n");
    printf("  Context size: %d tokens\n", cparams.n_ctx);
    printf("  Batch size: %d tokens\n", cparams.n_batch);
    printf("\n");

    // Run inference
    printf("[3/3] Running token generation...\n");
    const char * prompt = "Hello, how are you?";

    // Tokenize prompt
    std::vector<llama_token> prompt_tokens;
    prompt_tokens.resize(128);
    int n_prompt_tokens = llama_tokenize(vocab, prompt, (int)strlen(prompt), prompt_tokens.data(), (int)prompt_tokens.size(), true, true);
    if (n_prompt_tokens < 0) {
        fprintf(stderr, "ERROR: Failed to tokenize prompt\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    prompt_tokens.resize(n_prompt_tokens);

    printf("✓ Prompt tokenized: %d tokens\n", n_prompt_tokens);
    printf("  Prompt: \"%s\"\n", prompt);
    printf("\n");

    // Process initial prompt tokens
    printf("Processing %d prompt tokens...\n", n_prompt_tokens);
    for (int i = 0; i < n_prompt_tokens; i++) {
        llama_token token = prompt_tokens[i];
        llama_batch batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "ERROR: Failed to decode prompt token\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
    }

    // Create a greedy sampler
    llama_sampler * smpl = llama_sampler_init_greedy();
    if (!smpl) {
        fprintf(stderr, "ERROR: Failed to create sampler\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // Generate new tokens
    printf("Generating 64 tokens...\n");
    printf("---\n");

    auto t_gen_start = std::chrono::high_resolution_clock::now();
    int generated = 0;
    llama_token last_token = prompt_tokens.back();

    for (int i = 0; i < 64; i++) {
        // Sample next token
        llama_token next_token = llama_sampler_sample(smpl, ctx, -1);

        // Print token text
        const char * piece = llama_vocab_get_text(vocab, next_token);
        if (piece) {
            printf("%s", piece);
            fflush(stdout);
        }

        generated++;

        // Check for end of sequence
        if (llama_vocab_is_eog(vocab, next_token)) {
            printf("\n");
            break;
        }

        // Process next token
        llama_batch batch = llama_batch_get_one(&next_token, 1);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "ERROR: Failed to decode token\n");
            break;
        }

        // Reset sampler state for next iteration
        llama_sampler_reset(smpl);
    }

    auto t_gen_end = std::chrono::high_resolution_clock::now();
    auto gen_time = std::chrono::duration<double>(t_gen_end - t_gen_start).count();

    printf("\n---\n\n");
    printf("Results:\n");
    printf("  Tokens generated: %d\n", generated);
    printf("  Generation time: %.2f seconds\n", gen_time);
    if (gen_time > 0.0) {
        printf("  Speed: %.2f tokens/sec\n", (double)generated / gen_time);
    }
    printf("\n");

    // Cleanup
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    printf("===============================================\n");
    printf("✓ FP4 Model inference test completed\n");
    printf("===============================================\n");

    return 0;
}
