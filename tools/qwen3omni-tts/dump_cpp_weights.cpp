// Dump C++ loaded weights for comparison with HF
// Build: cmake --build build --target dump_cpp_weights

#include <cstdio>
#include <fstream>
#include "llama.h"

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("Usage: %s <talker.gguf>\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];

    llama_backend_init();

    printf("Loading Talker model from %s...\n", model_path);
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // CPU only to read weights
    mparams.use_mmap = true;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        printf("Failed to load model\n");
        return 1;
    }

    printf("Model loaded successfully\n");
    printf("n_embd: %d\n", llama_model_n_embd(model));
    printf("n_vocab: %d\n", llama_model_n_vocab(model));
    printf("n_layer: %d\n", llama_model_n_layer(model));

    // Dump weights to file for comparison
    printf("\n=== Dumping weights ===\n");

    // Access tensors via ggml backend
    // This is hacky but works for debugging
    struct ggml_context * ctx = nullptr;

    // Try to get tensor by name using llama API (if available)
    // For now, just print model stats
    printf("Model loaded. Use llama-gguf tool to inspect tensors.\n");

    llama_model_free(model);
    llama_backend_free();

    return 0;
}
