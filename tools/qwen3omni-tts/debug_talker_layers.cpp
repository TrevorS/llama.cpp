/**
 * Debug Talker Layer Outputs
 *
 * Runs Talker prefill and extracts hidden states from each of the 20 transformer layers.
 * Saves outputs to files for comparison with HuggingFace.
 *
 * Usage:
 *   debug-talker-layers --model talker.gguf --prefill prefill_embeds.bin --output-dir /models/debug/cpp_talker
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <numeric>

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

// Internal headers for accessing graph
#include "llama-context.h"
#include "llama-graph.h"

// CUDA headers for error checking (when available)
#if defined(GGML_USE_CUDA)
#include <cuda_runtime.h>
#define CUDA_CHECK_ERROR(msg) do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error [%s]: %s\n", msg, cudaGetErrorString(err)); \
    } \
} while(0)
#define CUDA_SYNC_AND_CHECK(msg) do { \
    cudaError_t sync_err = cudaDeviceSynchronize(); \
    if (sync_err != cudaSuccess) { \
        fprintf(stderr, "CUDA Sync Error [%s]: %s\n", msg, cudaGetErrorString(sync_err)); \
    } \
    CUDA_CHECK_ERROR(msg); \
} while(0)
#else
#define CUDA_CHECK_ERROR(msg) ((void)0)
#define CUDA_SYNC_AND_CHECK(msg) ((void)0)
#endif

struct debug_params {
    std::string model_path;
    std::string prefill_path;
    std::string output_dir;
    int n_gpu_layers = 999;
    bool verbose = false;
};

static void print_usage(const char * prog) {
    printf("Usage: %s --model <talker.gguf> --prefill <prefill_embeds.bin> --output-dir <dir>\n", prog);
    printf("\n");
    printf("Options:\n");
    printf("  --model        Path to Talker GGUF model\n");
    printf("  --prefill      Path to prefill embeddings (binary format)\n");
    printf("  --output-dir   Directory to save layer outputs\n");
    printf("  --n-gpu-layers GPU layers (default: 999)\n");
    printf("  --verbose      Verbose output\n");
}

static bool parse_args(int argc, char ** argv, debug_params & params) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if (strcmp(argv[i], "--prefill") == 0 && i + 1 < argc) {
            params.prefill_path = argv[++i];
        } else if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            params.output_dir = argv[++i];
        } else if (strcmp(argv[i], "--n-gpu-layers") == 0 && i + 1 < argc) {
            params.n_gpu_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--verbose") == 0) {
            params.verbose = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return false;
        }
    }

    if (params.model_path.empty() || params.prefill_path.empty() || params.output_dir.empty()) {
        fprintf(stderr, "Error: --model, --prefill, and --output-dir are required\n");
        return false;
    }

    return true;
}

// Load embeddings from binary file
// Format: [ndims: uint32] [dim0: uint32] [dim1: uint32] ... [data: float32]
static std::vector<float> load_embeddings(const std::string & path, int & n_tokens, int & n_embd) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        fprintf(stderr, "Error: Cannot open embeddings file: %s\n", path.c_str());
        return {};
    }

    uint32_t ndims;
    fin.read(reinterpret_cast<char*>(&ndims), sizeof(uint32_t));

    std::vector<uint32_t> shape(ndims);
    for (uint32_t i = 0; i < ndims; ++i) {
        fin.read(reinterpret_cast<char*>(&shape[i]), sizeof(uint32_t));
    }

    if (ndims != 2) {
        fprintf(stderr, "Error: Expected 2D embeddings, got %u dims\n", ndims);
        return {};
    }

    n_tokens = shape[0];
    n_embd = shape[1];

    std::vector<float> data(n_tokens * n_embd);
    fin.read(reinterpret_cast<char*>(data.data()), n_tokens * n_embd * sizeof(float));

    return data;
}

// Save tensor to binary file (matching HuggingFace format)
// Format: [ndims: uint32] [dim0: uint32] ... [data: float32]
static bool save_tensor(const std::string & path, const float * data,
                        const std::vector<uint32_t> & shape) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Error: Cannot create file: %s\n", path.c_str());
        return false;
    }

    uint32_t ndims = shape.size();
    fwrite(&ndims, sizeof(uint32_t), 1, f);

    uint32_t n_elements = 1;
    for (uint32_t dim : shape) {
        fwrite(&dim, sizeof(uint32_t), 1, f);
        n_elements *= dim;
    }

    fwrite(data, sizeof(float), n_elements, f);
    fclose(f);

    return true;
}

int main(int argc, char ** argv) {
    debug_params params;
    if (!parse_args(argc, argv, params)) {
        print_usage(argv[0]);
        return 1;
    }

    // Load embeddings
    int n_tokens, n_embd;
    std::vector<float> embeddings = load_embeddings(params.prefill_path, n_tokens, n_embd);
    if (embeddings.empty()) {
        return 1;
    }
    printf("Loaded embeddings: [%d, %d]\n", n_tokens, n_embd);

    // Initialize llama - must load backends first
    ggml_backend_load_all();
    llama_backend_init();

    // Load model
    printf("Loading Talker model from %s...\n", params.model_path.c_str());
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = params.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(params.model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "Error: Failed to load model\n");
        return 1;
    }

    int model_n_embd = llama_model_n_embd(model);
    int model_n_embd_inp = llama_model_n_embd_inp(model);
    int model_n_layer = llama_model_n_layer(model);
    printf("Model: n_embd=%d, n_embd_inp=%d, n_layer=%d\n", model_n_embd, model_n_embd_inp, model_n_layer);

    if (model_n_embd != n_embd) {
        fprintf(stderr, "Error: Model n_embd (%d) != embeddings n_embd (%d)\n", model_n_embd, n_embd);
        llama_model_free(model);
        return 1;
    }

    if (model_n_embd_inp != n_embd) {
        fprintf(stderr, "Warning: Model n_embd_inp (%d) != embeddings n_embd (%d)\n", model_n_embd_inp, n_embd);
    }

    // Print input embedding statistics
    printf("\n=== Input Embeddings from File ===\n");
    {
        float sum = 0, sum_sq = 0;
        for (int i = 0; i < n_tokens * n_embd; ++i) {
            sum += embeddings[i];
            sum_sq += embeddings[i] * embeddings[i];
        }
        float mean = sum / (n_tokens * n_embd);
        float std_dev = sqrtf(sum_sq / (n_tokens * n_embd) - mean * mean);
        printf("  Stats: mean=%.6f, std=%.6f\n", mean, std_dev);
        printf("  First 5 values: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
               embeddings[0], embeddings[1], embeddings[2], embeddings[3], embeddings[4]);
        printf("  Token 0, last 5: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
               embeddings[n_embd-5], embeddings[n_embd-4], embeddings[n_embd-3], embeddings[n_embd-2], embeddings[n_embd-1]);
        printf("  Token 1, first 5: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
               embeddings[n_embd], embeddings[n_embd+1], embeddings[n_embd+2], embeddings[n_embd+3], embeddings[n_embd+4]);
    }

    // Create context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = n_tokens;
    cparams.embeddings = true;  // Enable hidden state output

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Error: Failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // Enable debug mode to preserve layer outputs (prevents buffer reuse)
    llama_set_debug_layer_outputs(ctx, true);

    // Create batch with embeddings
    printf("\n=== Creating batch ===\n");
    llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
    batch.n_tokens = n_tokens;
    printf("  Allocated batch.embd at %p, size=%d floats\n", (void*)batch.embd, n_tokens * n_embd);

    // Copy embeddings
    memcpy(batch.embd, embeddings.data(), n_tokens * n_embd * sizeof(float));

    // Verify the copy worked
    printf("\n=== Batch Embeddings after memcpy ===\n");
    {
        float sum = 0, sum_sq = 0;
        for (int i = 0; i < n_tokens * n_embd; ++i) {
            sum += batch.embd[i];
            sum_sq += batch.embd[i] * batch.embd[i];
        }
        float mean = sum / (n_tokens * n_embd);
        float std_dev = sqrtf(sum_sq / (n_tokens * n_embd) - mean * mean);
        printf("  Stats: mean=%.6f, std=%.6f\n", mean, std_dev);
        printf("  First 5 values: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
               batch.embd[0], batch.embd[1], batch.embd[2], batch.embd[3], batch.embd[4]);
    }

    for (int i = 0; i < n_tokens; ++i) {
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
    }

    printf("\nRunning prefill with %d tokens...\n", n_tokens);

    // Clear any pending CUDA errors before decode
    CUDA_CHECK_ERROR("before decode");

    // Run decode
    int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        fprintf(stderr, "Error: llama_decode failed: %d\n", ret);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // Check for CUDA errors after decode
    CUDA_SYNC_AND_CHECK("after llama_decode");

    printf("Decode complete. Extracting layer outputs...\n");

    // Access the graph result using the public getter
    llm_graph_result * res = ctx->get_gf_res_prev();
    if (!res) {
        fprintf(stderr, "Error: No graph result available\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    ggml_cgraph * gf = res->get_gf();
    if (!gf) {
        fprintf(stderr, "Error: No graph available\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    printf("Graph has %d nodes\n", ggml_graph_n_nodes(gf));

    // Check first layer's attention norm output (to see if something goes wrong early)
    {
        ggml_tensor * attn_norm = ggml_graph_get_tensor(gf, "attn_norm-0");
        if (attn_norm) {
            printf("\n=== Layer 0 Attn Norm Output ===\n");
            int64_t ne0 = attn_norm->ne[0];
            int64_t ne1 = attn_norm->ne[1];
            printf("  Shape: [%lld, %lld]\n", (long long)ne0, (long long)ne1);

            std::vector<float> norm_data(ne0 * ne1);
            ggml_backend_tensor_get(attn_norm, norm_data.data(), 0, ggml_nbytes(attn_norm));

            float sum = 0, sum_sq = 0;
            for (size_t i = 0; i < norm_data.size(); ++i) {
                sum += norm_data[i];
                sum_sq += norm_data[i] * norm_data[i];
            }
            float mean = sum / norm_data.size();
            float std_dev = sqrtf(sum_sq / norm_data.size() - mean * mean);
            printf("  Stats: mean=%.6f, std=%.6f\n", mean, std_dev);
            printf("  First 5: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                   norm_data[0], norm_data[1], norm_data[2], norm_data[3], norm_data[4]);

            // Check for infinities or NaNs
            bool has_inf = false, has_nan = false;
            for (size_t i = 0; i < norm_data.size(); ++i) {
                if (std::isinf(norm_data[i])) has_inf = true;
                if (std::isnan(norm_data[i])) has_nan = true;
            }
            if (has_inf) printf("  WARNING: Contains infinity values!\n");
            if (has_nan) printf("  WARNING: Contains NaN values!\n");
        } else {
            printf("\n=== Layer 0 Attn Norm Output ===\n");
            printf("  NOT FOUND (attn_norm-0)\n");
        }
    }

    // Extract each layer's output
    printf("\n=== Per-Layer Outputs ===\n");
    std::vector<float> layer_data;
    std::vector<int> problematic_layers;  // Track layers with issues
    for (int il = 0; il < model_n_layer; ++il) {
        char tensor_name[64];
        snprintf(tensor_name, sizeof(tensor_name), "l_out-%d", il);

        ggml_tensor * layer_tensor = ggml_graph_get_tensor(gf, tensor_name);
        if (!layer_tensor) {
            fprintf(stderr, "Warning: Tensor '%s' not found in graph\n", tensor_name);
            continue;
        }

        // Get tensor dimensions
        // ggml uses reversed dimensions: [n_embd, n_tokens] in ggml = [n_tokens, n_embd] in PyTorch
        int64_t ne0 = layer_tensor->ne[0];  // n_embd (1024)
        int64_t ne1 = layer_tensor->ne[1];  // n_tokens (9)
        size_t n_bytes = ggml_nbytes(layer_tensor);

        if (params.verbose) {
            printf("  Layer %d: ne=[%lld, %lld], bytes=%zu, data=%p, backend=%s\n",
                   il, (long long)ne0, (long long)ne1, n_bytes,
                   (void*)layer_tensor->data,
                   layer_tensor->buffer ? ggml_backend_buffer_name(layer_tensor->buffer) : "null");
        }

        // Allocate buffer and extract data
        layer_data.resize(ne0 * ne1);
        ggml_backend_tensor_get(layer_tensor, layer_data.data(), 0, n_bytes);

        // Save to file
        // Note: Data is in ggml format [n_embd, n_tokens], need to match HF format [n_tokens, n_embd]
        // HF saves as [n_tokens, n_embd], so we need to transpose
        std::vector<float> transposed(ne0 * ne1);
        for (int64_t t = 0; t < ne1; ++t) {
            for (int64_t e = 0; e < ne0; ++e) {
                // ggml layout: data[e + t * ne0]
                // HF layout: data[t * ne0 + e] (same if we just interpret dimensions differently)
                transposed[t * ne0 + e] = layer_data[e + t * ne0];
            }
        }

        std::string out_path = params.output_dir + "/hidden_layer" + std::to_string(il) + ".bin";
        std::vector<uint32_t> shape = {(uint32_t)ne1, (uint32_t)ne0};  // [n_tokens, n_embd]
        save_tensor(out_path, transposed.data(), shape);

        // Print stats and check for NaN/Inf
        float sum = 0, sum_sq = 0;
        int n_nan = 0, n_inf = 0, n_zero = 0;
        float abs_max = 0;
        for (size_t i = 0; i < transposed.size(); ++i) {
            float v = transposed[i];
            if (std::isnan(v)) { n_nan++; continue; }
            if (std::isinf(v)) { n_inf++; continue; }
            if (v == 0.0f) n_zero++;
            sum += v;
            sum_sq += v * v;
            abs_max = std::max(abs_max, std::fabs(v));
        }
        size_t valid = transposed.size() - n_nan - n_inf;
        float mean = valid > 0 ? sum / valid : 0;
        float std_dev = valid > 0 ? sqrtf(sum_sq / valid - mean * mean) : 0;

        // Print with warning indicators for problematic layers
        const char * status = "";
        bool is_problematic = false;
        if (n_nan > 0 || n_inf > 0) {
            status = " [!!! NaN/Inf !!!]";
            is_problematic = true;
        } else if (std_dev < 0.01f) {
            status = " [WARNING: near-zero std]";
            is_problematic = true;
        } else if (n_zero > (int)(transposed.size() * 0.9)) {
            status = " [WARNING: >90% zeros]";
            is_problematic = true;
        }
        if (is_problematic) {
            problematic_layers.push_back(il);
        }
        printf("  Layer %2d: mean=%10.6f, std=%10.6f, max=%10.4f, NaN=%d, Inf=%d, zeros=%d%s\n",
               il, mean, std_dev, abs_max, n_nan, n_inf, n_zero, status);

        if (params.verbose) {
            printf("           first=[%.4f, %.4f], saved to %s\n",
                   transposed[0], transposed[1], out_path.c_str());
        }

        // Check for CUDA errors after each layer extraction
        CUDA_CHECK_ERROR("after layer extraction");
    }

    // Also save final hidden state (after output norm)
    {
        ggml_tensor * norm_tensor = ggml_graph_get_tensor(gf, "result_norm");
        if (norm_tensor) {
            int64_t ne0 = norm_tensor->ne[0];
            int64_t ne1 = norm_tensor->ne[1];
            size_t n_bytes = ggml_nbytes(norm_tensor);

            layer_data.resize(ne0 * ne1);
            ggml_backend_tensor_get(norm_tensor, layer_data.data(), 0, n_bytes);

            // Transpose for HF format
            std::vector<float> transposed(ne0 * ne1);
            for (int64_t t = 0; t < ne1; ++t) {
                for (int64_t e = 0; e < ne0; ++e) {
                    transposed[t * ne0 + e] = layer_data[e + t * ne0];
                }
            }

            std::string out_path = params.output_dir + "/hidden_after_norm.bin";
            std::vector<uint32_t> shape = {(uint32_t)ne1, (uint32_t)ne0};
            save_tensor(out_path, transposed.data(), shape);

            float sum = 0, sum_sq = 0;
            for (size_t i = 0; i < transposed.size(); ++i) {
                sum += transposed[i];
                sum_sq += transposed[i] * transposed[i];
            }
            float mean = sum / transposed.size();
            float std_dev = sqrtf(sum_sq / transposed.size() - mean * mean);
            printf("  Norm out: mean=%.6f, std=%.6f, first=[%.4f, %.4f], saved to %s\n",
                   mean, std_dev, transposed[0], transposed[1], out_path.c_str());
        }
    }

    // First check if inp_embd matches what we passed
    {
        ggml_tensor * inp_tensor = ggml_graph_get_tensor(gf, "inp_embd");
        if (inp_tensor) {
            int64_t ne0 = inp_tensor->ne[0];
            int64_t ne1 = inp_tensor->ne[1];
            size_t n_bytes = ggml_nbytes(inp_tensor);

            printf("\n=== Input Embeddings Check ===\n");
            printf("  inp_embd tensor: ne=[%lld, %lld], bytes=%zu, buffer=%s\n",
                   (long long)ne0, (long long)ne1, n_bytes,
                   inp_tensor->buffer ? ggml_backend_buffer_name(inp_tensor->buffer) : "null");

            std::vector<float> inp_data(ne0 * ne1);
            ggml_backend_tensor_get(inp_tensor, inp_data.data(), 0, n_bytes);

            // Compare with original embeddings
            float sum = 0, sum_sq = 0, diff_sum = 0;
            for (size_t i = 0; i < inp_data.size(); ++i) {
                sum += inp_data[i];
                sum_sq += inp_data[i] * inp_data[i];
                diff_sum += std::abs(inp_data[i] - embeddings[i]);
            }
            float mean = sum / inp_data.size();
            float std_dev = sqrtf(sum_sq / inp_data.size() - mean * mean);
            float mean_diff = diff_sum / inp_data.size();

            printf("  inp_embd stats: mean=%.6f, std=%.6f\n", mean, std_dev);
            printf("  Original stats: mean=%.6f, std=%.6f\n",
                   std::accumulate(embeddings.begin(), embeddings.end(), 0.0f) / embeddings.size(),
                   sqrtf(std::inner_product(embeddings.begin(), embeddings.end(), embeddings.begin(), 0.0f) / embeddings.size() -
                         std::pow(std::accumulate(embeddings.begin(), embeddings.end(), 0.0f) / embeddings.size(), 2)));
            printf("  Mean abs diff from original: %.6f\n", mean_diff);

            // Print first few values from both sources
            printf("\n  First 5 from inp_embd (raw GGML layout): [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                   inp_data[0], inp_data[1], inp_data[2], inp_data[3], inp_data[4]);
            printf("  First 5 from original file:              [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                   embeddings[0], embeddings[1], embeddings[2], embeddings[3], embeddings[4]);

            // Note: GGML tensor [n_embd, n_tokens] stores data with n_embd stride 1
            // So first n_embd values are token 0, next n_embd are token 1, etc.
            // This should match row-major [n_tokens, n_embd] from the file
            printf("\n  Token 0 comparison (first 5 dims):\n");
            printf("    inp_embd:  [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                   inp_data[0], inp_data[1], inp_data[2], inp_data[3], inp_data[4]);
            printf("    original:  [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                   embeddings[0], embeddings[1], embeddings[2], embeddings[3], embeddings[4]);

            printf("\n  Token 1 comparison (first 5 dims):\n");
            printf("    inp_embd:  [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                   inp_data[ne0], inp_data[ne0+1], inp_data[ne0+2], inp_data[ne0+3], inp_data[ne0+4]);
            printf("    original:  [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                   embeddings[n_embd], embeddings[n_embd+1], embeddings[n_embd+2], embeddings[n_embd+3], embeddings[n_embd+4]);

            // Check if maybe it's transposed
            printf("\n  Check if transposed (inp_embd[i*ne0] vs original[i]):\n");
            printf("    inp_embd[0*%lld]: %.6f, original[0]: %.6f\n", (long long)ne0, inp_data[0], embeddings[0]);
            printf("    inp_embd[1*%lld]: %.6f, original[1]: %.6f\n", (long long)ne0, inp_data[ne0], embeddings[1]);
            printf("    inp_embd[2*%lld]: %.6f, original[2]: %.6f\n", (long long)ne0, inp_data[2*ne0], embeddings[2]);

            // Save inp_embd for comparison
            std::string out_path = params.output_dir + "/debug_inp_embd.bin";
            std::vector<uint32_t> shape = {(uint32_t)ne1, (uint32_t)ne0};
            save_tensor(out_path, inp_data.data(), shape);
            printf("\n  Saved to %s\n", out_path.c_str());
        } else {
            printf("\n=== Input Embeddings Check ===\n");
            printf("  inp_embd: NOT FOUND in graph\n");
        }
    }

    // Check l_out-0 vs hidden_layer0 to verify extraction is correct
    {
        ggml_tensor * l_out = ggml_graph_get_tensor(gf, "l_out-0");
        if (l_out) {
            int64_t ne0 = l_out->ne[0];
            int64_t ne1 = l_out->ne[1];
            size_t n_bytes = ggml_nbytes(l_out);

            std::vector<float> l_out_data(ne0 * ne1);
            ggml_backend_tensor_get(l_out, l_out_data.data(), 0, n_bytes);

            printf("\n=== Layer 0 Output Check ===\n");
            printf("  l_out-0: ne=[%lld, %lld]\n", (long long)ne0, (long long)ne1);
            printf("    first 10: [");
            for (int i = 0; i < std::min(10, (int)l_out_data.size()); ++i) {
                printf("%.4f%s", l_out_data[i], i < 9 ? ", " : "");
            }
            printf("]\n");

            // Compare with what we saved as hidden_layer0
            float sum = 0, sum_sq = 0;
            for (size_t i = 0; i < l_out_data.size(); ++i) {
                sum += l_out_data[i];
                sum_sq += l_out_data[i] * l_out_data[i];
            }
            float mean = sum / l_out_data.size();
            float std_dev = sqrtf(sum_sq / l_out_data.size() - mean * mean);
            printf("    mean=%.6f, std=%.6f\n", mean, std_dev);
        }
    }

    // Extract raw norm-0 (before weight multiplication)
    {
        ggml_tensor * norm_tensor = ggml_graph_get_tensor(gf, "norm-0");
        if (norm_tensor) {
            int64_t ne0 = norm_tensor->ne[0];
            int64_t ne1 = norm_tensor->ne[1];
            size_t n_bytes = ggml_nbytes(norm_tensor);

            printf("\n=== Raw RMSNorm Output (before weight) ===\n");
            printf("  norm-0: ne=[%lld, %lld], bytes=%zu\n", (long long)ne0, (long long)ne1, n_bytes);

            std::vector<float> norm_data(ne0 * ne1);
            ggml_backend_tensor_get(norm_tensor, norm_data.data(), 0, n_bytes);

            float sum = 0, sum_sq = 0;
            int n_nan = 0, n_inf = 0;
            for (size_t i = 0; i < norm_data.size(); ++i) {
                if (std::isnan(norm_data[i])) n_nan++;
                else if (std::isinf(norm_data[i])) n_inf++;
                else {
                    sum += norm_data[i];
                    sum_sq += norm_data[i] * norm_data[i];
                }
            }
            size_t valid = norm_data.size() - n_nan - n_inf;
            float mean = valid > 0 ? sum / valid : 0;
            float std_dev = valid > 0 ? sqrtf(sum_sq / valid - mean * mean) : 0;

            printf("    stats: mean=%.6f, std=%.6f, NaN=%d, Inf=%d\n", mean, std_dev, n_nan, n_inf);
            printf("    first 5: [");
            for (int i = 0; i < std::min(5, (int)norm_data.size()); ++i) {
                printf("%.4f%s", norm_data[i], i < 4 ? ", " : "");
            }
            printf("]\n");
        } else {
            printf("\n=== Raw RMSNorm Output (before weight) ===\n");
            printf("  norm-0: NOT FOUND\n");
        }
    }

    // Extract attention tensors for Layer 0 to debug QKV
    printf("\n=== Layer 0 Attention Tensors ===\n");
    const char * attn_names[] = {"attn_norm", "Qcur", "Kcur", "Vcur", "Qcur_normed", "Kcur_normed"};
    for (const char * name : attn_names) {
        char tensor_name[64];
        snprintf(tensor_name, sizeof(tensor_name), "%s-%d", name, 0);

        ggml_tensor * tensor = ggml_graph_get_tensor(gf, tensor_name);
        if (!tensor) {
            printf("  %s: NOT FOUND\n", name);
            continue;
        }

        // Handle different dimensionalities
        int64_t ne0 = tensor->ne[0];
        int64_t ne1 = tensor->ne[1];
        int64_t ne2 = tensor->ne[2];
        int64_t ne3 = tensor->ne[3];
        size_t n_bytes = ggml_nbytes(tensor);

        printf("  %s: ne=[%lld, %lld, %lld, %lld], bytes=%zu\n",
               name, (long long)ne0, (long long)ne1, (long long)ne2, (long long)ne3, n_bytes);

        std::vector<float> data(n_bytes / sizeof(float));
        ggml_backend_tensor_get(tensor, data.data(), 0, n_bytes);

        // Save to file - keep ggml order, will handle in Python
        std::string out_path = params.output_dir + "/debug_" + std::string(name) + "_layer0.bin";

        // Determine shape based on dimensionality
        std::vector<uint32_t> shape;
        if (ne3 > 1) {
            shape = {(uint32_t)ne3, (uint32_t)ne2, (uint32_t)ne1, (uint32_t)ne0};
        } else if (ne2 > 1) {
            shape = {(uint32_t)ne2, (uint32_t)ne1, (uint32_t)ne0};
        } else {
            shape = {(uint32_t)ne1, (uint32_t)ne0};
        }
        save_tensor(out_path, data.data(), shape);

        // Print stats
        float sum = 0, sum_sq = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            sum += data[i];
            sum_sq += data[i] * data[i];
        }
        float mean = sum / data.size();
        float std_dev = sqrtf(sum_sq / data.size() - mean * mean);
        printf("    mean=%.6f, std=%.6f, saved to %s\n", mean, std_dev, out_path.c_str());
    }

    // Extract intermediate tensors for layers around collapse point
    printf("\n=== Intermediate tensors for layers 7-10 (collapse zone) ===\n");
    const char * intermediate_names[] = {"ffn_inp", "ffn_norm", "ffn_moe_logits", "ffn_moe_probs", "ffn_moe_out", "ffn_shexp", "ffn_moe_shexp_out", "ffn_residual"};
    for (int il = 7; il <= 10; ++il) {
        printf("Layer %d intermediates:\n", il);
        for (const char * name : intermediate_names) {
            char tensor_name[64];
            snprintf(tensor_name, sizeof(tensor_name), "%s-%d", name, il);

            ggml_tensor * tensor = ggml_graph_get_tensor(gf, tensor_name);
            if (!tensor) {
                continue;  // Skip if not found (e.g., ffn_shexp may not exist)
            }

            int64_t ne0 = tensor->ne[0];
            int64_t ne1 = tensor->ne[1];
            size_t n_bytes = ggml_nbytes(tensor);

            layer_data.resize(ne0 * ne1);
            ggml_backend_tensor_get(tensor, layer_data.data(), 0, n_bytes);

            // Transpose for HF format
            std::vector<float> transposed(ne0 * ne1);
            for (int64_t t = 0; t < ne1; ++t) {
                for (int64_t e = 0; e < ne0; ++e) {
                    transposed[t * ne0 + e] = layer_data[e + t * ne0];
                }
            }

            // Save to file
            std::string out_path = params.output_dir + "/" + std::string(name) + "_layer" + std::to_string(il) + ".bin";
            std::vector<uint32_t> shape = {(uint32_t)ne1, (uint32_t)ne0};
            save_tensor(out_path, transposed.data(), shape);

            // Print stats
            float sum = 0, sum_sq = 0;
            for (size_t i = 0; i < transposed.size(); ++i) {
                sum += transposed[i];
                sum_sq += transposed[i] * transposed[i];
            }
            float mean = sum / transposed.size();
            float std_dev = sqrtf(sum_sq / transposed.size() - mean * mean);
            printf("  %s: mean=%.6f, std=%.6f, saved to %s\n", name, mean, std_dev, out_path.c_str());
        }
    }

    // Print logits top-5
    {
        float * logits = llama_get_logits(ctx);
        const struct llama_vocab * vocab = llama_model_get_vocab(model);
        int n_vocab = llama_vocab_n_tokens(vocab);
        if (logits && n_vocab > 0) {
            std::vector<std::pair<float, int>> logits_with_idx;
            for (int i = 0; i < n_vocab; ++i) {
                logits_with_idx.push_back({logits[i], i});
            }
            std::partial_sort(logits_with_idx.begin(), logits_with_idx.begin() + 5, logits_with_idx.end(),
                              [](auto a, auto b) { return a.first > b.first; });

            printf("\nTop-5 logits:\n");
            for (int i = 0; i < 5; ++i) {
                printf("  [%d] token=%d, logit=%.4f\n", i, logits_with_idx[i].second, logits_with_idx[i].first);
            }
        }
    }

    // Print summary
    printf("\n=== Layer extraction complete ===\n");
    printf("Output directory: %s\n", params.output_dir.c_str());

    // Summary of problematic layers
    if (!problematic_layers.empty()) {
        printf("\n*** WARNING: Problematic layers detected ***\n");
        printf("Layers with issues: ");
        for (size_t i = 0; i < problematic_layers.size(); ++i) {
            printf("%d%s", problematic_layers[i],
                   (i < problematic_layers.size() - 1) ? ", " : "\n");
        }
        printf("\nThis may indicate a CUDA bug or numerical instability.\n");
        printf("Try running with --n-gpu-layers 0 to verify CPU produces correct output.\n");
    } else {
        printf("\nAll layers produced valid output.\n");
    }

    // Final CUDA error check
    CUDA_SYNC_AND_CHECK("end of program");

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
