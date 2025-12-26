/**
 * mtmd-tts-gpu.h - GPU-accelerated TTS pipeline components for Qwen3-Omni
 *
 * This header provides GPU-optimized implementations of:
 *   1. Code Predictor (ggml graph-based, replaces CPU matmul)
 *   2. Batched token generation
 *   3. GPU embedding lookups
 *
 * These implementations eliminate CPU-GPU memory transfers by keeping
 * all intermediate states on GPU and using ggml operations.
 */

#ifndef MTMD_TTS_GPU_H
#define MTMD_TTS_GPU_H

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <vector>
#include <random>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// GPU Code Predictor Context
// =============================================================================

struct mtmd_code_predictor_gpu;

/**
 * Initialize GPU-accelerated Code Predictor
 *
 * This creates a reusable ggml computation graph for the Code Predictor
 * transformer. The graph is built once and reused for all 15 codebook
 * predictions, eliminating repeated weight copies.
 *
 * @param model      The Talker model containing Code Predictor weights
 * @param cpu_only   If true, use CPU backend (for testing/fallback)
 * @return           Opaque context pointer, or nullptr on failure
 */
struct mtmd_code_predictor_gpu * mtmd_code_predictor_gpu_init(
    const struct llama_model * model,
    bool cpu_only);

/**
 * Free GPU Code Predictor context
 */
void mtmd_code_predictor_gpu_free(struct mtmd_code_predictor_gpu * ctx);

/**
 * Run Code Predictor on GPU
 *
 * Generates 15 codebook tokens (codebooks 1-15) given:
 *   - past_hidden: hidden state from previous Talker step [n_embd]
 *   - last_id_hidden: embedding of last generated codec token [n_embd]
 *
 * @param ctx               GPU Code Predictor context
 * @param past_hidden       Hidden state from Talker [1024 floats]
 * @param last_id_hidden    Last token embedding [1024 floats]
 * @param codec_embeddings  Output: embeddings for each codebook [15 x 1024]
 * @param codebook_tokens   Output: sampled token IDs [15 ints]
 * @param temperature       Sampling temperature (0 = greedy)
 * @param rng               Random number generator for sampling
 * @return                  true on success
 */
bool mtmd_code_predictor_gpu_run(
    struct mtmd_code_predictor_gpu * ctx,
    const float * past_hidden,
    const float * last_id_hidden,
    std::vector<std::vector<float>> & codec_embeddings,
    std::vector<int> & codebook_tokens,
    float temperature,
    std::mt19937 & rng);

// =============================================================================
// GPU Embedding Lookup
// =============================================================================

struct mtmd_gpu_embedding_table;

/**
 * Initialize GPU embedding table
 *
 * Keeps the embedding table on GPU and provides efficient lookups
 * without copying the entire table to CPU.
 *
 * @param tensor     The embedding tensor (tok_embd or similar)
 * @param cpu_only   If true, use CPU backend
 * @return           Opaque context pointer
 */
struct mtmd_gpu_embedding_table * mtmd_gpu_embedding_init(
    const struct ggml_tensor * tensor,
    bool cpu_only);

/**
 * Free GPU embedding table
 */
void mtmd_gpu_embedding_free(struct mtmd_gpu_embedding_table * ctx);

/**
 * Look up embeddings for multiple tokens on GPU
 *
 * @param ctx        GPU embedding context
 * @param token_ids  Array of token IDs to look up
 * @param n_tokens   Number of tokens
 * @param output     Output buffer [n_tokens * n_embd floats]
 * @return           true on success
 */
bool mtmd_gpu_embedding_lookup(
    struct mtmd_gpu_embedding_table * ctx,
    const int * token_ids,
    int n_tokens,
    float * output);

// =============================================================================
// Batched Token Generation
// =============================================================================

struct mtmd_batched_generator;

/**
 * Parameters for batched generation
 */
struct mtmd_batched_params {
    int batch_size;        // Number of tokens to generate per batch (default: 4)
    float temperature;     // Sampling temperature
    int top_k;             // Top-k sampling
    float top_p;           // Top-p (nucleus) sampling
    float rep_penalty;     // Repetition penalty
    int max_tokens;        // Maximum tokens to generate
};

/**
 * Get default batched generation parameters
 */
struct mtmd_batched_params mtmd_batched_params_default(void);

/**
 * Initialize batched token generator
 *
 * This wraps the Talker model to generate multiple tokens per GPU call,
 * reducing CPU-GPU synchronization overhead.
 */
struct mtmd_batched_generator * mtmd_batched_generator_init(
    struct llama_context * talker_ctx,
    struct mtmd_code_predictor_gpu * code_pred,
    struct mtmd_batched_params params);

/**
 * Free batched generator
 */
void mtmd_batched_generator_free(struct mtmd_batched_generator * ctx);

#ifdef __cplusplus
}
#endif

#endif // MTMD_TTS_GPU_H
