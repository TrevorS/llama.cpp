/**
 * Qwen3-Omni TTS Pipeline Tool
 *
 * End-to-end TTS: Text -> Thinker (layer 18) -> TextProj -> Talker -> Code Predictor -> Code2Wav -> WAV
 *
 * Usage:
 *   llama-qwen3omni-tts --thinker thinker.gguf --talker talker.gguf -p "Hello world" -o output.wav
 *
 * The pipeline:
 *   1. Tokenize input text
 *   2. Run Thinker (48 layers) but extract hidden states at layer 18
 *   3. Apply text projection MLP (2048 -> 1024)
 *   4. Talker generates audio codec tokens autoregressively
 *   5. Code Predictor expands to multi-codebook (1 -> 16 codebooks)
 *   6. Code2Wav synthesizes waveform from codec tokens
 *   7. Write output WAV file
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <set>

#include "llama.h"
#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"

// Internal headers for model tensor access and graph extraction
#include "llama-model.h"
#include "llama-context.h"
#include "llama-graph.h"

struct tts_params {
    std::string thinker_path;
    std::string talker_path;
    std::string prompt;
    std::string output_path = "output.wav";
    std::string dump_tokens_path;   // Debug: dump codec tokens to file for HF comparison
    std::string load_tokens_path;   // Debug: load 16 codebook tokens from file (skip Talker)
    std::string dump_tensors_path;  // Debug: dump intermediate tensors to directory
    int sample_rate = 24000;
    int n_layer_output = 24;      // Extract from Thinker at this layer (accept_hidden_layer)
    int n_gpu_layers = 99;
    int max_codec_tokens = 500;   // Maximum codec tokens to generate
    float temperature = 0.9f;   // Match HuggingFace talker_temperature
    int top_k = 50;              // Match HuggingFace talker_top_k
    int seed = -1;               // Random seed (-1 for random)
    bool verbose = false;
    bool thinker_only = false;    // Test mode: only run Thinker extraction
    bool skip_code2wav = false;   // Skip Code2Wav for faster testing
    bool use_mmap = true;         // Use mmap for model loading (disable for UMA systems)
    bool code2wav_only = false;   // Internal: skip Thinker when --load-tokens provided
    bool c2w_cpu_only = false;    // Force CPU for Code2Wav (workaround for CUDA IM2COL issues)
};

// Special tokens for Talker (from talker_config in config.json)
// Fix #26: Use correct codec special token IDs from model config
static const int TALKER_CODEC_PAD_ID = 2148;      // codec_pad_id
static const int TALKER_CODEC_BOS_ID = 2149;      // codec_bos_id
static const int TALKER_CODEC_EOS_ID = 2150;      // codec_eos_token_id - CORRECT EOS!
static const int TALKER_CODEC_NOTHINK_ID = 2155;  // codec_nothink_id
static const int TALKER_CODEC_THINK_BOS_ID = 2156; // codec_think_bos_id
static const int TALKER_CODEC_THINK_EOS_ID = 2157; // codec_think_eos_id

// Check if token is an EOS token that should end generation
static bool is_talker_eos(int token) {
    // Fix #26: Use correct codec_eos_token_id from config
    return token == TALKER_CODEC_EOS_ID || token == TALKER_CODEC_THINK_EOS_ID;
}

// Write WAV file header
static void write_wav_header(FILE * f, int sample_rate, int num_samples) {
    // RIFF header
    fwrite("RIFF", 1, 4, f);
    uint32_t file_size = 36 + num_samples * sizeof(int16_t);
    fwrite(&file_size, sizeof(file_size), 1, f);
    fwrite("WAVE", 1, 4, f);

    // fmt chunk
    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    fwrite(&fmt_size, sizeof(fmt_size), 1, f);
    uint16_t audio_format = 1;  // PCM
    fwrite(&audio_format, sizeof(audio_format), 1, f);
    uint16_t num_channels = 1;  // Mono
    fwrite(&num_channels, sizeof(num_channels), 1, f);
    uint32_t sr = sample_rate;
    fwrite(&sr, sizeof(sr), 1, f);
    uint32_t byte_rate = sample_rate * sizeof(int16_t);
    fwrite(&byte_rate, sizeof(byte_rate), 1, f);
    uint16_t block_align = sizeof(int16_t);
    fwrite(&block_align, sizeof(block_align), 1, f);
    uint16_t bits_per_sample = 16;
    fwrite(&bits_per_sample, sizeof(bits_per_sample), 1, f);

    // data chunk
    fwrite("data", 1, 4, f);
    uint32_t data_size = num_samples * sizeof(int16_t);
    fwrite(&data_size, sizeof(data_size), 1, f);
}

// Write audio samples to WAV file
static bool write_wav(const std::string & path, const float * samples, int num_samples, int sample_rate) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Error: Cannot create WAV file: %s\n", path.c_str());
        return false;
    }

    write_wav_header(f, sample_rate, num_samples);

    // Convert float [-1, 1] to int16
    for (int i = 0; i < num_samples; ++i) {
        float s = samples[i];
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        int16_t sample = static_cast<int16_t>(s * 32767.0f);
        fwrite(&sample, sizeof(sample), 1, f);
    }

    fclose(f);
    return true;
}

// Save embeddings to binary file for debugging/comparison
static bool save_embeddings(const std::string & path, const float * data, int seq_len, int hidden_dim) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "Error: Cannot create embeddings file: %s\n", path.c_str());
        return false;
    }

    uint32_t sl = seq_len;
    uint32_t hd = hidden_dim;
    f.write(reinterpret_cast<const char*>(&sl), sizeof(sl));
    f.write(reinterpret_cast<const char*>(&hd), sizeof(hd));
    f.write(reinterpret_cast<const char*>(data), seq_len * hidden_dim * sizeof(float));

    return f.good();
}

// Save tensor to binary file matching HuggingFace format
// Format: [num_dims: uint32] [dim0: uint32] [dim1: uint32] ... [data: float32]
static bool save_tensor_hf_format(const std::string & path, const float * data,
                                   const std::vector<uint32_t> & shape) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Error: Cannot create tensor file: %s\n", path.c_str());
        return false;
    }

    uint32_t num_dims = shape.size();
    fwrite(&num_dims, sizeof(uint32_t), 1, f);

    uint32_t num_elements = 1;
    for (uint32_t dim : shape) {
        fwrite(&dim, sizeof(uint32_t), 1, f);
        num_elements *= dim;
    }

    fwrite(data, sizeof(float), num_elements, f);
    fclose(f);

    printf("  Saved %s: shape [", path.c_str());
    for (size_t i = 0; i < shape.size(); ++i) {
        printf("%u%s", shape[i], i < shape.size() - 1 ? ", " : "");
    }
    printf("], %u bytes\n", (unsigned)(sizeof(uint32_t) * (1 + shape.size()) + num_elements * sizeof(float)));

    return true;
}

// Load codebook tokens from file for debugging Code2Wav
// File format: 16 lines, one per codebook, space-separated token IDs
// Returns: vector of frames, each frame has 16 codebook tokens
static std::vector<std::vector<int>> load_codebook_tokens(const std::string & path) {
    std::vector<std::vector<int>> result;

    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "Error: Cannot open token file: %s\n", path.c_str());
        return result;
    }

    // Read 16 codebook lines
    std::vector<std::vector<int>> codebook_tokens(16);
    std::string line;
    int cb = 0;
    while (std::getline(f, line) && cb < 16) {
        std::istringstream iss(line);
        int token;
        while (iss >> token) {
            codebook_tokens[cb].push_back(token);
        }
        if (!codebook_tokens[cb].empty()) {
            cb++;
        }
    }

    if (cb != 16) {
        fprintf(stderr, "Error: Token file has %d codebook lines, expected 16\n", cb);
        return result;
    }

    // Verify all codebooks have same length
    size_t n_frames = codebook_tokens[0].size();
    for (int i = 1; i < 16; ++i) {
        if (codebook_tokens[i].size() != n_frames) {
            fprintf(stderr, "Error: Codebook %d has %zu tokens, expected %zu\n",
                    i, codebook_tokens[i].size(), n_frames);
            return result;
        }
    }

    // Transpose: [16 codebooks, n_frames] -> [n_frames, 16 codebooks]
    result.resize(n_frames);
    for (size_t frame = 0; frame < n_frames; ++frame) {
        result[frame].resize(16);
        for (int c = 0; c < 16; ++c) {
            result[frame][c] = codebook_tokens[c][frame];
        }
    }

    printf("Loaded %zu frames × 16 codebooks from %s\n", n_frames, path.c_str());
    return result;
}

// Sample from logits with temperature, top-k, top-p, and repetition penalty
// Excludes tokens 2301-2303 which have abnormally high embedding norms
static int sample_token(const float * logits, int n_vocab, float temperature, int top_k,
                        std::mt19937 & rng, const std::vector<int> & recent_tokens = {},
                        float rep_penalty = 1.1f, float top_p = 0.9f) {
    // Greedy sampling for temperature <= 0
    if (temperature <= 0.0f) {
        float max_logit = -1e30f;
        int best_token = 0;
        for (int i = 0; i < n_vocab; ++i) {
            // Same token filtering as stochastic path
            if (i >= 2048 && i != TALKER_CODEC_EOS_ID && i != TALKER_CODEC_THINK_EOS_ID) {
                continue;
            }
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                best_token = i;
            }
        }
        return best_token;
    }

    std::vector<std::pair<float, int>> logits_sorted;
    logits_sorted.reserve(n_vocab);

    // Build set of recent tokens for O(1) lookup
    std::set<int> recent_set(recent_tokens.begin(), recent_tokens.end());

    for (int i = 0; i < n_vocab; ++i) {
        // Fix #38: Force audio token generation (0-2047) by suppressing all special tokens
        // except EOS (2150) which is needed for proper termination.
        // The model was outputting speaker tokens (2301-2303) instead of audio.
        // This is more aggressive than HuggingFace but necessary without proper conditioning.
        if (i >= 2048 && i != TALKER_CODEC_EOS_ID && i != TALKER_CODEC_THINK_EOS_ID) {
            continue;  // Skip all special tokens except EOS
        }

        float logit = logits[i];
        // Apply repetition penalty
        if (recent_set.count(i) > 0) {
            logit = (logit > 0) ? logit / rep_penalty : logit * rep_penalty;
        }
        logits_sorted.push_back({logit, i});
    }

    // Sort by logit value descending
    std::sort(logits_sorted.begin(), logits_sorted.end(),
        [](const auto & a, const auto & b) { return a.first > b.first; }
    );

    // Apply temperature and compute softmax
    float max_logit = logits_sorted[0].first;
    std::vector<float> probs(logits_sorted.size());
    float sum = 0.0f;

    for (size_t i = 0; i < logits_sorted.size(); ++i) {
        float p = expf((logits_sorted[i].first - max_logit) / temperature);
        probs[i] = p;
        sum += p;
    }

    // Normalize
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum;
    }

    // Apply top-p (nucleus) sampling
    float cumsum = 0.0f;
    size_t top_p_cutoff = probs.size();
    for (size_t i = 0; i < probs.size(); ++i) {
        cumsum += probs[i];
        if (cumsum >= top_p) {
            top_p_cutoff = i + 1;
            break;
        }
    }

    // Apply top-k cutoff
    size_t actual_k = std::min({(size_t)top_k, top_p_cutoff, probs.size()});

    // Renormalize over selected candidates
    sum = 0.0f;
    for (size_t i = 0; i < actual_k; ++i) {
        sum += probs[i];
    }
    for (size_t i = 0; i < actual_k; ++i) {
        probs[i] /= sum;
    }

    // Sample
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    cumsum = 0.0f;

    for (size_t i = 0; i < actual_k; ++i) {
        cumsum += probs[i];
        if (r <= cumsum) {
            return logits_sorted[i].second;
        }
    }

    return logits_sorted[0].second;  // Fallback to top token
}

// SiLU activation function (Swish): x * sigmoid(x)
// Used by Qwen3-Omni text projection MLP
static float silu(float x) {
    return x / (1.0f + expf(-x));
}

// GELU activation function (approximation) - kept for Code2Wav compatibility
static float gelu(float x) {
    // Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    const float sqrt_2_over_pi = 0.7978845608f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// RMSNorm: x / sqrt(mean(x^2) + eps) * weight
static void rms_norm(const float * x, const float * weight, float * out, int n, float eps = 1e-6f) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / n + eps);
    for (int i = 0; i < n; ++i) {
        out[i] = (x[i] / rms) * weight[i];
    }
}

// Rotary Position Embedding (RoPE) for Code Predictor
// Reference: HuggingFace apply_rotary_pos_emb in modeling_qwen3_omni_moe.py
// Applies rotation to Q or K tensor for a single head
static void apply_rope_to_head(
    float * qk,         // [head_dim] - modified in place
    int head_dim,
    int pos,
    float rope_theta) {

    const int half_dim = head_dim / 2;

    for (int i = 0; i < half_dim; ++i) {
        // Compute frequency for this dimension
        float freq = 1.0f / powf(rope_theta, 2.0f * i / head_dim);
        float angle = pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        // Apply rotation to pairs [i, i + half_dim]
        float x0 = qk[i];
        float x1 = qk[i + half_dim];
        qk[i] = x0 * cos_val - x1 * sin_val;
        qk[i + half_dim] = x0 * sin_val + x1 * cos_val;
    }
}

// Apply RoPE to all heads in Q or K tensor
static void apply_rope(
    float * qk,         // [n_head, head_dim] - modified in place
    int n_head,
    int head_dim,
    int pos,
    float rope_theta) {

    for (int h = 0; h < n_head; ++h) {
        apply_rope_to_head(qk + h * head_dim, head_dim, pos, rope_theta);
    }
}

// Copy tensor data to CPU buffer, handling different types and GPU memory
static bool copy_tensor_to_cpu(const ggml_tensor * t, std::vector<float> & out) {
    if (!t) return false;

    int64_t n_elem = ggml_nelements(t);
    out.resize(n_elem);

    // Check if tensor is on GPU (buffer is not null and not CPU)
    if (t->buffer) {
        // Use ggml_backend_tensor_get to copy from any backend to CPU
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(t, out.data(), 0, n_elem * sizeof(float));
            return true;
        } else if (t->type == GGML_TYPE_F16) {
            // For F16, first copy raw bytes then convert
            std::vector<ggml_fp16_t> tmp(n_elem);
            ggml_backend_tensor_get(t, tmp.data(), 0, n_elem * sizeof(ggml_fp16_t));
            for (int64_t i = 0; i < n_elem; ++i) {
                out[i] = ggml_fp16_to_fp32(tmp[i]);
            }
            return true;
        }
        return false;
    }

    // Tensor is on CPU, direct access
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), t->data, n_elem * sizeof(float));
        return true;
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = (const ggml_fp16_t *)t->data;
        for (int64_t i = 0; i < n_elem; ++i) {
            out[i] = ggml_fp16_to_fp32(src[i]);
        }
        return true;
    }
    return false;
}

// Extract pre-norm hidden state from Talker graph
// HuggingFace uses hidden_states[0][-1] which is BEFORE output_norm, but llama_get_embeddings()
// returns AFTER output_norm. This function extracts the pre-norm tensor directly from the graph.
static bool extract_pre_norm_hidden(llama_context * ctx, float * out, int n_embd) {
    // CRITICAL: Synchronize to ensure GPU computation is complete and data is available
    llama_synchronize(ctx);

    // Get the graph result from the last decode
    llm_graph_result * res = ctx->get_gf_res_prev();
    if (!res) {
        fprintf(stderr, "Warning: No graph result available for pre_norm extraction\n");
        return false;
    }

    ggml_cgraph * gf = res->get_gf();
    if (!gf) {
        fprintf(stderr, "Warning: No graph available for pre_norm extraction\n");
        return false;
    }

    // Find the pre_norm_hidden tensor by name
    ggml_tensor * pre_norm = ggml_graph_get_tensor(gf, "pre_norm_hidden");
    if (!pre_norm) {
        fprintf(stderr, "Warning: pre_norm_hidden tensor not found in graph\n");
        return false;
    }

    // Debug: print tensor info including tensor name
    fprintf(stderr, "DEBUG extract_pre_norm_hidden: tensor '%s' shape=[%lld,%lld,%lld,%lld], type=%d\n",
            pre_norm->name,
            (long long)pre_norm->ne[0], (long long)pre_norm->ne[1],
            (long long)pre_norm->ne[2], (long long)pre_norm->ne[3],
            (int)pre_norm->type);

    // Copy the tensor data to CPU
    std::vector<float> temp;
    if (!copy_tensor_to_cpu(pre_norm, temp)) {
        fprintf(stderr, "Warning: Failed to copy pre_norm_hidden tensor to CPU\n");
        return false;
    }

    // Get the last token's hidden state (ne[1] is sequence length)
    int64_t seq_len = pre_norm->ne[1];
    int64_t hidden_dim = pre_norm->ne[0];
    if (hidden_dim != n_embd) {
        fprintf(stderr, "Warning: pre_norm_hidden dim mismatch: %lld vs %d\n",
                (long long)hidden_dim, n_embd);
        return false;
    }

    // Debug: print first and last token values to compare
    fprintf(stderr, "DEBUG: seq_len=%lld, hidden_dim=%lld, total elements=%zu\n",
            (long long)seq_len, (long long)hidden_dim, temp.size());
    fprintf(stderr, "DEBUG: First token [0:8]: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
            temp[0], temp[1], temp[2], temp[3], temp[4], temp[5], temp[6], temp[7]);
    if (seq_len > 1) {
        int64_t last_offset = (seq_len - 1) * hidden_dim;
        fprintf(stderr, "DEBUG: Last token [0:8]: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
                temp[last_offset], temp[last_offset + 1], temp[last_offset + 2], temp[last_offset + 3],
                temp[last_offset + 4], temp[last_offset + 5], temp[last_offset + 6], temp[last_offset + 7]);
    }

    // Compute stats for debugging
    float sum_sq = 0.0f, min_val = temp[0], max_val = temp[0];
    for (size_t i = 0; i < temp.size(); ++i) {
        sum_sq += temp[i] * temp[i];
        if (temp[i] < min_val) min_val = temp[i];
        if (temp[i] > max_val) max_val = temp[i];
    }
    float rms = sqrtf(sum_sq / temp.size());
    float l2_norm = sqrtf(sum_sq);
    fprintf(stderr, "DEBUG: L2 norm=%.4f, RMS=%.4f, min=%.4f, max=%.4f\n", l2_norm, rms, min_val, max_val);
    fprintf(stderr, "DEBUG: tensor buffer=%p, data=%p\n", (void*)pre_norm->buffer, pre_norm->data);

    // Copy the last token's hidden state
    int64_t last_token_offset = (seq_len - 1) * hidden_dim;
    memcpy(out, temp.data() + last_token_offset, n_embd * sizeof(float));

    // Debug: Compute L2 of JUST the extracted last token
    float out_l2 = 0.0f;
    for (int i = 0; i < n_embd; ++i) {
        out_l2 += out[i] * out[i];
    }
    out_l2 = sqrtf(out_l2);
    fprintf(stderr, "DEBUG: Extracted last token L2=%.4f, first 4: [%.4f, %.4f, %.4f, %.4f]\n",
            out_l2, out[0], out[1], out[2], out[3]);

    return true;
}

// Matrix multiplication: out[m,n] = a[m,k] @ b[k,n]^T (b stored as [n,k])
static void matmul(const float * a, const float * b, float * out,
                   int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += a[i * k + l] * b[j * k + l];
            }
            out[i * n + j] = sum;
        }
    }
}

// Add bias to matrix
static void add_bias(float * x, const float * bias, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            x[i * cols + j] += bias[j];
        }
    }
}

// SwiGLU FFN: out = (silu(gate(x)) * up(x)) @ down
static void swiglu_ffn(const float * x, int n_embd, int n_ff,
                       const float * gate_w, const float * up_w, const float * down_w,
                       float * out, float * scratch) {
    // gate = x @ gate_w^T
    // up = x @ up_w^T
    std::vector<float> gate(n_ff), up(n_ff);
    matmul(x, gate_w, gate.data(), 1, n_embd, n_ff);
    matmul(x, up_w, up.data(), 1, n_embd, n_ff);

    // Apply SiLU to gate and multiply with up
    for (int i = 0; i < n_ff; ++i) {
        scratch[i] = silu(gate[i]) * up[i];
    }

    // down = scratch @ down_w^T
    matmul(scratch, down_w, out, 1, n_ff, n_embd);
}

// Simple attention for single token (no KV cache, used for Code Predictor)
// For single token, attention is just the value scaled (self-attention with single token)
static void simple_self_attention_single_token(
    const float * x, int n_embd, int n_head, int n_head_kv, int head_dim,
    const float * wq, const float * wk, const float * wv, const float * wo,
    const float * q_norm_w, const float * k_norm_w,
    float * out) {

    int n_embd_head = head_dim;
    (void)n_embd_head;
    int n_kv_head = n_head_kv;
    int gqa_ratio = n_head / n_head_kv;

    // Project Q, K, V
    std::vector<float> q(n_head * head_dim), k(n_kv_head * head_dim), v(n_kv_head * head_dim);
    matmul(x, wq, q.data(), 1, n_embd, n_head * head_dim);
    matmul(x, wk, k.data(), 1, n_embd, n_kv_head * head_dim);
    matmul(x, wv, v.data(), 1, n_embd, n_kv_head * head_dim);

    // Apply QK normalization per-head
    std::vector<float> q_normed(n_head * head_dim), k_normed(n_kv_head * head_dim);
    for (int h = 0; h < n_head; ++h) {
        rms_norm(q.data() + h * head_dim, q_norm_w, q_normed.data() + h * head_dim, head_dim);
    }
    for (int h = 0; h < n_kv_head; ++h) {
        rms_norm(k.data() + h * head_dim, k_norm_w, k_normed.data() + h * head_dim, head_dim);
    }

    // For single token: attention output is just value (score=1, softmax(1)=1)
    // With GQA, each query head uses corresponding KV head
    std::vector<float> attn_out(n_head * head_dim);
    for (int h = 0; h < n_head; ++h) {
        int kv_h = h / gqa_ratio;
        // For single token, attention score is q.k / sqrt(d), softmax(score) = 1
        // So attention output is just v
        for (int d = 0; d < head_dim; ++d) {
            attn_out[h * head_dim + d] = v[kv_h * head_dim + d];
        }
    }

    // Project output
    matmul(attn_out.data(), wo, out, 1, n_head * head_dim, n_embd);
}

// Causal multi-head attention with KV cache for Code Predictor
// Processes a query at position `pos` attending to all KV cache entries 0..pos
// kv_cache_k: [max_seq, n_kv_head, head_dim] - K values for all past positions
// kv_cache_v: [max_seq, n_kv_head, head_dim] - V values for all past positions
// rope_theta: RoPE base frequency (1000000 for Code Predictor)
static void causal_attention_with_kv_cache(
    const float * x, int n_embd, int n_head, int n_head_kv, int head_dim, int pos,
    const float * wq, const float * wk, const float * wv, const float * wo,
    const float * q_norm_w, const float * k_norm_w,
    float rope_theta,  // RoPE base frequency
    float * kv_cache_k, float * kv_cache_v,  // KV cache, modified in place
    float * out) {

    int gqa_ratio = n_head / n_head_kv;
    float scale = 1.0f / sqrtf((float)head_dim);

    // Project Q, K, V for current position
    std::vector<float> q(n_head * head_dim), k(n_head_kv * head_dim), v(n_head_kv * head_dim);
    matmul(x, wq, q.data(), 1, n_embd, n_head * head_dim);
    matmul(x, wk, k.data(), 1, n_embd, n_head_kv * head_dim);
    matmul(x, wv, v.data(), 1, n_embd, n_head_kv * head_dim);

    // Apply QK normalization per-head
    std::vector<float> q_normed(n_head * head_dim), k_normed(n_head_kv * head_dim);
    for (int h = 0; h < n_head; ++h) {
        rms_norm(q.data() + h * head_dim, q_norm_w, q_normed.data() + h * head_dim, head_dim);
    }
    for (int h = 0; h < n_head_kv; ++h) {
        rms_norm(k.data() + h * head_dim, k_norm_w, k_normed.data() + h * head_dim, head_dim);
    }

    // Apply RoPE (Rotary Position Embedding) to Q and K after normalization
    // Reference: HuggingFace modeling_qwen3_omni_moe.py:2376
    // query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    apply_rope(q_normed.data(), n_head, head_dim, pos, rope_theta);
    apply_rope(k_normed.data(), n_head_kv, head_dim, pos, rope_theta);

    // Store K, V in cache at position `pos`
    // Cache layout: [pos, kv_head, head_dim]
    for (int h = 0; h < n_head_kv; ++h) {
        for (int d = 0; d < head_dim; ++d) {
            kv_cache_k[pos * n_head_kv * head_dim + h * head_dim + d] = k_normed[h * head_dim + d];
            kv_cache_v[pos * n_head_kv * head_dim + h * head_dim + d] = v[h * head_dim + d];
        }
    }

    // Compute attention for each query head
    std::vector<float> attn_out(n_head * head_dim, 0.0f);
    for (int qh = 0; qh < n_head; ++qh) {
        int kv_h = qh / gqa_ratio;

        // Compute attention scores for all positions 0..pos
        std::vector<float> scores(pos + 1);
        float max_score = -1e30f;
        for (int p = 0; p <= pos; ++p) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q_normed[qh * head_dim + d] *
                         kv_cache_k[p * n_head_kv * head_dim + kv_h * head_dim + d];
            }
            score *= scale;
            scores[p] = score;
            if (score > max_score) max_score = score;
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int p = 0; p <= pos; ++p) {
            scores[p] = expf(scores[p] - max_score);
            sum_exp += scores[p];
        }
        for (int p = 0; p <= pos; ++p) {
            scores[p] /= sum_exp;
        }

        // Weighted sum of values
        for (int d = 0; d < head_dim; ++d) {
            float acc = 0.0f;
            for (int p = 0; p <= pos; ++p) {
                acc += scores[p] * kv_cache_v[p * n_head_kv * head_dim + kv_h * head_dim + d];
            }
            attn_out[qh * head_dim + d] = acc;
        }
    }

    // Project output
    matmul(attn_out.data(), wo, out, 1, n_head * head_dim, n_embd);
}

// Snake activation (SnakeBeta): f(x) = x + (1/(exp(beta) + ε)) * sin^2(x * exp(alpha))
// Reference: HuggingFace SnakeBeta from modeling_qwen3_omni_moe.py:3668-3680
// IMPORTANT: alpha/beta are stored as log-scale, must exponentiate before use
// IMPORTANT: ε = 1e-9 (no_div_by_zero) is added to prevent division by zero
// alpha/beta are per-channel
static void snake_activation(const float * x, const float * alpha, const float * beta,
                              float * out, int seq_len, int n_channels) {
    const float eps = 1e-9f;
    for (int s = 0; s < seq_len; ++s) {
        for (int c = 0; c < n_channels; ++c) {
            int idx = s * n_channels + c;
            // Exponentiate alpha and beta (they're stored as log-scale)
            float exp_alpha = (alpha && alpha[c] != 0.0f) ? expf(alpha[c]) : 1.0f;
            float exp_beta = (beta && beta[c] != 0.0f) ? expf(beta[c]) : 1.0f;
            // SnakeBeta: x + (1/beta) * sin²(x * alpha)
            float sin_val = sinf(x[idx] * exp_alpha);
            out[idx] = x[idx] + (1.0f / (exp_beta + eps)) * sin_val * sin_val;
        }
    }
}

// 1D convolution with same padding: out[seq_out, c_out] = conv(x[seq_in, c_in], w[k, c_in, c_out])
static void conv1d_same(const float * x, const float * w,
                        float * out, int seq_len, int c_in, int c_out, int kernel_size) {
    int pad = kernel_size / 2;
    for (int s = 0; s < seq_len; ++s) {
        for (int co = 0; co < c_out; ++co) {
            float sum = 0.0f;
            for (int k = 0; k < kernel_size; ++k) {
                int si = s - pad + k;
                if (si >= 0 && si < seq_len) {
                    for (int ci = 0; ci < c_in; ++ci) {
                        sum += x[si * c_in + ci] * w[k * c_in * c_out + ci * c_out + co];
                    }
                }
            }
            out[s * c_out + co] = sum;
        }
    }
}

// 1D transposed convolution for upsampling
static void conv1d_transpose(const float * x, const float * w,
                             float * out, int seq_in, int c_in, int c_out,
                             int kernel_size, int stride) {
    int seq_out = seq_in * stride;
    // Zero output
    memset(out, 0, seq_out * c_out * sizeof(float));

    // Transposed conv: scatter each input to multiple outputs
    for (int si = 0; si < seq_in; ++si) {
        for (int k = 0; k < kernel_size; ++k) {
            int so = si * stride + k;
            if (so < seq_out) {
                for (int co = 0; co < c_out; ++co) {
                    for (int ci = 0; ci < c_in; ++ci) {
                        out[so * c_out + co] += x[si * c_in + ci] *
                            w[k * c_in * c_out + ci * c_out + co];
                    }
                }
            }
        }
    }
}

// =============================================================================
// Code2Wav Vocoder - ggml Graph-Based Implementation
// =============================================================================
//
// Architecture:
//   1. Codebook embedding sum (16 codebooks → 1024 dim)
//   2. Pre-transformer: 8 layers with LayerScale
//   3. ConvNeXt upsample: 2 blocks (4× total)
//   4. HiFi-GAN decoder: 4 stages (480× total) with Snake activations
//   5. Final conv + tanh → mono audio

// Code2Wav context for ggml graph execution
struct code2wav_context {
    ggml_backend_t backend_cpu;
    std::vector<ggml_backend_t> backends;
    std::vector<ggml_backend_buffer_type_t> backend_bufts;
    ggml_backend_sched_t sched;
    std::vector<uint8_t> buf_compute_meta;
    int max_nodes;

    code2wav_context() : backend_cpu(nullptr), sched(nullptr), max_nodes(32768) {}
    ~code2wav_context() {
        if (sched) {
            ggml_backend_sched_free(sched);
        }
        // Free ALL backends we created (GPU backend was being leaked!)
        for (auto backend : backends) {
            if (backend && backend != backend_cpu) {
                ggml_backend_free(backend);
            }
        }
        if (backend_cpu) {
            ggml_backend_free(backend_cpu);
        }
    }
};

// Initialize Code2Wav context
static bool init_code2wav_context(code2wav_context * ctx, const llama_model * model, bool cpu_only = false) {
    // Get backends from model - the tensors are on GPU, we need that backend
    // Try to get GPU backend first (where model tensors live)
    if (!cpu_only) {
        ggml_backend_t backend_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (backend_gpu) {
            ctx->backends.push_back(backend_gpu);
            ctx->backend_bufts.push_back(ggml_backend_get_default_buffer_type(backend_gpu));
            fprintf(stderr, "Code2Wav: using GPU backend (%s)\n", ggml_backend_name(backend_gpu));
        }
    } else {
        fprintf(stderr, "Code2Wav: CPU-only mode (--c2w-cpu)\n");
    }

    // Get CPU backend as fallback
    ctx->backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!ctx->backend_cpu) {
        fprintf(stderr, "Error: Failed to initialize CPU backend\n");
        return false;
    }
    ctx->backends.push_back(ctx->backend_cpu);
    ctx->backend_bufts.push_back(ggml_backend_get_default_buffer_type(ctx->backend_cpu));

    // Create scheduler with all backends
    ctx->sched = ggml_backend_sched_new(
        ctx->backends.data(),
        ctx->backend_bufts.data(),
        ctx->backends.size(),
        ctx->max_nodes,
        false,  // parallel
        true    // op_offload
    );

    if (!ctx->sched) {
        fprintf(stderr, "Error: Failed to create backend scheduler\n");
        return false;
    }

    // Allocate compute meta buffer
    ctx->buf_compute_meta.resize(ctx->max_nodes * ggml_tensor_overhead() + ggml_graph_overhead());

    (void)model;  // Model used indirectly through tensor buffers
    return true;
}

// Build Snake activation (SnakeBeta): f(x) = x + (1/(exp(β) + ε)) * sin²(x * exp(α))
// Reference: HuggingFace SnakeBeta from modeling_qwen3_omni_moe.py:3668-3680
// IMPORTANT: alpha/beta are stored as log-scale, must exponentiate before use
// IMPORTANT: ε = 1e-9 (no_div_by_zero) is added to exp(β) to match HuggingFace exactly
// x shape: [channels, seq], alpha/beta shape: [channels]
// Need to reshape alpha/beta to [channels, 1] for proper broadcasting
static ggml_tensor * build_snake(ggml_context * ctx, ggml_tensor * x,
                                  ggml_tensor * alpha, ggml_tensor * beta) {
    if (!alpha) return x;

    // DEBUG: Try simpler Snake without ggml_repeat (use manual broadcasting via ggml_mul)
    // The issue might be with ggml_repeat producing NaN

    // Exponentiate alpha and beta (they're stored in log-scale)
    ggml_tensor * exp_alpha = ggml_exp(ctx, alpha);
    ggml_tensor * exp_beta = beta ? ggml_exp(ctx, beta) : nullptr;

    // Use direct multiply which should broadcast [channels] to [channels, seq] automatically
    // sin(x * exp(α))
    ggml_tensor * scaled_x = ggml_mul(ctx, x, exp_alpha);
    ggml_tensor * sin_val = ggml_sin(ctx, scaled_x);

    // sin²(x * exp(α))
    ggml_tensor * sin2 = ggml_sqr(ctx, sin_val);

    // (1/(exp(β) + epsilon)) * sin²(...)
    // Match HuggingFace exactly: beta + no_div_by_zero where no_div_by_zero = 1e-9
    ggml_tensor * term;
    if (exp_beta) {
        // Add epsilon (1e-9) to prevent division by zero - matches HF exactly
        // Use ggml_scale_bias: result = 1.0 * exp_beta + 1e-9
        ggml_tensor * beta_safe = ggml_scale_bias(ctx, exp_beta, 1.0f, 1e-9f);
        term = ggml_div(ctx, sin2, beta_safe);
    } else {
        term = sin2;
    }

    // x + (1/(exp(β) + ε)) * sin²(x * exp(α))
    return ggml_add(ctx, x, term);
}

// Build RMSNorm: x / sqrt(mean(x^2) + eps) * weight
// Code2Wav uses eps=1e-5 (from HF config.rms_norm_eps), other models use 1e-6
static constexpr float C2W_RMS_NORM_EPS = 1e-5f;

// Code2Wav pre-transformer sliding window attention size (from HF config.sliding_window)
static constexpr int C2W_SLIDING_WINDOW = 72;

static ggml_tensor * build_rms_norm(ggml_context * ctx, ggml_tensor * x,
                                     ggml_tensor * weight, float eps = 1e-6f) {
    x = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, x, weight);
}

// NOTE: Removed build_sliding_window_mask() - was writing to NULL tensor data during graph construction
// Use ggml_diag_mask_inf() instead which is a proper GGML graph operation

// Build dilated 1D convolution with proper symmetric padding
// For dilated convs, effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1)
// Symmetric padding = effective_kernel / 2
// Input: [seq, channels], Output: [seq, out_channels]
static ggml_tensor * build_dilated_conv1d(ggml_context * ctx, ggml_tensor * kernel,
                                           ggml_tensor * input, int dilation) {
    // kernel shape: [kernel_size, in_channels, out_channels] in GGML
    int kernel_size = kernel->ne[0];
    int effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1);
    int padding = effective_kernel / 2;

    // Use ggml_conv_1d with explicit padding and dilation
    return ggml_conv_1d(ctx, kernel, input, 1, padding, dilation);
}

// Fix #17: Build causal 1D convolution with left-only padding
// For causal conv: effective_kernel = kernel + (kernel-1) * (dilation-1)
// Causal padding = effective_kernel - stride (all on LEFT)
// Input: [seq, channels], Output: [seq, out_channels]
static ggml_tensor * build_causal_conv1d(ggml_context * ctx, ggml_tensor * kernel,
                                          ggml_tensor * input, int dilation) {
    int kernel_size = kernel->ne[0];
    int effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1);
    int causal_pad = effective_kernel - 1;  // stride=1, so pad = effective_kernel - 1

    // Pad LEFT side only: ggml_pad_ext(lp0, rp0, lp1, rp1, ...)
    // Input is [seq, channels], so pad dimension 0 (seq)
    ggml_tensor * padded = ggml_pad_ext(ctx, input, causal_pad, 0, 0, 0, 0, 0, 0, 0);

    // Conv with padding=0 since we already padded
    return ggml_conv_1d(ctx, kernel, padded, 1, 0, dilation);
}

// Debug: Check tensor for NaN/Inf and print stats including sequence variance
static void debug_check_tensor(const char * name, ggml_backend_buffer_t buf, ggml_tensor * tensor) {
    int64_t n = ggml_nelements(tensor);
    std::vector<float> data(n);
    ggml_backend_tensor_get(tensor, data.data(), 0, n * sizeof(float));

    int nan_count = 0, inf_count = 0;
    float min_val = data[0], max_val = data[0], sum = 0;
    for (int64_t i = 0; i < n; i++) {
        if (std::isnan(data[i])) nan_count++;
        else if (std::isinf(data[i])) inf_count++;
        else {
            if (data[i] < min_val) min_val = data[i];
            if (data[i] > max_val) max_val = data[i];
            sum += data[i];
        }
    }

    // Compute variance along sequence dimension to check if signal varies
    // Tensor is [channels, seq], so variance across seq positions
    int64_t channels = tensor->ne[0];
    int64_t seq_len = tensor->ne[1];
    float seq_var = 0;
    if (seq_len > 1 && channels > 0) {
        // For each channel, compute variance across sequence
        for (int64_t c = 0; c < std::min(channels, (int64_t)10); c++) {  // Sample first 10 channels
            float ch_sum = 0, ch_sum_sq = 0;
            for (int64_t s = 0; s < seq_len; s++) {
                float v = data[c + s * channels];
                ch_sum += v;
                ch_sum_sq += v * v;
            }
            float ch_mean = ch_sum / seq_len;
            float ch_var = ch_sum_sq / seq_len - ch_mean * ch_mean;
            seq_var += ch_var;
        }
        seq_var /= std::min(channels, (int64_t)10);
    }

    float nan_pct = 100.0f * nan_count / n;
    printf("  DEBUG %s [%ld, %ld]: min=%.4f, max=%.4f, seq_var=%.6f, NaN=%.1f%%\n",
           name, (long)channels, (long)seq_len,
           min_val, max_val, seq_var, nan_pct);
    (void)buf;  // silence unused warning
}

// Build Code2Wav compute graph
static ggml_tensor * build_code2wav_graph(
        ggml_context * ctx,
        ggml_cgraph * gf,
        const llama_model * model,
        ggml_tensor * input_embd,  // [c2w_n_embd, n_frames]
        int n_frames,
        bool verbose,
        std::vector<ggml_tensor *> * debug_tensors = nullptr,
        ggml_tensor ** out_attn_mask = nullptr) {

    const int c2w_n_embd = 1024;
    const int c2w_n_head = 16;
    const int c2w_head_dim = 64;
    const int c2w_n_layer = 8;
    const int c2w_up_n_block = 2;
    const int c2w_dec_dim = 1536;
    const int upsample_rates[] = {8, 5, 4, 3};
    const int c2w_dec_n_stage = 4;
    const int c2w_dec_n_resblk = 3;

    ggml_tensor * cur = input_embd;
    int seq_len = n_frames;

    // Build sliding window causal mask for pre-transformer (window=72)
    // Mask is [seq, seq] with 0.0 for valid positions, filled later before compute
    ggml_tensor * c2w_attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_len, seq_len);
    ggml_set_name(c2w_attn_mask, "c2w_attn_mask");
    ggml_set_input(c2w_attn_mask);
    if (out_attn_mask) {
        *out_attn_mask = c2w_attn_mask;
    }

    // =========================================================================
    // Pre-transformer: 8 layers with sliding window causal attention
    // =========================================================================
    if (verbose) {
        printf("  Building pre-transformer (%d layers)...\n", c2w_n_layer);
    }

    for (int il = 0; il < c2w_n_layer && il < (int)model->c2w_pre_layers.size(); ++il) {
        const auto & layer = model->c2w_pre_layers[il];
        ggml_tensor * inpSA = cur;

        // Debug: Track pre-transformer layer input (first layer only)
        if (debug_tensors && il == 0) {
            ggml_set_name(cur, "pretrans_layer0_input");
            ggml_set_output(cur);
            debug_tensors->push_back(cur);
        }

        // Attention norm (RMSNorm with Code2Wav epsilon)
        if (layer.attn_norm) {
            cur = build_rms_norm(ctx, cur, layer.attn_norm, C2W_RMS_NORM_EPS);
            // Debug: Track after attn norm (layer 0 and 7)
            if (debug_tensors && (il == 0 || il == 7)) {
                char name[64];
                snprintf(name, sizeof(name), "pretrans_layer%d_after_attn_norm", il);
                ggml_set_name(cur, name);
                ggml_set_output(cur);
                debug_tensors->push_back(cur);
            }
        }

        // Self-attention (simplified single-head for compatibility)
        // Use basic attention without flash_attn to avoid mask issues
        if (layer.wq && layer.wk && layer.wv && layer.wo) {
            // Q/K/V projections: cur is [n_embd, seq]
            ggml_tensor * Qcur = ggml_mul_mat(ctx, layer.wq, cur);  // [n_embd, seq]
            ggml_tensor * Kcur = ggml_mul_mat(ctx, layer.wk, cur);
            ggml_tensor * Vcur = ggml_mul_mat(ctx, layer.wv, cur);

            // Reshape for multi-head: [n_embd, seq] -> [head_dim, n_head, seq]
            Qcur = ggml_reshape_3d(ctx, Qcur, c2w_head_dim, c2w_n_head, seq_len);
            Kcur = ggml_reshape_3d(ctx, Kcur, c2w_head_dim, c2w_n_head, seq_len);
            Vcur = ggml_reshape_3d(ctx, Vcur, c2w_head_dim, c2w_n_head, seq_len);

            // Apply RoPE (rotary position embeddings) to all heads
            // Note: HuggingFace Code2Wav applies RoPE to all heads (not just head 0)
            ggml_tensor * pos_f32 = ggml_arange(ctx, 0, seq_len, 1);
            ggml_tensor * pos = ggml_cast(ctx, pos_f32, GGML_TYPE_I32);
            ggml_set_name(pos, "rope_pos");

            Qcur = ggml_rope_ext(ctx, Qcur, pos, nullptr, c2w_head_dim, GGML_ROPE_TYPE_NEOX,
                                 2048, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            Kcur = ggml_rope_ext(ctx, Kcur, pos, nullptr, c2w_head_dim, GGML_ROPE_TYPE_NEOX,
                                 2048, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            // Permute for attention: [head_dim, n_head, seq] -> [head_dim, seq, n_head]
            Qcur = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
            Kcur = ggml_permute(ctx, Kcur, 0, 2, 1, 3);
            Vcur = ggml_permute(ctx, Vcur, 0, 2, 1, 3);
            Qcur = ggml_cont(ctx, Qcur);
            Kcur = ggml_cont(ctx, Kcur);
            Vcur = ggml_cont(ctx, Vcur);

            // Compute attention scores: Q @ K^T
            // Q: [head_dim, seq, n_head], K: [head_dim, seq, n_head]
            // KQ = mul_mat(K, Q) gives [seq, seq, n_head]
            ggml_tensor * KQ = ggml_mul_mat(ctx, Kcur, Qcur);

            // Scale + sliding window mask + softmax (combined for efficiency)
            // Sliding window mask limits attention to past 72 tokens (matching HF)
            float scale = 1.0f / sqrtf((float)c2w_head_dim);
            KQ = ggml_soft_max_ext(ctx, KQ, c2w_attn_mask, scale, 0.0f);

            // Attention output: KQ @ V
            // KQ: [seq, seq, n_head], V: [head_dim, seq, n_head]
            // For each head: [seq, seq] @ [head_dim, seq]^T = [head_dim, seq]
            // Need V transposed: [seq, head_dim, n_head]
            ggml_tensor * Vt = ggml_permute(ctx, Vcur, 1, 0, 2, 3);
            Vt = ggml_cont(ctx, Vt);
            // KQV = mul_mat(Vt, KQ) = [head_dim, seq, n_head]
            ggml_tensor * KQV = ggml_mul_mat(ctx, Vt, KQ);

            // Permute back: [head_dim, seq, n_head] -> [head_dim, n_head, seq]
            KQV = ggml_permute(ctx, KQV, 0, 2, 1, 3);
            KQV = ggml_cont(ctx, KQV);

            // Reshape to [n_embd, seq]
            KQV = ggml_reshape_2d(ctx, KQV, c2w_n_embd, seq_len);

            // Output projection
            cur = ggml_mul_mat(ctx, layer.wo, KQV);
        }

        // Debug: Track attention output before LayerScale (layer 0 and 7)
        if (debug_tensors && (il == 0 || il == 7)) {
            char name[64];
            snprintf(name, sizeof(name), "pretrans_layer%d_attn_out", il);
            ggml_set_name(cur, name);
            ggml_set_output(cur);
            debug_tensors->push_back(cur);
        }

        // LayerScale for attention
        if (layer.attn_scale) {
            cur = ggml_mul(ctx, cur, layer.attn_scale);
        }

        // Debug: Track attention output after LayerScale (layer 0 and 7)
        if (debug_tensors && (il == 0 || il == 7)) {
            char name[64];
            snprintf(name, sizeof(name), "pretrans_layer%d_attn_scaled", il);
            ggml_set_name(cur, name);
            ggml_set_output(cur);
            debug_tensors->push_back(cur);
        }

        // Residual
        cur = ggml_add(ctx, cur, inpSA);
        ggml_tensor * ffn_inp = cur;

        // Debug: Track after attention residual (layer 0 and 7)
        if (debug_tensors && (il == 0 || il == 7)) {
            char name[64];
            snprintf(name, sizeof(name), "pretrans_layer%d_after_attn_res", il);
            ggml_set_name(cur, name);
            ggml_set_output(cur);
            debug_tensors->push_back(cur);
        }

        // FFN norm (RMSNorm with Code2Wav epsilon)
        if (layer.ffn_norm) {
            cur = build_rms_norm(ctx, cur, layer.ffn_norm, C2W_RMS_NORM_EPS);
        }

        // Debug: Track after FFN norm (layer 0 and 7)
        if (debug_tensors && (il == 0 || il == 7)) {
            char name[64];
            snprintf(name, sizeof(name), "pretrans_layer%d_ffn_norm", il);
            ggml_set_name(cur, name);
            ggml_set_output(cur);
            debug_tensors->push_back(cur);
        }

        // FFN (SwiGLU)
        if (layer.ffn_gate && layer.ffn_up && layer.ffn_down) {
            ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate, cur);
            ggml_tensor * up = ggml_mul_mat(ctx, layer.ffn_up, cur);

            // Debug: Track FFN intermediates (layer 7 only)
            if (debug_tensors && il == 7) {
                ggml_set_name(gate, "pretrans_layer7_ffn_gate_raw");
                ggml_set_output(gate);
                debug_tensors->push_back(gate);
                ggml_set_name(up, "pretrans_layer7_ffn_up");
                ggml_set_output(up);
                debug_tensors->push_back(up);
            }

            gate = ggml_silu(ctx, gate);

            // Debug: Track after silu (layer 7 only)
            if (debug_tensors && il == 7) {
                ggml_set_name(gate, "pretrans_layer7_ffn_gate_silu");
                ggml_set_output(gate);
                debug_tensors->push_back(gate);
            }

            cur = ggml_mul(ctx, gate, up);

            // Debug: Track gate*up (layer 7 only)
            if (debug_tensors && il == 7) {
                ggml_set_name(cur, "pretrans_layer7_ffn_gate_up");
                ggml_set_output(cur);
                debug_tensors->push_back(cur);
            }

            cur = ggml_mul_mat(ctx, layer.ffn_down, cur);
        }

        // Debug: Track FFN output before LayerScale (layer 0 and 7)
        if (debug_tensors && (il == 0 || il == 7)) {
            char name[64];
            snprintf(name, sizeof(name), "pretrans_layer%d_ffn_out", il);
            ggml_set_name(cur, name);
            ggml_set_output(cur);
            debug_tensors->push_back(cur);
        }

        // LayerScale for FFN
        if (layer.ffn_scale) {
            cur = ggml_mul(ctx, cur, layer.ffn_scale);
        }

        // Debug: Track FFN output after LayerScale (layer 0 and 7)
        if (debug_tensors && (il == 0 || il == 7)) {
            char name[64];
            snprintf(name, sizeof(name), "pretrans_layer%d_ffn_scaled", il);
            ggml_set_name(cur, name);
            ggml_set_output(cur);
            debug_tensors->push_back(cur);
        }

        // Residual
        cur = ggml_add(ctx, cur, ffn_inp);

        // Debug: Track full layer output (all layers)
        if (debug_tensors) {
            char name[64];
            snprintf(name, sizeof(name), "pretrans_layer%d_output", il);
            ggml_set_name(cur, name);
            ggml_set_output(cur);
            debug_tensors->push_back(cur);
        }
    }

    // Debug: Track before output norm
    if (debug_tensors) {
        ggml_set_name(cur, "pretrans_before_output_norm");
        ggml_set_output(cur);
        debug_tensors->push_back(cur);
    }

    // Pre-transformer output norm (RMSNorm with Code2Wav epsilon)
    if (model->c2w_pre_output_norm) {
        cur = build_rms_norm(ctx, cur, model->c2w_pre_output_norm, C2W_RMS_NORM_EPS);
    }

    // Debug: Track after pre-transformer
    if (debug_tensors) {
        ggml_set_name(cur, "after_pretrans");
        ggml_set_output(cur);
        debug_tensors->push_back(cur);
    }

    // =========================================================================
    // ConvNeXt Upsample: 2 blocks (4× total)
    // =========================================================================
    if (verbose) {
        printf("  Building ConvNeXt upsample (%d blocks)...\n", c2w_up_n_block);
    }

    for (int ib = 0; ib < c2w_up_n_block && ib < (int)model->c2w_up_blocks.size(); ++ib) {
        const auto & block = model->c2w_up_blocks[ib];

        // Transpose convolution for 2x upsampling
        // GGML conv_transpose_1d expects input [seq, channels]
        // Our data is [channels, seq], so transpose before and after
        if (block.conv) {
            if (verbose) {
                printf("    ConvNeXt block %d: cur [%ld, %ld], conv kernel [%ld, %ld, %ld]\n",
                       ib,
                       (long)cur->ne[0], (long)cur->ne[1],
                       (long)block.conv->ne[0], (long)block.conv->ne[1], (long)block.conv->ne[2]);
            }
            // Transpose to [seq, channels] for conv
            cur = ggml_cont(ctx, ggml_transpose(ctx, cur));

            // Debug: Track transposed input before transconv
            if (debug_tensors) {
                char name[32];
                snprintf(name, sizeof(name), "cnxt%d_transconv_input", ib);
                ggml_set_name(cur, name);
                ggml_set_output(cur);
                debug_tensors->push_back(cur);
            }

            cur = ggml_conv_transpose_1d(ctx, block.conv, cur, 2, 0, 1);
            // Transpose back to [channels, seq]
            cur = ggml_cont(ctx, ggml_transpose(ctx, cur));

            // Apply conv bias (Fix #25)
            if (block.conv_bias) {
                ggml_tensor * bias_2d = ggml_reshape_2d(ctx, block.conv_bias, block.conv_bias->ne[0], 1);
                cur = ggml_add(ctx, cur, bias_2d);
            }

            seq_len *= 2;
            if (verbose) {
                printf("    After conv_transpose: [%ld, %ld]\n",
                       (long)cur->ne[0], (long)cur->ne[1]);
            }

        }

        // Debug: Track after transpose conv (before clamp)
        if (debug_tensors) {
            char name[32];
            snprintf(name, sizeof(name), "cnxt%d_transconv_raw", ib);
            ggml_set_name(cur, name);
            ggml_set_output(cur);
            debug_tensors->push_back(cur);
        }

        // Fix #7: Save input for ConvNeXt residual connection
        // HuggingFace: input = hidden_states (saved AFTER upsample, BEFORE ConvNeXt ops)
        ggml_tensor * convnext_residual = cur;

        // =====================================================================
        // Fix #36: Correct ConvNeXt dimension flow
        //
        // Key insight: GGML stores tensors as [features, seq] and mul_mat expects this.
        // But ggml_norm normalizes over ne[0] (first dim), which for [channels, seq]
        // means it normalizes each channel across seq positions - WRONG for LayerNorm.
        //
        // PyTorch LayerNorm normalizes each position across channels (last dim).
        // Fix: transpose around LayerNorm only, keep [channels, seq] for mul_mat.
        //
        // HuggingFace ConvNeXt flow:
        //   dwconv -> permute -> LayerNorm -> Linear -> GELU -> Linear -> gamma -> permute -> residual
        //
        // GGML: Keep [channels, seq], transpose around LayerNorm only
        // =====================================================================

        // Depthwise conv with CAUSAL padding (not symmetric "same" padding)
        // HuggingFace uses Qwen3OmniMoeCausalConvNet which pads left only:
        //   padding = kernel_size - stride = 7 - 1 = 6 (left padding, 0 right)
        //
        // GGUF converter permutes values: [C,1,K] -> [K,1,C] in memory
        // But GGUF stores shape as {1024,1,7} due to dimension reversal
        // Runtime permute changes shape to {7,1,1024} for ggml_conv_1d_dw
        if (block.dwconv) {
            // Transpose input: [channels, seq] -> [seq, channels]
            cur = ggml_cont(ctx, ggml_transpose(ctx, cur));

            // Causal padding: 6 on left (kernel_size - 1), 0 on right
            // ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3)
            // dim 0 is seq after transpose
            int kernel_size = 7;
            int left_pad = kernel_size - 1;  // 6
            cur = ggml_pad_ext(ctx, cur, left_pad, 0, 0, 0, 0, 0, 0, 0);

            // Permute kernel shape from {1024, 1, 7} to {7, 1, 1024}
            // Values are already in correct order from GGUF converter
            ggml_tensor * kernel = ggml_permute(ctx, block.dwconv, 2, 1, 0, 3);
            kernel = ggml_cont(ctx, kernel);

            // Use ggml_conv_1d_dw with p0=0 (no additional padding)
            cur = ggml_conv_1d_dw(ctx, kernel, cur, 1, 0, 1);

            // Transpose back: [seq, channels] -> [channels, seq]
            cur = ggml_cont(ctx, ggml_transpose(ctx, cur));

            // Apply dwconv bias
            if (block.dwconv_bias) {
                ggml_tensor * bias_2d = ggml_reshape_2d(ctx, block.dwconv_bias, block.dwconv_bias->ne[0], 1);
                cur = ggml_add(ctx, cur, bias_2d);
            }
        }

        // Debug: Track after dwconv
        if (debug_tensors) {
            char name[32];
            snprintf(name, sizeof(name), "cnxt%d_dwconv", ib);
            ggml_set_name(cur, name);
            ggml_set_output(cur);
            debug_tensors->push_back(cur);
        }

        // LayerNorm: cur has shape ne[0]=channels=1024, ne[1]=seq
        // ggml_norm normalizes over ne[0] (channels) - this is correct for LayerNorm
        // (each seq position normalized independently across all channels)
        if (block.norm) {
            cur = ggml_norm(ctx, cur, 1e-6f);

            // Reshape norm weight/bias to 2D [channels, 1] for broadcasting
            ggml_tensor * norm_2d = ggml_reshape_2d(ctx, block.norm, block.norm->ne[0], 1);
            cur = ggml_mul(ctx, cur, norm_2d);

            if (block.norm_bias) {
                ggml_tensor * bias_2d = ggml_reshape_2d(ctx, block.norm_bias, block.norm_bias->ne[0], 1);
                cur = ggml_add(ctx, cur, bias_2d);
            }
        }

        // Debug: Track after norm
        if (debug_tensors) {
            char name[32];
            snprintf(name, sizeof(name), "cnxt%d_norm", ib);
            ggml_set_name(cur, name);
            ggml_set_output(cur);
            debug_tensors->push_back(cur);
        }

        // Pointwise conv 1 + GELU (mul_mat on [channels, seq])
        if (block.pwconv1) {
            cur = ggml_mul_mat(ctx, block.pwconv1, cur);

            // Apply pwconv1 bias (Fix #25) - before GELU
            if (block.pwconv1_bias) {
                ggml_tensor * bias_2d = ggml_reshape_2d(ctx, block.pwconv1_bias, block.pwconv1_bias->ne[0], 1);
                cur = ggml_add(ctx, cur, bias_2d);
            }

            // Debug: Track after pwconv1, before GELU
            if (debug_tensors) {
                char name[32];
                snprintf(name, sizeof(name), "cnxt%d_pw1", ib);
                ggml_set_name(cur, name);
                ggml_set_output(cur);
                debug_tensors->push_back(cur);
            }

            cur = ggml_gelu(ctx, cur);
        }

        // Debug: Track after pwconv1+GELU
        if (debug_tensors) {
            char name[32];
            snprintf(name, sizeof(name), "cnxt%d_gelu", ib);
            ggml_set_name(cur, name);
            ggml_set_output(cur);
            debug_tensors->push_back(cur);
        }

        // Pointwise conv 2
        if (block.pwconv2) {
            cur = ggml_mul_mat(ctx, block.pwconv2, cur);

            // Apply pwconv2 bias (Fix #25)
            if (block.pwconv2_bias) {
                ggml_tensor * bias_2d = ggml_reshape_2d(ctx, block.pwconv2_bias, block.pwconv2_bias->ne[0], 1);
                cur = ggml_add(ctx, cur, bias_2d);
            }
        }

        // Debug: Track after pwconv2, before gamma
        if (debug_tensors) {
            char name[32];
            snprintf(name, sizeof(name), "cnxt%d_pw2", ib);
            ggml_set_name(cur, name);
            ggml_set_output(cur);
            debug_tensors->push_back(cur);
        }

        // LayerScale (gamma)
        if (block.gamma) {
            cur = ggml_mul(ctx, cur, block.gamma);
        }

        // Debug: Track after LayerScale
        if (debug_tensors) {
            char name[32];
            snprintf(name, sizeof(name), "cnxt%d_scale", ib);
            ggml_set_name(cur, name);
            ggml_set_output(cur);
            debug_tensors->push_back(cur);
        }

        // Fix #7: Add residual connection
        cur = ggml_add(ctx, cur, convnext_residual);
    }

    // Debug: Track after ConvNeXt
    if (debug_tensors) {
        ggml_set_name(cur, "after_convnext");
        ggml_set_output(cur);
        debug_tensors->push_back(cur);
    }

    // =========================================================================
    // HiFi-GAN Decoder: 4 stages (480× total)
    // =========================================================================
    if (verbose) {
        printf("  Building HiFi-GAN decoder (%d stages)...\n", c2w_dec_n_stage);
    }

    // Initial conv: 1024 → 1536
    // GGML conv expects input [seq, channels], so transpose from [channels, seq]
    // Fix #17: Use causal (left-only) padding instead of symmetric
    if (model->c2w_dec_conv_in) {
        cur = ggml_cont(ctx, ggml_transpose(ctx, cur));  // [channels, seq] -> [seq, channels]
        cur = build_causal_conv1d(ctx, model->c2w_dec_conv_in, cur, 1);  // kernel=7, dilation=1
        cur = ggml_cont(ctx, ggml_transpose(ctx, cur));  // [seq, channels] -> [channels, seq]
    }

    // Apply conv_in bias (important for proper signal operating point)
    if (model->c2w_dec_conv_in_b) {
        // Reshape bias from [channels] to [channels, 1] for broadcasting over seq
        ggml_tensor * bias_2d = ggml_reshape_2d(ctx, model->c2w_dec_conv_in_b,
                                                model->c2w_dec_conv_in_b->ne[0], 1);
        cur = ggml_add(ctx, cur, bias_2d);
    }

    // Debug: Track after HiFi-GAN conv_in
    if (debug_tensors) {
        ggml_set_name(cur, "after_hifi_convin");
        ggml_set_output(cur);
        debug_tensors->push_back(cur);
    }

    // 4 upsample stages
    for (int stage = 0; stage < c2w_dec_n_stage && stage < (int)model->c2w_dec_blocks.size(); ++stage) {
        const auto & dec_block = model->c2w_dec_blocks[stage];
        int rate = upsample_rates[stage];

        if (verbose) {
            printf("    HiFi-GAN stage %d (rate %d): cur [%ld, %ld]\n",
                   stage, rate, (long)cur->ne[0], (long)cur->ne[1]);
        }

        // Debug: Track before outer Snake
        if (debug_tensors) {
            char name[64];
            snprintf(name, sizeof(name), "hifi_stage%d_before_snake", stage);
            ggml_set_name(cur, name);
            ggml_set_output(cur);
            debug_tensors->push_back(cur);
        }

        // Outer Snake activation (element-wise, needs [channels, seq])
        cur = build_snake(ctx, cur, dec_block.snake_alpha, dec_block.snake_beta);

        // Debug: Track after outer Snake
        if (debug_tensors) {
            char name[64];
            snprintf(name, sizeof(name), "hifi_stage%d_after_snake", stage);
            ggml_set_name(cur, name);
            ggml_set_output(cur);
            debug_tensors->push_back(cur);
        }

        // Transpose conv for upsampling
        // GGML conv_transpose_1d expects input [seq, channels]
        if (dec_block.upsample) {
            cur = ggml_cont(ctx, ggml_transpose(ctx, cur));  // [channels, seq] -> [seq, channels]
            cur = ggml_conv_transpose_1d(ctx, dec_block.upsample, cur, rate, 0, 1);
            cur = ggml_cont(ctx, ggml_transpose(ctx, cur));  // [seq', channels'] -> [channels', seq']

            // Fix #16: Trim transpose conv output (HuggingFace CausalTransConvNet)
            // kernel = 2 * rate, so trim = kernel - rate = rate
            int trim = rate;  // left_pad = right_pad = rate
            int64_t out_seq = cur->ne[1];  // [channels, seq]
            if (out_seq > 2 * trim) {
                // View to trim: skip first 'trim' and last 'trim' samples
                cur = ggml_view_2d(ctx, cur, cur->ne[0], out_seq - 2*trim,
                                   cur->nb[1], trim * cur->nb[1]);
                cur = ggml_cont(ctx, cur);
            }
            seq_len = seq_len * rate - 2 * trim;  // Adjust for trimming

            // Add transpose conv bias (cur: [channels, seq], bias: [channels])
            // Reshape bias to [channels, 1] for broadcasting over seq
            if (dec_block.upsample_bias) {
                ggml_tensor * bias_2d = ggml_reshape_2d(ctx, dec_block.upsample_bias,
                                                        dec_block.upsample_bias->ne[0], 1);
                cur = ggml_add(ctx, cur, bias_2d);
            }
        }

        // Debug: Track after upsample
        if (debug_tensors) {
            char name[64];
            snprintf(name, sizeof(name), "hifi_stage%d_after_upsample", stage);
            ggml_set_name(cur, name);
            ggml_set_output(cur);
            debug_tensors->push_back(cur);
        }

        // 3 residual blocks per stage with dilations (1, 3, 9)
        // HuggingFace: for dilation in (1, 3, 9): block.append(DecoderResidualUnit(out_dim, dilation))
        const int dilations[] = {1, 3, 9};
        for (int rb = 0; rb < c2w_dec_n_resblk; ++rb) {
            int flat_idx = stage * c2w_dec_n_resblk + rb;
            if (flat_idx >= (int)model->c2w_dec_res_blks.size()) break;

            const auto & res_blk = model->c2w_dec_res_blks[flat_idx];
            ggml_tensor * residual = cur;
            int dilation = dilations[rb];  // dilation pattern: 1, 3, 9

            if (verbose && rb == 0) {
                printf("      ResBlk %d-%d: dilation=%d, conv1=%s\n",
                       stage, rb, dilation, res_blk.conv1 ? "yes" : "no");
            }

            // Snake1 (element-wise on [channels, seq])
            cur = build_snake(ctx, cur, res_blk.act1_alpha, res_blk.act1_beta);

            // Conv1 with dilation - GGML expects [seq, channels]
            // HuggingFace: self.conv1 = CausalConvNet(dim, dim, kernel_size=7, dilation=dilation)
            // Fix #17: Use causal (left-only) padding instead of symmetric
            if (res_blk.conv1) {
                cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
                cur = build_causal_conv1d(ctx, res_blk.conv1, cur, dilation);
                cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
                // Fix #23: Add conv1 bias
                if (res_blk.conv1_bias) {
                    ggml_tensor * bias_2d = ggml_reshape_2d(ctx, res_blk.conv1_bias,
                                                            res_blk.conv1_bias->ne[0], 1);
                    cur = ggml_add(ctx, cur, bias_2d);
                }
            } else if (res_blk.conv) {
                cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
                cur = build_causal_conv1d(ctx, res_blk.conv, cur, dilation);
                cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
            }

            // Snake2 (element-wise)
            cur = build_snake(ctx, cur, res_blk.act2_alpha, res_blk.act2_beta);

            // Conv2 with kernel=1 (no dilation effect) - GGML expects [seq, channels]
            // HuggingFace: self.conv2 = CausalConvNet(dim, dim, kernel_size=1)
            if (res_blk.conv2) {
                cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
                cur = ggml_conv_1d_ph(ctx, res_blk.conv2, cur, 1, 1);
                cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
                // Fix #23: Add conv2 bias
                if (res_blk.conv2_bias) {
                    ggml_tensor * bias_2d = ggml_reshape_2d(ctx, res_blk.conv2_bias,
                                                            res_blk.conv2_bias->ne[0], 1);
                    cur = ggml_add(ctx, cur, bias_2d);
                }
            }

            // Residual connection
            cur = ggml_add(ctx, cur, residual);

            // Debug: Track ResBlock output
            if (debug_tensors && rb == 2) {  // Track last ResBlock of each stage
                char name[64];
                snprintf(name, sizeof(name), "hifi_stage%d_after_resblk", stage);
                ggml_set_name(cur, name);
                ggml_set_output(cur);
                debug_tensors->push_back(cur);
            }
        }
    }

    // Fix #10: Final Snake activation before final conv
    // HuggingFace: decoder.5 = SnakeBeta(output_dim)
    // Without this, the final conv gets non-activated input which produces wrong output
    if (model->c2w_dec_final_snake_a && model->c2w_dec_final_snake_b) {
        cur = build_snake(ctx, cur, model->c2w_dec_final_snake_a, model->c2w_dec_final_snake_b);
    }

    // Debug: Track after final snake
    if (debug_tensors) {
        ggml_set_name(cur, "after_final_snake");
        ggml_set_output(cur);
        debug_tensors->push_back(cur);
    }

    // Final conv: 96 → 1
    // GGML conv expects [seq, channels]
    // Fix #17: Use causal (left-only) padding instead of symmetric
    if (model->c2w_dec_conv_out) {
        cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
        cur = build_causal_conv1d(ctx, model->c2w_dec_conv_out, cur, 1);  // kernel=7, dilation=1
        cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
        // Fix #23: Add conv_out bias
        if (model->c2w_dec_conv_out_b) {
            ggml_tensor * bias_2d = ggml_reshape_2d(ctx, model->c2w_dec_conv_out_b,
                                                    model->c2w_dec_conv_out_b->ne[0], 1);
            cur = ggml_add(ctx, cur, bias_2d);
        }
    }

    // Debug: Track before clamp
    if (debug_tensors) {
        ggml_tensor * before_clamp = ggml_cont(ctx, cur);
        ggml_set_name(before_clamp, "before_clamp");
        ggml_set_output(before_clamp);
        ggml_build_forward_expand(gf, before_clamp);
        debug_tensors->push_back(before_clamp);
    }

    // Fix #34: Use clamp instead of tanh (matches HuggingFace)
    // HuggingFace: return wav.clamp(min=-1, max=1)
    // Values are already in reasonable range [-3, +4], just clamp directly
    cur = ggml_clamp(ctx, cur, -1.0f, 1.0f);

    // Mark as output
    ggml_build_forward_expand(gf, cur);

    return cur;
}

// Chunked decode constants
// CUDA IM2COL kernel has grid y-dimension limit of 65535
// Max safe frames = 65535 / (4 * 480) = 34
// Must account for context: chunk_size + context <= 34
// Using 25 new frames + 5 context = 30 total per chunk
static const int CODE2WAV_CHUNK_SIZE = 25;        // 25 new frames per chunk (CUDA IM2COL limit)
static const int CODE2WAV_LEFT_CONTEXT = 5;       // 5 context frames
static const int CODE2WAV_TOTAL_UPSAMPLE = 1920;  // 4 (ConvNeXt) × 480 (HiFi-GAN)

// Run Code2Wav vocoder for a single chunk using ggml graph
static std::vector<float> run_code2wav_chunk(
    const llama_model * model,
    const std::vector<std::vector<int>> & all_codebook_tokens,
    bool verbose,
    const std::string & dump_tensors_path = "",
    bool cpu_only = false) {

    const int c2w_n_embd = 1024;
    const int n_frames = all_codebook_tokens.size();
    const int n_codebooks = 16;

    // Total upsampling: 4 (ConvNeXt) × 480 (HiFi-GAN) = 1920
    const int total_upsample = 4 * 8 * 5 * 4 * 3;
    const int n_samples = n_frames * total_upsample;

    if (verbose) {
        printf("Code2Wav (ggml): %d frames → %d samples (%.2f sec @ 24kHz)\n",
               n_frames, n_samples, n_samples / 24000.0f);

        // Debug: Check which HiFi-GAN tensors are loaded
        printf("  Checking HiFi-GAN decoder tensors:\n");
        printf("    c2w_dec_conv_in: %s\n", model->c2w_dec_conv_in ? "loaded" : "NULL");
        printf("    c2w_dec_conv_out: %s\n", model->c2w_dec_conv_out ? "loaded" : "NULL");
        // Debug: Print final conv weight stats
        if (model->c2w_dec_conv_out) {
            std::vector<float> conv_out_w;
            if (copy_tensor_to_cpu(model->c2w_dec_conv_out, conv_out_w)) {
                float w_min = conv_out_w[0], w_max = conv_out_w[0], w_sum = 0;
                for (float w : conv_out_w) {
                    if (w < w_min) w_min = w;
                    if (w > w_max) w_max = w;
                    w_sum += w;
                }
                printf("      conv_out weights: shape=[%ld,%ld,%ld], range=[%.4f, %.4f], mean=%.6f\n",
                       (long)model->c2w_dec_conv_out->ne[0],
                       (long)model->c2w_dec_conv_out->ne[1],
                       (long)model->c2w_dec_conv_out->ne[2],
                       w_min, w_max, w_sum / conv_out_w.size());
            }
        }
        printf("    c2w_dec_blocks: %zu stages\n", model->c2w_dec_blocks.size());
        for (size_t i = 0; i < model->c2w_dec_blocks.size(); ++i) {
            const auto & block = model->c2w_dec_blocks[i];
            printf("      Stage %zu: snake_alpha=%s, snake_beta=%s, upsample=%s\n",
                   i, block.snake_alpha ? "loaded" : "NULL",
                   block.snake_beta ? "loaded" : "NULL",
                   block.upsample ? "loaded" : "NULL");
            if (block.upsample) {
                printf("        upsample shape: [%ld, %ld, %ld]\n",
                       (long)block.upsample->ne[0], (long)block.upsample->ne[1], (long)block.upsample->ne[2]);
            }
        }
        printf("    c2w_dec_res_blks: %zu blocks\n", model->c2w_dec_res_blks.size());
        for (size_t i = 0; i < model->c2w_dec_res_blks.size() && i < 3; ++i) {
            const auto & blk = model->c2w_dec_res_blks[i];
            printf("      ResBlk %zu: conv1=%s, conv2=%s, act1_alpha=%s\n",
                   i, blk.conv1 ? "loaded" : "NULL",
                   blk.conv2 ? "loaded" : "NULL",
                   blk.act1_alpha ? "loaded" : "NULL");
        }

        // Debug: Check ConvNeXt gamma (LayerScale) values
        printf("    ConvNeXt (c2w_up) blocks: %zu\n", model->c2w_up_blocks.size());
        for (size_t i = 0; i < model->c2w_up_blocks.size(); ++i) {
            const auto & block = model->c2w_up_blocks[i];
            printf("      Block %zu: gamma=%s, pwconv1=%s, pwconv2=%s\n",
                   i, block.gamma ? "loaded" : "NULL",
                   block.pwconv1 ? "loaded" : "NULL",
                   block.pwconv2 ? "loaded" : "NULL");
            if (block.gamma) {
                std::vector<float> gamma_data;
                if (copy_tensor_to_cpu(block.gamma, gamma_data)) {
                    float g_min = gamma_data[0], g_max = gamma_data[0], g_sum = 0;
                    for (size_t j = 0; j < gamma_data.size(); j++) {
                        if (gamma_data[j] < g_min) g_min = gamma_data[j];
                        if (gamma_data[j] > g_max) g_max = gamma_data[j];
                        g_sum += gamma_data[j];
                    }
                    printf("        gamma stats: min=%.4f, max=%.4f, mean=%.4f\n",
                           g_min, g_max, g_sum / gamma_data.size());
                }
            }
            // Check pwconv2 weight stats (this multiplies GELU output)
            if (block.pwconv2) {
                std::vector<float> pw2_data;
                if (copy_tensor_to_cpu(block.pwconv2, pw2_data)) {
                    float pw2_min = pw2_data[0], pw2_max = pw2_data[0];
                    for (size_t j = 0; j < pw2_data.size(); j++) {
                        if (pw2_data[j] < pw2_min) pw2_min = pw2_data[j];
                        if (pw2_data[j] > pw2_max) pw2_max = pw2_data[j];
                    }
                    printf("        pwconv2 stats: min=%.4f, max=%.4f, shape=[%ld, %ld]\n",
                           pw2_min, pw2_max, (long)block.pwconv2->ne[0], (long)block.pwconv2->ne[1]);
                }
            }
        }
    }

    // Initialize context
    code2wav_context c2w_ctx;
    if (!init_code2wav_context(&c2w_ctx, model, cpu_only)) {
        fprintf(stderr, "Error: Failed to initialize Code2Wav context\n");
        return std::vector<float>(n_samples, 0.0f);
    }

    // Step 1: Embed codebook tokens and sum (on CPU)
    std::vector<float> embd_data;
    if (!copy_tensor_to_cpu(model->c2w_code_embd, embd_data)) {
        fprintf(stderr, "Error: Failed to copy Code2Wav embedding\n");
        return std::vector<float>(n_samples, 0.0f);
    }

    // Sum embeddings for each frame: c2w_code_embd is [n_embd, vocab=32768]
    std::vector<float> input_data(n_frames * c2w_n_embd, 0.0f);
    for (int f = 0; f < n_frames; ++f) {
        for (int cb = 0; cb < n_codebooks; ++cb) {
            int token = all_codebook_tokens[f][cb];
            int vocab_idx = cb * 2048 + (token % 2048);
            for (int i = 0; i < c2w_n_embd; ++i) {
                input_data[f * c2w_n_embd + i] += embd_data[vocab_idx * c2w_n_embd + i];
            }
        }
    }

    // Fix #8: Average across codebooks (HuggingFace: .mean(1))
    // Without this, input amplitude is 16x too large!
    for (int f = 0; f < n_frames; ++f) {
        for (int i = 0; i < c2w_n_embd; ++i) {
            input_data[f * c2w_n_embd + i] /= (float)n_codebooks;
        }
    }

    if (verbose) {
        printf("  Embedded %d frames × %d codebooks (averaged)\n", n_frames, n_codebooks);
        // Debug: Print input stats
        float min_val = input_data[0], max_val = input_data[0], sum = 0;
        for (size_t i = 0; i < input_data.size(); i++) {
            if (input_data[i] < min_val) min_val = input_data[i];
            if (input_data[i] > max_val) max_val = input_data[i];
            sum += input_data[i];
        }
        printf("  Input stats: min=%.4f, max=%.4f, mean=%.4f\n",
               min_val, max_val, sum / input_data.size());
    }

    // Step 2: Create ggml context for graph building
    struct ggml_init_params params = {
        /*.mem_size   =*/ c2w_ctx.buf_compute_meta.size(),
        /*.mem_buffer =*/ c2w_ctx.buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Error: Failed to create ggml context\n");
        return std::vector<float>(n_samples, 0.0f);
    }

    // Create graph
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, c2w_ctx.max_nodes, false);

    // Create input tensor
    ggml_tensor * input_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c2w_n_embd, n_frames);
    ggml_set_name(input_tensor, "c2w_input");
    ggml_set_input(input_tensor);

    // Build compute graph
    std::vector<ggml_tensor *> debug_tensors;
    bool collect_debug = verbose || !dump_tensors_path.empty();
    if (verbose) {
        printf("  Building compute graph...\n");
    }
    ggml_tensor * attn_mask = nullptr;
    ggml_tensor * output = build_code2wav_graph(ctx, gf, model, input_tensor, n_frames, verbose,
                                                 collect_debug ? &debug_tensors : nullptr, &attn_mask);

    // Step 3: Allocate graph
    if (verbose) {
        printf("  Allocating graph...\n");
    }
    ggml_backend_sched_reset(c2w_ctx.sched);

    if (!ggml_backend_sched_alloc_graph(c2w_ctx.sched, gf)) {
        fprintf(stderr, "Error: Failed to allocate graph\n");
        ggml_free(ctx);
        return std::vector<float>(n_samples, 0.0f);
    }

    // Step 4: Set input data
    ggml_backend_tensor_set(input_tensor, input_data.data(), 0, input_data.size() * sizeof(float));

    // Step 4b: Fill sliding window attention mask
    // For each query position q, attend only to key positions k where:
    // 1. k <= q (causal: no future tokens)
    // 2. q - k < sliding_window (sliding window: no distant past tokens)
    if (attn_mask) {
        std::vector<float> mask_data(n_frames * n_frames);
        for (int q = 0; q < n_frames; q++) {
            for (int k = 0; k < n_frames; k++) {
                int idx = q * n_frames + k;
                bool masked = (k > q) || (q - k >= C2W_SLIDING_WINDOW);
                mask_data[idx] = masked ? -INFINITY : 0.0f;
            }
        }
        ggml_backend_tensor_set(attn_mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
    }

    // Step 5: Compute graph
    if (verbose) {
        printf("  Computing graph (%d nodes)...\n", ggml_graph_n_nodes(gf));
    }

    if (ggml_backend_sched_graph_compute(c2w_ctx.sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Error: Graph computation failed\n");
        ggml_free(ctx);
        return std::vector<float>(n_samples, 0.0f);
    }

    // Debug: Check intermediate tensors for NaN
    if (verbose && !debug_tensors.empty()) {
        printf("  Checking intermediate tensors:\n");
        for (auto * t : debug_tensors) {
            if (t && t->name) {
                debug_check_tensor(t->name, nullptr, t);
            }
        }
    }

    // Dump tensors to files if requested
    if (!dump_tensors_path.empty() && !debug_tensors.empty()) {
        printf("  Dumping %zu tensors to %s/\n", debug_tensors.size(), dump_tensors_path.c_str());
        for (auto * t : debug_tensors) {
            if (t && t->name) {
                int64_t n_elem = ggml_nelements(t);
                std::vector<float> data(n_elem);
                ggml_backend_tensor_get(t, data.data(), 0, n_elem * sizeof(float));

                // Write raw binary (can be loaded with numpy.fromfile)
                std::string filepath = dump_tensors_path + "/" + std::string(t->name) + ".bin";
                std::ofstream ofs(filepath, std::ios::binary);
                if (ofs.is_open()) {
                    ofs.write(reinterpret_cast<const char*>(data.data()), n_elem * sizeof(float));
                    ofs.close();
                    printf("    %s: [%lld, %lld] -> %s\n",
                           t->name, (long long)t->ne[0], (long long)t->ne[1], filepath.c_str());
                }
            }
        }
    }

    // Step 6: Extract output
    int64_t output_size = ggml_nelements(output);
    std::vector<float> audio(output_size);
    ggml_backend_tensor_get(output, audio.data(), 0, output_size * sizeof(float));

    if (verbose) {
        printf("  Generated %ld samples\n", (long)output_size);
        // Debug: Print raw output stats
        float out_min = audio[0], out_max = audio[0], out_sum = 0;
        int near_zero = 0, nan_count = 0, inf_count = 0;
        for (size_t i = 0; i < audio.size(); i++) {
            if (std::isnan(audio[i])) { nan_count++; continue; }
            if (std::isinf(audio[i])) { inf_count++; continue; }
            if (audio[i] < out_min) out_min = audio[i];
            if (audio[i] > out_max) out_max = audio[i];
            out_sum += audio[i];
            if (std::abs(audio[i]) < 0.01f) near_zero++;
        }
        size_t valid = audio.size() - nan_count - inf_count;
        printf("  Raw output stats: min=%.6f, max=%.6f, mean=%.6f\n",
               out_min, out_max, valid > 0 ? out_sum / valid : 0.0f);
        printf("  Near-zero samples (<0.01): %d (%.1f%%)\n",
               near_zero, 100.0f * near_zero / audio.size());
        printf("  NaN: %d, Inf: %d (%.1f%% invalid)\n",
               nan_count, inf_count, 100.0f * (nan_count + inf_count) / audio.size());
    }

    // Replace NaN/Inf with 0 before normalization
    for (size_t i = 0; i < audio.size(); i++) {
        if (std::isnan(audio[i]) || std::isinf(audio[i])) {
            audio[i] = 0.0f;
        }
    }

    // Normalize audio to use full dynamic range
    // The tanh output is often too quiet, so scale to 0.9 max amplitude
    float max_abs = 0.0f;
    for (size_t i = 0; i < audio.size(); i++) {
        float abs_val = std::abs(audio[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    if (max_abs > 1e-6f) {
        float scale = 0.9f / max_abs;
        if (verbose) {
            printf("  Audio normalization: max_abs=%.6f, scale=%.2f\n", max_abs, scale);
        }
        for (size_t i = 0; i < audio.size(); i++) {
            audio[i] *= scale;
        }
    }

    // Cleanup
    ggml_free(ctx);

    return audio;
}

// Run Code2Wav vocoder with chunked decode for large inputs
// This prevents CUDA IM2COL errors on large sequences
static std::vector<float> run_code2wav_ggml(
    const llama_model * model,
    const std::vector<std::vector<int>> & all_codebook_tokens,
    bool verbose,
    const std::string & dump_tensors_path = "",
    bool cpu_only = false) {

    const int n_frames = (int)all_codebook_tokens.size();

    // For small inputs, process directly
    if (n_frames <= CODE2WAV_CHUNK_SIZE) {
        return run_code2wav_chunk(model, all_codebook_tokens, verbose, dump_tensors_path, cpu_only);
    }

    // Chunked decode for large inputs
    if (verbose) {
        printf("Code2Wav: Using chunked decode (%d frames, chunk_size=%d)\n",
               n_frames, CODE2WAV_CHUNK_SIZE);
    }

    std::vector<float> full_wav;
    int start_idx = 0;
    int chunk_num = 0;

    while (start_idx < n_frames) {
        int end_idx = std::min(start_idx + CODE2WAV_CHUNK_SIZE, n_frames);
        int context = (start_idx > CODE2WAV_LEFT_CONTEXT) ? CODE2WAV_LEFT_CONTEXT : start_idx;

        // Extract chunk with context
        std::vector<std::vector<int>> chunk(
            all_codebook_tokens.begin() + start_idx - context,
            all_codebook_tokens.begin() + end_idx
        );

        if (verbose) {
            printf("  Chunk %d: frames [%d-%d) with %d context frames\n",
                   chunk_num, start_idx - context, end_idx, context);
        }

        // Process chunk
        std::vector<float> wav_chunk = run_code2wav_chunk(model, chunk, false, "", cpu_only);

        // Discard context portion and append
        int discard_samples = context * CODE2WAV_TOTAL_UPSAMPLE;
        if (discard_samples < (int)wav_chunk.size()) {
            full_wav.insert(full_wav.end(),
                wav_chunk.begin() + discard_samples,
                wav_chunk.end());
        }

        start_idx = end_idx;
        chunk_num++;
    }

    if (verbose) {
        printf("Code2Wav: Generated %zu total samples from %d chunks\n",
               full_wav.size(), chunk_num);
    }

    return full_wav;
}

// =============================================================================
// Legacy Code2Wav (placeholder - kept for reference)
// =============================================================================

// Run Code2Wav vocoder (simplified version)
static std::vector<float> run_code2wav(
    const llama_model * model,
    const std::vector<std::vector<int>> & all_codebook_tokens,
    bool verbose) {

    const int c2w_n_embd = 1024;
    const int c2w_dec_dim = 1536;
    const int n_frames = all_codebook_tokens.size();
    const int n_codebooks = 16;

    // Upsampling rates for HiFi-GAN decoder
    const int upsample_rates[] = {8, 5, 4, 3};
    const int n_stages = 4;

    // Calculate output length
    // Pre-transformer: no upsample (1×)
    // ConvNeXt: 2×2 = 4×
    // HiFi-GAN: 8×5×4×3 = 480×
    // Total: 4 × 480 = 1920×
    int seq_len = n_frames;
    int total_upsample = 4 * 8 * 5 * 4 * 3;  // 1920
    int n_samples = n_frames * total_upsample;

    if (verbose) {
        printf("Code2Wav: %d frames → %d samples (%.2f sec @ 24kHz)\n",
               n_frames, n_samples, n_samples / 24000.0f);
    }

    // Step 1: Embed codebook tokens and sum
    std::vector<float> embd_sum;
    if (!copy_tensor_to_cpu(model->c2w_code_embd, embd_sum)) {
        fprintf(stderr, "Error: Failed to copy Code2Wav embedding\n");
        return std::vector<float>(n_samples, 0.0f);
    }

    // c2w_code_embd shape is [c2w_n_embd, vocab] where vocab=32768 (16 codebooks × 2048)
    // Each codebook has 2048 entries
    std::vector<float> cur(n_frames * c2w_n_embd, 0.0f);
    for (int f = 0; f < n_frames; ++f) {
        for (int cb = 0; cb < n_codebooks; ++cb) {
            int token = all_codebook_tokens[f][cb];
            // Token index in combined vocabulary
            int vocab_idx = cb * 2048 + (token % 2048);
            for (int i = 0; i < c2w_n_embd; ++i) {
                cur[f * c2w_n_embd + i] += embd_sum[vocab_idx * c2w_n_embd + i];
            }
        }
    }

    if (verbose) {
        printf("Code2Wav: Embedded %d frames × %d codebooks\n", n_frames, n_codebooks);
    }

    // Step 2: Skip pre-transformer for now (8 layers)
    // Just apply output norm
    std::vector<float> pre_norm;
    if (copy_tensor_to_cpu(model->c2w_pre_output_norm, pre_norm)) {
        std::vector<float> normed(n_frames * c2w_n_embd);
        for (int f = 0; f < n_frames; ++f) {
            rms_norm(cur.data() + f * c2w_n_embd, pre_norm.data(),
                     normed.data() + f * c2w_n_embd, c2w_n_embd);
        }
        cur = std::move(normed);
    }

    // Step 3: ConvNeXt upsample blocks (2 blocks, 2× each = 4× total)
    // Simplified: just duplicate frames
    seq_len = n_frames;
    for (int ub = 0; ub < 2; ++ub) {
        std::vector<float> upsampled(seq_len * 2 * c2w_n_embd);
        for (int s = 0; s < seq_len; ++s) {
            // Duplicate each frame
            for (int i = 0; i < c2w_n_embd; ++i) {
                upsampled[(s * 2) * c2w_n_embd + i] = cur[s * c2w_n_embd + i];
                upsampled[(s * 2 + 1) * c2w_n_embd + i] = cur[s * c2w_n_embd + i];
            }
        }
        cur = std::move(upsampled);
        seq_len *= 2;
    }

    if (verbose) {
        printf("Code2Wav: After ConvNeXt upsample: %d frames\n", seq_len);
    }

    // Step 4: Project to decoder dimension (1024 → 1536)
    // For simplicity, pad with zeros
    std::vector<float> dec_cur(seq_len * c2w_dec_dim, 0.0f);
    for (int s = 0; s < seq_len; ++s) {
        for (int i = 0; i < std::min(c2w_n_embd, c2w_dec_dim); ++i) {
            dec_cur[s * c2w_dec_dim + i] = cur[s * c2w_n_embd + i];
        }
    }
    cur = std::move(dec_cur);

    // Step 5: HiFi-GAN decoder stages
    // Each stage: Snake → TransposeConv(upsample) → 3 ResBlocks
    for (int stage = 0; stage < n_stages; ++stage) {
        int rate = upsample_rates[stage];

        // Apply Snake activation (simplified: use constant alpha=1)
        std::vector<float> snake_out(seq_len * c2w_dec_dim);
        std::vector<float> ones(c2w_dec_dim, 1.0f);
        snake_activation(cur.data(), ones.data(), nullptr,
                        snake_out.data(), seq_len, c2w_dec_dim);

        // Upsample via simple interpolation (not real transposed conv)
        int new_seq_len = seq_len * rate;
        std::vector<float> upsampled(new_seq_len * c2w_dec_dim);
        for (int s = 0; s < seq_len; ++s) {
            for (int r = 0; r < rate; ++r) {
                float t = (float)r / rate;
                int next_s = std::min(s + 1, seq_len - 1);
                for (int i = 0; i < c2w_dec_dim; ++i) {
                    // Linear interpolation
                    upsampled[(s * rate + r) * c2w_dec_dim + i] =
                        (1.0f - t) * snake_out[s * c2w_dec_dim + i] +
                        t * snake_out[next_s * c2w_dec_dim + i];
                }
            }
        }
        cur = std::move(upsampled);
        seq_len = new_seq_len;

        // Skip ResBlocks for simplified implementation

        if (verbose) {
            printf("Code2Wav: Stage %d (×%d): %d frames\n", stage, rate, seq_len);
        }
    }

    // Step 6: Final projection to mono audio
    // Average all channels and apply tanh
    std::vector<float> audio(seq_len);
    for (int s = 0; s < seq_len; ++s) {
        float sum = 0.0f;
        for (int i = 0; i < c2w_dec_dim; ++i) {
            sum += cur[s * c2w_dec_dim + i];
        }
        audio[s] = tanhf(sum / c2w_dec_dim);
    }

    // Normalize audio
    float max_val = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        max_val = std::max(max_val, std::abs(audio[i]));
    }
    if (max_val > 0.0f) {
        for (int i = 0; i < seq_len; ++i) {
            audio[i] = audio[i] / max_val * 0.9f;  // Normalize to 0.9 max
        }
    }

    return audio;
}

// Run Code Predictor for a single token to predict next codebook
// Returns the predicted token ID
static int run_code_predictor_step(
    const llama_model * model,
    int input_token,
    int codebook_idx,  // 0-14 for codebooks 2-16
    bool verbose) {

    const int cp_n_embd = 1024;
    const int cp_n_head = 16;
    const int cp_n_head_kv = 8;
    const int cp_head_dim = 128;
    const int cp_n_ff = 3072;
    const int cp_n_layer = 5;
    const int cp_vocab = 2048;

    // Get embedding tensor for this codebook
    if (codebook_idx >= (int)model->talker_cp_codec_embd.size()) {
        fprintf(stderr, "Error: codebook_idx %d out of range\n", codebook_idx);
        return 0;
    }

    const ggml_tensor * embd_tensor = model->talker_cp_codec_embd[codebook_idx];
    if (!embd_tensor) {
        fprintf(stderr, "Error: embedding tensor for codebook %d is null\n", codebook_idx);
        return 0;
    }

    // Copy embedding table to CPU
    std::vector<float> embd_data;
    if (!copy_tensor_to_cpu(embd_tensor, embd_data)) {
        fprintf(stderr, "Error: failed to copy embedding tensor\n");
        return 0;
    }

    // Get the embedding for input_token
    std::vector<float> cur(cp_n_embd);
    if (input_token < 0 || input_token >= cp_vocab) {
        fprintf(stderr, "Warning: input_token %d out of range, clamping\n", input_token);
        input_token = std::max(0, std::min(input_token, cp_vocab - 1));
    }
    for (int i = 0; i < cp_n_embd; ++i) {
        cur[i] = embd_data[input_token * cp_n_embd + i];
    }

    // Run through 5 transformer layers
    std::vector<float> residual(cp_n_embd);
    std::vector<float> normed(cp_n_embd);
    std::vector<float> attn_out(cp_n_embd);
    std::vector<float> ffn_out(cp_n_embd);
    std::vector<float> scratch(cp_n_ff);

    for (int il = 0; il < cp_n_layer; ++il) {
        const auto & layer = model->talker_cp_layers[il];

        // Copy weights to CPU (if not already there)
        std::vector<float> attn_norm_w, wq, wk, wv, wo, q_norm_w, k_norm_w;
        std::vector<float> ffn_norm_w, ffn_gate, ffn_up, ffn_down;

        if (!copy_tensor_to_cpu(layer.attn_norm, attn_norm_w) ||
            !copy_tensor_to_cpu(layer.wq, wq) ||
            !copy_tensor_to_cpu(layer.wk, wk) ||
            !copy_tensor_to_cpu(layer.wv, wv) ||
            !copy_tensor_to_cpu(layer.wo, wo) ||
            !copy_tensor_to_cpu(layer.attn_q_norm, q_norm_w) ||
            !copy_tensor_to_cpu(layer.attn_k_norm, k_norm_w) ||
            !copy_tensor_to_cpu(layer.ffn_norm, ffn_norm_w) ||
            !copy_tensor_to_cpu(layer.ffn_gate, ffn_gate) ||
            !copy_tensor_to_cpu(layer.ffn_up, ffn_up) ||
            !copy_tensor_to_cpu(layer.ffn_down, ffn_down)) {
            fprintf(stderr, "Error: failed to copy layer %d weights\n", il);
            return 0;
        }

        // Save residual
        memcpy(residual.data(), cur.data(), cp_n_embd * sizeof(float));

        // Attention norm
        rms_norm(cur.data(), attn_norm_w.data(), normed.data(), cp_n_embd);

        // Self-attention
        simple_self_attention_single_token(
            normed.data(), cp_n_embd, cp_n_head, cp_n_head_kv, cp_head_dim,
            wq.data(), wk.data(), wv.data(), wo.data(),
            q_norm_w.data(), k_norm_w.data(),
            attn_out.data());

        // Add residual
        for (int i = 0; i < cp_n_embd; ++i) {
            cur[i] = residual[i] + attn_out[i];
        }

        // Save residual for FFN
        memcpy(residual.data(), cur.data(), cp_n_embd * sizeof(float));

        // FFN norm
        rms_norm(cur.data(), ffn_norm_w.data(), normed.data(), cp_n_embd);

        // SwiGLU FFN
        swiglu_ffn(normed.data(), cp_n_embd, cp_n_ff,
                   ffn_gate.data(), ffn_up.data(), ffn_down.data(),
                   ffn_out.data(), scratch.data());

        // Add residual
        for (int i = 0; i < cp_n_embd; ++i) {
            cur[i] = residual[i] + ffn_out[i];
        }
    }

    // Apply output norm
    std::vector<float> output_norm_w;
    if (!copy_tensor_to_cpu(model->talker_cp_output_norm, output_norm_w)) {
        fprintf(stderr, "Error: failed to copy output norm\n");
        return 0;
    }
    rms_norm(cur.data(), output_norm_w.data(), normed.data(), cp_n_embd);

    // Apply LM head
    std::vector<float> lm_head_w;
    if (codebook_idx >= (int)model->talker_cp_lm_head.size()) {
        fprintf(stderr, "Error: LM head %d out of range\n", codebook_idx);
        return 0;
    }
    if (!copy_tensor_to_cpu(model->talker_cp_lm_head[codebook_idx], lm_head_w)) {
        fprintf(stderr, "Error: failed to copy LM head\n");
        return 0;
    }

    std::vector<float> logits(cp_vocab);
    matmul(normed.data(), lm_head_w.data(), logits.data(), 1, cp_n_embd, cp_vocab);

    // Argmax
    int best_token = 0;
    float best_logit = logits[0];
    for (int i = 1; i < cp_vocab; ++i) {
        if (logits[i] > best_logit) {
            best_logit = logits[i];
            best_token = i;
        }
    }

    return best_token;
}

// Run Code Predictor inline during Talker generation
// Reference: HuggingFace modeling_qwen3_omni_moe.py lines 3237-3256
//
// HuggingFace sums 16 CODEC EMBEDDINGS (not transformer hidden states):
//   - last_id_hidden: codebook 0 embedding (from Talker tok_embd)
//   - mid_residual_hiddens: codebook 1-14 embeddings (input embeddings from Code Predictor)
//   - last_residual_hidden: codebook 15 embedding (from final codebook embedding table)
//
// The Code Predictor generates 15 tokens, each from a different codebook embedding table.
// We collect all 16 embeddings (codebook 0-15) and sum them for feedback to Talker.
static bool run_code_predictor_inline(
    const llama_model * model,
    const std::vector<float> & past_hidden,    // [1024] from Talker prev step (for transformer context)
    const std::vector<float> & last_id_hidden, // [1024] codec token embedding (codebook 0)
    std::vector<std::vector<float>> & codec_embeddings, // Output: 15 codec embeddings (codebooks 1-15)
    std::vector<int> & codebook_tokens,        // Output: 15 tokens for codebooks 1-15
    std::mt19937 & rng,                        // RNG for sampling
    float temperature,                         // Sampling temperature (0 = greedy)
    bool verbose) {

    const int cp_n_embd = 1024;
    const int cp_n_head = 16;
    const int cp_n_head_kv = 8;
    const int cp_head_dim = 128;
    const int cp_n_ff = 3072;
    const int cp_n_layer = 5;
    const int cp_vocab = 2048;
    const int n_codebooks = 15; // Codebooks 1-15 (codebook 0 is last_id_hidden)
    const int max_seq_len = 20; // 2 input tokens + up to 18 generated
    const float cp_rope_theta = 1000000.0f; // RoPE base frequency from model config

    codec_embeddings.clear();
    codec_embeddings.reserve(n_codebooks);
    codebook_tokens.clear();
    codebook_tokens.reserve(n_codebooks);

    const struct llama_model * m = model;

    // KV cache for each layer: [max_seq, n_kv_head, head_dim]
    std::vector<std::vector<float>> kv_cache_k(cp_n_layer,
        std::vector<float>(max_seq_len * cp_n_head_kv * cp_head_dim, 0.0f));
    std::vector<std::vector<float>> kv_cache_v(cp_n_layer,
        std::vector<float>(max_seq_len * cp_n_head_kv * cp_head_dim, 0.0f));

    // Buffers for transformer computation
    std::vector<float> cur(cp_n_embd);
    std::vector<float> residual(cp_n_embd);
    std::vector<float> normed(cp_n_embd);
    std::vector<float> attn_out(cp_n_embd);
    std::vector<float> ffn_out(cp_n_embd);
    std::vector<float> scratch(cp_n_ff);

    // Pre-copy output norm (shared across all steps)
    std::vector<float> output_norm_w;
    if (!copy_tensor_to_cpu(m->talker_cp_output_norm, output_norm_w)) {
        fprintf(stderr, "Error: failed to copy Code Predictor output norm\n");
        return false;
    }

    // Pre-copy all layer weights (to avoid repeated copies)
    struct LayerWeights {
        std::vector<float> attn_norm_w, wq, wk, wv, wo, q_norm_w, k_norm_w;
        std::vector<float> ffn_norm_w, ffn_gate, ffn_up, ffn_down;
    };
    std::vector<LayerWeights> layer_weights(cp_n_layer);

    for (int il = 0; il < cp_n_layer; ++il) {
        const auto & layer = m->talker_cp_layers[il];
        auto & lw = layer_weights[il];

        if (!copy_tensor_to_cpu(layer.attn_norm, lw.attn_norm_w) ||
            !copy_tensor_to_cpu(layer.wq, lw.wq) ||
            !copy_tensor_to_cpu(layer.wk, lw.wk) ||
            !copy_tensor_to_cpu(layer.wv, lw.wv) ||
            !copy_tensor_to_cpu(layer.wo, lw.wo) ||
            !copy_tensor_to_cpu(layer.attn_q_norm, lw.q_norm_w) ||
            !copy_tensor_to_cpu(layer.attn_k_norm, lw.k_norm_w) ||
            !copy_tensor_to_cpu(layer.ffn_norm, lw.ffn_norm_w) ||
            !copy_tensor_to_cpu(layer.ffn_gate, lw.ffn_gate) ||
            !copy_tensor_to_cpu(layer.ffn_up, lw.ffn_up) ||
            !copy_tensor_to_cpu(layer.ffn_down, lw.ffn_down)) {
            fprintf(stderr, "Error: failed to copy Code Predictor layer %d weights\n", il);
            return false;
        }
    }

    // Helper to run one token through transformer at position `pos`
    auto run_transformer_step = [&](const std::vector<float>& input, int pos) {
        cur = input;
        for (int il = 0; il < cp_n_layer; ++il) {
            const auto & lw = layer_weights[il];

            // Save residual
            residual = cur;

            // Attention norm
            rms_norm(cur.data(), lw.attn_norm_w.data(), normed.data(), cp_n_embd);

            // Self-attention with KV cache and RoPE
            causal_attention_with_kv_cache(
                normed.data(), cp_n_embd, cp_n_head, cp_n_head_kv, cp_head_dim, pos,
                lw.wq.data(), lw.wk.data(), lw.wv.data(), lw.wo.data(),
                lw.q_norm_w.data(), lw.k_norm_w.data(),
                cp_rope_theta,  // RoPE with theta=1000000
                kv_cache_k[il].data(), kv_cache_v[il].data(),
                attn_out.data());

            // Add residual
            for (int i = 0; i < cp_n_embd; ++i) {
                cur[i] = residual[i] + attn_out[i];
            }

            // Save residual for FFN
            residual = cur;

            // FFN norm
            rms_norm(cur.data(), lw.ffn_norm_w.data(), normed.data(), cp_n_embd);

            // SwiGLU FFN
            swiglu_ffn(normed.data(), cp_n_embd, cp_n_ff,
                       lw.ffn_gate.data(), lw.ffn_up.data(), lw.ffn_down.data(),
                       ffn_out.data(), scratch.data());

            // Add residual
            for (int i = 0; i < cp_n_embd; ++i) {
                cur[i] = residual[i] + ffn_out[i];
            }
        }
    };

    // Reference: inputs_embeds = torch.cat((past_hidden, last_id_hidden), dim=1)
    // Process 2-token input sequence through transformer:
    // Position 0: past_hidden (Talker hidden state)
    // Position 1: last_id_hidden (codec token embedding)
    run_transformer_step(past_hidden, 0);
    run_transformer_step(last_id_hidden, 1);

    // Generate 15 codebook tokens autoregressively starting at position 2
    for (int cb = 0; cb < n_codebooks; ++cb) {
        int pos = cb + 2;  // Positions 2-16

        // Apply output norm to current hidden state
        rms_norm(cur.data(), output_norm_w.data(), normed.data(), cp_n_embd);

        // Apply LM head for this codebook
        if (cb >= (int)m->talker_cp_lm_head.size()) {
            fprintf(stderr, "Error: LM head %d out of range\n", cb);
            return false;
        }
        std::vector<float> lm_head_w;
        if (!copy_tensor_to_cpu(m->talker_cp_lm_head[cb], lm_head_w)) {
            fprintf(stderr, "Error: failed to copy LM head %d\n", cb);
            return false;
        }

        std::vector<float> logits(cp_vocab);
        matmul(normed.data(), lm_head_w.data(), logits.data(), 1, cp_n_embd, cp_vocab);

        // Match HuggingFace Code Predictor sampling: top_k=50, top_p=0.8, temp=0.9 (or greedy if temp=0)
        // Note: Code Predictor uses different params than Talker (which uses top_p=1.0)
        float cp_temp = (temperature <= 0.0f) ? 0.0f : 0.9f;  // Greedy if main temp is 0
        int best_token = sample_token(logits.data(), cp_vocab, cp_temp, 50, rng, {}, 1.0f, 0.8f);
        codebook_tokens.push_back(best_token);

        // Get embedding for this token from codebook embedding table
        if (cb < (int)m->talker_cp_codec_embd.size()) {
            const ggml_tensor * embd_tensor = m->talker_cp_codec_embd[cb];
            std::vector<float> embd_data;
            if (copy_tensor_to_cpu(embd_tensor, embd_data)) {
                std::vector<float> token_embd(cp_n_embd);
                if (best_token >= 0 && best_token < cp_vocab) {
                    for (int i = 0; i < cp_n_embd; ++i) {
                        token_embd[i] = embd_data[best_token * cp_n_embd + i];
                    }
                }

                // Collect INPUT EMBEDDING before transformer step
                // Key insight: HuggingFace's generate() with output_hidden_states=True returns:
                //   hidden_states[token_idx] = tuple of (num_layers + 1,) states
                //   where hid[0] = INPUT EMBEDDING (before any layer processing)
                //   and hid[1-N] = outputs of layers 0 to N-1
                // HuggingFace: mid_residual_hiddens = [hid[0] for hid in hidden_states[1:]]
                // This collects INPUT EMBEDDINGS for each generated token (skipping prefill)
                codec_embeddings.push_back(token_embd);  // INPUT embedding, not layer output

                // Run transformer to get logits for next token sampling
                run_transformer_step(token_embd, pos);
            }
        }
    }

    // Add last_residual_hidden: embedding of last token from the LAST table (index 14 in 15-table model)
    // Reference: last_residual_hidden = self.code_predictor.get_input_embeddings()[-1](sequences[-1])
    // HuggingFace sums 17 embeddings: last_id_hidden (1) + mid_residual_hiddens (15) + last_residual_hidden (1)
    // We collected 15 input embeddings above. Now add the last_residual_hidden (1 more).
    // The model has 15 embedding tables (0-14), so [-1] means table 14
    if (!codebook_tokens.empty() && !m->talker_cp_codec_embd.empty()) {
        int last_table_idx = (int)m->talker_cp_codec_embd.size() - 1;  // Table 14 (last table)
        const ggml_tensor * last_embd_table = m->talker_cp_codec_embd[last_table_idx];
        std::vector<float> last_embd_data;
        if (copy_tensor_to_cpu(last_embd_table, last_embd_data)) {
            int last_token = codebook_tokens.back();
            std::vector<float> last_residual_hidden(cp_n_embd);
            if (last_token >= 0 && last_token < cp_vocab) {
                for (int i = 0; i < cp_n_embd; ++i) {
                    last_residual_hidden[i] = last_embd_data[last_token * cp_n_embd + i];
                }
            }
            codec_embeddings.push_back(last_residual_hidden);  // 16 embeddings (15 input embeds + 1 last_residual)
        }
    }

    if (verbose) {
        fprintf(stderr, "  Code Predictor inline: generated %zu codebook tokens, %zu embeddings\n",
                codebook_tokens.size(), codec_embeddings.size());
    }

    return true;
}

// Helper to get tensor data as float pointer
static const float * get_tensor_data_f32(const ggml_tensor * t) {
    if (!t) return nullptr;
    if (t->type == GGML_TYPE_F32) {
        return (const float *)t->data;
    }
    return nullptr;
}

// Apply text projection MLP: input (2048) -> fc1 (2048) -> GELU -> fc2 (1024)
// Uses simple CPU matrix multiplication
static bool apply_text_projection(
        const llama_model * model,
        const float * input,
        float * output,
        int n_tokens,
        bool verbose) {

    // Access internal model structure
    const struct llama_model * m = model;

    if (!m->talker_text_proj_fc1 || !m->talker_text_proj_fc2) {
        fprintf(stderr, "Warning: Text projection tensors not found in model\n");
        return false;
    }

    const int n_embd_in = 2048;   // Thinker output dimension
    const int n_hidden = 2048;    // Hidden dimension (same as input for this MLP)
    const int n_embd_out = 1024;  // Talker input dimension

    // Verify tensor dimensions
    if (verbose) {
        printf("Text projection fc1 shape: [%lld, %lld]\n",
               (long long)m->talker_text_proj_fc1->ne[0],
               (long long)m->talker_text_proj_fc1->ne[1]);
        printf("Text projection fc2 shape: [%lld, %lld]\n",
               (long long)m->talker_text_proj_fc2->ne[0],
               (long long)m->talker_text_proj_fc2->ne[1]);
    }

    // Copy weight data to CPU (handles both CPU and GPU tensors)
    std::vector<float> fc1_w_data, fc1_b_data, fc2_w_data, fc2_b_data;

    if (!copy_tensor_to_cpu(m->talker_text_proj_fc1, fc1_w_data) ||
        !copy_tensor_to_cpu(m->talker_text_proj_fc2, fc2_w_data)) {
        fprintf(stderr, "Warning: Failed to copy text projection weights\n");
        return false;
    }

    // Bias tensors are optional
    bool has_fc1_b = copy_tensor_to_cpu(m->talker_text_proj_fc1_b, fc1_b_data);
    bool has_fc2_b = copy_tensor_to_cpu(m->talker_text_proj_fc2_b, fc2_b_data);

    // Allocate intermediate buffer for hidden layer
    std::vector<float> hidden(n_tokens * n_hidden);

    // fc1: [n_tokens, n_embd_in] @ [n_embd_in, n_hidden]^T = [n_tokens, n_hidden]
    // In ggml, weights are stored as [out_features, in_features]
    for (int t = 0; t < n_tokens; ++t) {
        for (int h = 0; h < n_hidden; ++h) {
            float sum = 0.0f;
            for (int i = 0; i < n_embd_in; ++i) {
                // fc1_w[h * n_embd_in + i] is weight for output h, input i
                sum += input[t * n_embd_in + i] * fc1_w_data[h * n_embd_in + i];
            }
            // Add bias if present
            if (has_fc1_b) {
                sum += fc1_b_data[h];
            }
            // Apply SiLU activation (Qwen3-Omni uses SiLU, not GELU)
            hidden[t * n_hidden + h] = silu(sum);
        }
    }

    // fc2: [n_tokens, n_hidden] @ [n_hidden, n_embd_out]^T = [n_tokens, n_embd_out]
    for (int t = 0; t < n_tokens; ++t) {
        for (int o = 0; o < n_embd_out; ++o) {
            float sum = 0.0f;
            for (int h = 0; h < n_hidden; ++h) {
                sum += hidden[t * n_hidden + h] * fc2_w_data[o * n_hidden + h];
            }
            // Add bias if present
            if (has_fc2_b) {
                sum += fc2_b_data[o];
            }
            output[t * n_embd_out + o] = sum;
        }
    }

    return true;
}

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --thinker PATH      Path to Thinker GGUF model (required)\n");
    fprintf(stderr, "  --talker PATH       Path to Talker GGUF model (optional)\n");
    fprintf(stderr, "  -p, --prompt TEXT   Input text to synthesize (required)\n");
    fprintf(stderr, "  -o, --output PATH   Output WAV file (default: output.wav)\n");
    fprintf(stderr, "  --layer N           Extract from Thinker at layer N (default: 24)\n");
    fprintf(stderr, "  --max-tokens N      Maximum codec tokens to generate (default: 500)\n");
    fprintf(stderr, "  --temp T            Sampling temperature (default: 0.9)\n");
    fprintf(stderr, "  --top-k K           Top-k sampling (default: 50)\n");
    fprintf(stderr, "  -s, --seed N        Random seed (-1 for random, default: -1)\n");
    fprintf(stderr, "  --thinker-only      Only run Thinker extraction (for testing)\n");
    fprintf(stderr, "  --skip-code2wav     Skip Code2Wav (output codec tokens only)\n");
    fprintf(stderr, "  -ngl N              Number of GPU layers (default: 99)\n");
    fprintf(stderr, "  --no-mmap           Disable mmap (recommended for UMA systems)\n");
    fprintf(stderr, "  --dump-tokens FILE  Dump codec tokens to file (for HF comparison)\n");
    fprintf(stderr, "  --load-tokens FILE  Load 16-codebook tokens from file (skip Talker)\n");
    fprintf(stderr, "  --dump-tensors DIR  Dump intermediate tensors to directory\n");
    fprintf(stderr, "  --c2w-cpu           Force CPU for Code2Wav (workaround for CUDA issues)\n");
    fprintf(stderr, "  -v, --verbose       Enable verbose output\n");
    fprintf(stderr, "  -h, --help          Show this help\n");
}

static bool parse_args(int argc, char ** argv, tts_params & params) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--thinker") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing thinker path\n");
                return false;
            }
            params.thinker_path = argv[i];
        } else if (arg == "--talker") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing talker path\n");
                return false;
            }
            params.talker_path = argv[i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing prompt\n");
                return false;
            }
            params.prompt = argv[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing output path\n");
                return false;
            }
            params.output_path = argv[i];
        } else if (arg == "--layer") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing layer number\n");
                return false;
            }
            params.n_layer_output = std::atoi(argv[i]);
        } else if (arg == "--max-tokens") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing max tokens\n");
                return false;
            }
            params.max_codec_tokens = std::atoi(argv[i]);
        } else if (arg == "--temp") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing temperature\n");
                return false;
            }
            params.temperature = std::atof(argv[i]);
        } else if (arg == "--top-k") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing top-k\n");
                return false;
            }
            params.top_k = std::atoi(argv[i]);
        } else if (arg == "--seed" || arg == "-s") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing seed\n");
                return false;
            }
            params.seed = std::atoi(argv[i]);
        } else if (arg == "--thinker-only") {
            params.thinker_only = true;
        } else if (arg == "--skip-code2wav") {
            params.skip_code2wav = true;
        } else if (arg == "--no-mmap") {
            params.use_mmap = false;
        } else if (arg == "--dump-tokens") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing dump-tokens path\n");
                return false;
            }
            params.dump_tokens_path = argv[i];
        } else if (arg == "--load-tokens") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing load-tokens path\n");
                return false;
            }
            params.load_tokens_path = argv[i];
        } else if (arg == "--dump-tensors") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing dump-tensors path\n");
                return false;
            }
            params.dump_tensors_path = argv[i];
        } else if (arg == "-ngl") {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing GPU layers\n");
                return false;
            }
            params.n_gpu_layers = std::atoi(argv[i]);
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else if (arg == "--c2w-cpu") {
            params.c2w_cpu_only = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Error: Unknown option: %s\n", arg.c_str());
            return false;
        }
    }

    // If --load-tokens is provided with a Talker model, we can skip Thinker entirely
    if (!params.load_tokens_path.empty() && !params.talker_path.empty()) {
        params.code2wav_only = true;
        printf("Code2Wav-only mode: loading tokens from %s\n", params.load_tokens_path.c_str());
    } else {
        // Normal mode - require Thinker and prompt
        if (params.thinker_path.empty()) {
            fprintf(stderr, "Error: Thinker model path is required (--thinker)\n");
            return false;
        }
        if (params.prompt.empty()) {
            fprintf(stderr, "Error: Prompt is required (-p) unless using --load-tokens with --talker\n");
            return false;
        }
    }
    if (!params.thinker_only && !params.code2wav_only && params.talker_path.empty()) {
        fprintf(stderr, "Note: No Talker model specified, running in thinker-only mode\n");
        params.thinker_only = true;
    }

    return true;
}

int main(int argc, char ** argv) {
    tts_params params;
    if (!parse_args(argc, argv, params)) {
        print_usage(argv[0]);
        return 1;
    }

    // Initialize RNG for sampling
    std::mt19937 rng;
    if (params.seed >= 0) {
        rng.seed(params.seed);
        printf("Using fixed seed: %d\n", params.seed);
    } else {
        std::random_device rd;
        rng.seed(rd());
    }

    // Initialize backends and llama
    ggml_backend_load_all();
    llama_backend_init();

    printf("\n=== Qwen3-Omni TTS Pipeline ===\n\n");
    fflush(stdout);

    // ========================================
    // Code2Wav-only mode: skip Thinker and Talker, just run vocoder
    // ========================================
    if (params.code2wav_only) {
        printf("Loading Talker model (Code2Wav only): %s\n", params.talker_path.c_str());

        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = params.n_gpu_layers;
        mparams.use_mmap = params.use_mmap;

        llama_model * talker_model = llama_model_load_from_file(params.talker_path.c_str(), mparams);
        if (!talker_model) {
            fprintf(stderr, "Error: Failed to load Talker model\n");
            return 1;
        }

        // Load tokens from file
        auto all_codebook_tokens = load_codebook_tokens(params.load_tokens_path);
        if (all_codebook_tokens.empty()) {
            fprintf(stderr, "Error: Failed to load tokens from %s\n", params.load_tokens_path.c_str());
            llama_model_free(talker_model);
            return 1;
        }
        printf("Loaded %zu frames from %s\n", all_codebook_tokens.size(), params.load_tokens_path.c_str());

        // Run Code2Wav
        printf("\nRunning Code2Wav vocoder...\n");
        fflush(stdout);

        if (!talker_model->c2w_code_embd || talker_model->c2w_pre_layers.empty()) {
            fprintf(stderr, "Error: Code2Wav tensors not found in model\n");
            llama_model_free(talker_model);
            return 1;
        }

        printf("Code2Wav has %zu pre-transformer layers\n", talker_model->c2w_pre_layers.size());

        std::vector<float> audio_samples = run_code2wav_ggml(talker_model, all_codebook_tokens, params.verbose, params.dump_tensors_path, params.c2w_cpu_only);

        int n_samples = audio_samples.size();

        // Apply fade in/out
        int fade_samples = std::min(500, n_samples / 8);
        for (int i = 0; i < fade_samples; ++i) {
            float fade = static_cast<float>(i) / fade_samples;
            audio_samples[i] *= fade;
            audio_samples[n_samples - 1 - i] *= fade;
        }

        if (write_wav(params.output_path, audio_samples.data(), n_samples, params.sample_rate)) {
            printf("\nWrote audio to: %s\n", params.output_path.c_str());
            printf("Duration: %.2f seconds (%d samples at %d Hz)\n",
                   static_cast<float>(n_samples) / params.sample_rate, n_samples, params.sample_rate);
        }

        llama_model_free(talker_model);
        llama_backend_free();
        return 0;
    }

    // ========================================
    // Step 1: Load Thinker model
    // ========================================
    printf("Loading Thinker model: %s\n", params.thinker_path.c_str());

    llama_model_params thinker_mparams = llama_model_default_params();
    thinker_mparams.n_gpu_layers = params.n_gpu_layers;
    thinker_mparams.use_mmap = params.use_mmap;

    llama_model * thinker_model = llama_model_load_from_file(params.thinker_path.c_str(), thinker_mparams);
    if (!thinker_model) {
        fprintf(stderr, "Error: Failed to load Thinker model\n");
        return 1;
    }

    int n_embd_thinker = llama_model_n_embd(thinker_model);
    printf("Thinker hidden dim: %d\n", n_embd_thinker);

    // Create Thinker context for full generation
    // NOTE: n_layer_output=0 means run all layers including lm_head for text generation
    // Embedding extraction is done separately by reading tok_embd directly (not hidden states)
    llama_context_params thinker_cparams = llama_context_default_params();
    thinker_cparams.n_ctx = 2048;
    thinker_cparams.n_batch = 512;
    thinker_cparams.embeddings = true;  // We want embeddings output
    thinker_cparams.n_layer_output = 0;  // Full generation - run all 48 layers + lm_head

    printf("Running Thinker with full generation (all layers + lm_head)\n");

    llama_context * thinker_ctx = llama_init_from_model(thinker_model, thinker_cparams);
    if (!thinker_ctx) {
        fprintf(stderr, "Error: Failed to create Thinker context\n");
        llama_model_free(thinker_model);
        return 1;
    }

    // ========================================
    // Step 2: Format and tokenize as chat message
    // ========================================
    const llama_vocab * vocab = llama_model_get_vocab(thinker_model);

    // Qwen chat format special tokens
    const llama_token IM_START_ID = 151644;  // <|im_start|>
    const llama_token IM_END_ID = 151645;    // <|im_end|>

    // TTS special tokens from config - will be read from Thinker's tok_embd (not run through model)
    // These are Thinker vocab tokens that get projected to Talker space
    const int TTS_PAD_TOKEN_ID = 151671;  // tts_pad_token_id
    const int TTS_BOS_TOKEN_ID = 151672;  // tts_bos_token_id
    const int TTS_EOS_TOKEN_ID = 151673;  // tts_eos_token_id - signals end of text to Talker

    // Format as simple chat prompt (matches HuggingFace debug_hf_talker.py)
    // The Thinker generates a text response, and those token embeddings go to Talker
    // DO NOT include TTS tokens (<tts_text_bos>, <|audio_start|>, etc.) - they confuse the Thinker
    std::string chat_prompt = "<|im_start|>user\n" + params.prompt + "<|im_end|>\n<|im_start|>assistant\n";

    printf("\nFormatted chat prompt:\n%s\n", chat_prompt.c_str());

    std::vector<llama_token> prompt_tokens(chat_prompt.size() + 64);
    int n_prompt_tokens = llama_tokenize(vocab, chat_prompt.c_str(), chat_prompt.size(),
                                   prompt_tokens.data(), prompt_tokens.size(), true, true);
    if (n_prompt_tokens < 0) {
        fprintf(stderr, "Error: Tokenization failed\n");
        llama_free(thinker_ctx);
        llama_model_free(thinker_model);
        return 1;
    }
    prompt_tokens.resize(n_prompt_tokens);

    printf("Prompt tokens: %d\n", n_prompt_tokens);

    if (params.verbose) {
        printf("Prompt Token IDs: [");
        for (int i = 0; i < n_prompt_tokens; ++i) {
            printf("%d%s", prompt_tokens[i], i < n_prompt_tokens - 1 ? ", " : "");
        }
        printf("]\n");
    }

    // Find assistant start position in prompt (after last <|im_start|>)
    int assistant_start_pos = -1;
    for (int i = n_prompt_tokens - 1; i >= 0; --i) {
        if (prompt_tokens[i] == IM_START_ID) {
            assistant_start_pos = i;
            break;
        }
    }
    printf("Assistant segment starts at position: %d\n", assistant_start_pos);

    // ========================================
    // Step 3: Run Thinker generation
    // ========================================
    printf("\nRunning Thinker generation...\n");

    // Set up sampler for Thinker (standard text generation)
    llama_sampler * thinker_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(thinker_sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(thinker_sampler, llama_sampler_init_temp(0.7f));  // Standard text temp
    llama_sampler_chain_add(thinker_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // Process prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_decode(thinker_ctx, batch) != 0) {
        fprintf(stderr, "Error: Failed to decode prompt\n");
        llama_sampler_free(thinker_sampler);
        llama_free(thinker_ctx);
        llama_model_free(thinker_model);
        return 1;
    }

    // Collect all tokens (prompt + generated) for embedding extraction
    std::vector<llama_token> all_tokens = prompt_tokens;
    std::string generated_text;

    // Generation loop
    const int MAX_THINKER_TOKENS = 256;  // Max generated tokens
    for (int gen_i = 0; gen_i < MAX_THINKER_TOKENS; ++gen_i) {
        llama_token new_token = llama_sampler_sample(thinker_sampler, thinker_ctx, -1);

        // Check for end of generation
        if (new_token == IM_END_ID || llama_vocab_is_eog(vocab, new_token)) {
            all_tokens.push_back(new_token);  // Include end token
            break;
        }

        all_tokens.push_back(new_token);

        // Convert token to text for display
        char buf[256];
        int len = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, false);
        if (len > 0) {
            generated_text.append(buf, len);
            printf("%.*s", len, buf);
            fflush(stdout);
        }

        // Prepare next batch
        batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(thinker_ctx, batch) != 0) {
            fprintf(stderr, "\nError: Failed to decode token %d\n", gen_i);
            break;
        }
    }
    printf("\n");

    llama_sampler_free(thinker_sampler);

    int n_generated = (int)all_tokens.size() - n_prompt_tokens;
    printf("\nGenerated %d tokens: \"%s\"\n", n_generated, generated_text.c_str());

    // Find assistant end position (the <|im_end|> after generation)
    int assistant_end_pos = (int)all_tokens.size();
    if (!all_tokens.empty() && all_tokens.back() == IM_END_ID) {
        assistant_end_pos = (int)all_tokens.size() - 1;  // Exclude the end token
    }

    // The assistant segment is from assistant_start_pos to assistant_end_pos
    // This includes: <|im_start|>assistant\n{generated_text}
    // Per HuggingFace: we want the tokens AFTER <|im_start|>assistant\n
    // which is positions [assistant_start_pos + 2, assistant_end_pos) approximately

    int n_all_tokens = (int)all_tokens.size();
    printf("Total sequence: %d tokens (prompt: %d, generated: %d)\n",
           n_all_tokens, n_prompt_tokens, n_generated);
    printf("Assistant segment: positions %d to %d\n", assistant_start_pos, assistant_end_pos);

    // ========================================
    // Step 4: Extract token embeddings for entire sequence
    // ========================================
    printf("\nExtracting token embeddings (layer 0) from Thinker tok_embd...\n");

    std::vector<float> embeddings_copy(n_all_tokens * n_embd_thinker);

    {
        struct ggml_tensor * thinker_tok_embd = thinker_model->tok_embd;
        if (!thinker_tok_embd) {
            fprintf(stderr, "Error: Thinker model has no tok_embd tensor\n");
            llama_free(thinker_ctx);
            llama_model_free(thinker_model);
            return 1;
        }

        size_t elem_size = ggml_type_size(thinker_tok_embd->type);

        for (int t = 0; t < n_all_tokens; ++t) {
            int token_id = all_tokens[t];
            size_t row_offset = (size_t)token_id * n_embd_thinker;
            size_t byte_offset = row_offset * elem_size;
            float * dst = &embeddings_copy[t * n_embd_thinker];

            if (thinker_tok_embd->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(thinker_tok_embd, dst,
                                        byte_offset, n_embd_thinker * sizeof(float));
            } else if (thinker_tok_embd->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> f16_buf(n_embd_thinker);
                ggml_backend_tensor_get(thinker_tok_embd, f16_buf.data(),
                                        byte_offset, n_embd_thinker * sizeof(ggml_fp16_t));
                for (int i = 0; i < n_embd_thinker; ++i) {
                    dst[i] = ggml_fp16_to_fp32(f16_buf[i]);
                }
            }
        }
        printf("Extracted token embeddings for %d tokens\n", n_all_tokens);
    }

    // For Talker, we need the assistant segment embeddings
    // HuggingFace: assistant_hidden = text_projection(thinker_embed[:, im_start_index:segment_end_index])
    int n_assistant_tokens = assistant_end_pos - assistant_start_pos;
    printf("Assistant segment has %d tokens for TTS\n", n_assistant_tokens);

    // Print first few values for debugging
    if (params.verbose) {
        printf("First 8 values of first token embedding: [");
        for (int i = 0; i < 8 && i < n_embd_thinker; ++i) {
            printf("%.4f%s", embeddings_copy[i], i < 7 ? ", " : "");
        }
        printf("]\n");

        // Compute stats
        float sum = 0, sum_sq = 0;
        for (int i = 0; i < n_all_tokens * n_embd_thinker; ++i) {
            sum += embeddings_copy[i];
            sum_sq += embeddings_copy[i] * embeddings_copy[i];
        }
        float mean = sum / (n_all_tokens * n_embd_thinker);
        float var = sum_sq / (n_all_tokens * n_embd_thinker) - mean * mean;
        printf("Token embedding stats: mean=%.4f, std=%.4f\n", mean, sqrtf(var));
    }

    // Save embeddings for comparison with Python
    std::string embd_path = params.output_path + ".embeddings.bin";
    if (save_embeddings(embd_path, embeddings_copy.data(), n_all_tokens, n_embd_thinker)) {
        printf("Saved token embeddings to: %s\n", embd_path.c_str());
    }

    // Update variables for downstream use - use assistant segment
    std::vector<llama_token> tokens(all_tokens.begin() + assistant_start_pos, all_tokens.begin() + assistant_end_pos);
    int n_tokens = n_assistant_tokens;
    int n_text_tokens = n_tokens;

    // Create assistant segment embeddings for downstream projection
    // Slice embeddings_copy to just the assistant segment
    std::vector<float> assistant_embeddings(n_assistant_tokens * n_embd_thinker);
    for (int t = 0; t < n_assistant_tokens; ++t) {
        int src_idx = (assistant_start_pos + t) * n_embd_thinker;
        float * dst = &assistant_embeddings[t * n_embd_thinker];
        memcpy(dst, &embeddings_copy[src_idx], n_embd_thinker * sizeof(float));
    }

    // Replace embeddings_copy with assistant segment for downstream use
    embeddings_copy = std::move(assistant_embeddings);

    printf("Sliced embeddings for assistant segment: %d tokens x %d dims\n", n_tokens, n_embd_thinker);

    // Extract raw embedding for TTS_PAD_TOKEN_ID from Thinker's tok_embd
    // Per HuggingFace transformers: tts_pad_embed uses raw token embedding, NOT hidden state
    // Use ggml_backend_tensor_get() for safe GPU->CPU transfer (direct pointer access hangs on GPU tensors)
    std::vector<float> tts_pad_raw_embed(n_embd_thinker, 0.0f);
    {
        struct ggml_tensor * thinker_tok_embd = thinker_model->tok_embd;
        if (thinker_tok_embd && TTS_PAD_TOKEN_ID < (int64_t)thinker_tok_embd->ne[1]) {
            // tok_embd shape is [n_embd, n_vocab], row-major so token i is at offset i * n_embd
            size_t row_offset = (size_t)TTS_PAD_TOKEN_ID * n_embd_thinker;
            size_t elem_size = ggml_type_size(thinker_tok_embd->type);
            size_t byte_offset = row_offset * elem_size;

            if (thinker_tok_embd->type == GGML_TYPE_F32) {
                // Read directly as f32 using safe backend transfer
                ggml_backend_tensor_get(thinker_tok_embd, tts_pad_raw_embed.data(),
                                        byte_offset, n_embd_thinker * sizeof(float));
            } else if (thinker_tok_embd->type == GGML_TYPE_F16) {
                // Read as f16 and convert to f32
                std::vector<ggml_fp16_t> f16_buf(n_embd_thinker);
                ggml_backend_tensor_get(thinker_tok_embd, f16_buf.data(),
                                        byte_offset, n_embd_thinker * sizeof(ggml_fp16_t));
                for (int i = 0; i < n_embd_thinker; ++i) {
                    tts_pad_raw_embed[i] = ggml_fp16_to_fp32(f16_buf[i]);
                }
            }
            printf("Extracted raw TTS_PAD embedding from Thinker tok_embd (safe GPU transfer)\n");
            if (params.verbose) {
                printf("TTS_PAD raw embed first 8: [");
                for (int i = 0; i < 8 && i < n_embd_thinker; ++i) {
                    printf("%.4f%s", tts_pad_raw_embed[i], i < 7 ? ", " : "");
                }
                printf("]\n");
            }
        } else {
            fprintf(stderr, "Warning: Could not read TTS_PAD embedding from Thinker\n");
        }
    }

    // Extract raw embedding for TTS_EOS_TOKEN_ID - signals end of text to Talker
    // This is appended after text embeddings to tell Talker "text is done, wrap up audio"
    std::vector<float> tts_eos_raw_embed(n_embd_thinker, 0.0f);
    {
        struct ggml_tensor * thinker_tok_embd = thinker_model->tok_embd;
        if (thinker_tok_embd && TTS_EOS_TOKEN_ID < (int64_t)thinker_tok_embd->ne[1]) {
            size_t row_offset = (size_t)TTS_EOS_TOKEN_ID * n_embd_thinker;
            size_t elem_size = ggml_type_size(thinker_tok_embd->type);
            size_t byte_offset = row_offset * elem_size;

            if (thinker_tok_embd->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(thinker_tok_embd, tts_eos_raw_embed.data(),
                                        byte_offset, n_embd_thinker * sizeof(float));
            } else if (thinker_tok_embd->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> f16_buf(n_embd_thinker);
                ggml_backend_tensor_get(thinker_tok_embd, f16_buf.data(),
                                        byte_offset, n_embd_thinker * sizeof(ggml_fp16_t));
                for (int i = 0; i < n_embd_thinker; ++i) {
                    tts_eos_raw_embed[i] = ggml_fp16_to_fp32(f16_buf[i]);
                }
            }
            printf("Extracted raw TTS_EOS embedding from Thinker tok_embd\n");
        } else {
            fprintf(stderr, "Warning: Could not read TTS_EOS embedding from Thinker\n");
        }
    }

    if (params.thinker_only) {
        printf("\n=== Thinker-only mode complete ===\n");
        printf("Layer-%d hidden states extracted successfully.\n", params.n_layer_output);
        printf("You can compare %s with Python-extracted embeddings.\n", embd_path.c_str());
        llama_free(thinker_ctx);
        llama_model_free(thinker_model);
        llama_backend_free();
        return 0;
    }

    // ========================================
    // Step 4: Load Talker model
    // ========================================
    printf("\nLoading Talker model: %s\n", params.talker_path.c_str());

    llama_model_params talker_mparams = llama_model_default_params();
    talker_mparams.n_gpu_layers = params.n_gpu_layers;
    talker_mparams.use_mmap = params.use_mmap;

    llama_model * talker_model = llama_model_load_from_file(params.talker_path.c_str(), talker_mparams);
    if (!talker_model) {
        fprintf(stderr, "Error: Failed to load Talker model\n");
        llama_free(thinker_ctx);
        llama_model_free(thinker_model);
        return 1;
    }

    int n_embd_talker = llama_model_n_embd(talker_model);
    printf("Talker hidden dim: %d\n", n_embd_talker);

    // ========================================
    // Step 5: Apply text projection (2048 -> 1024)
    // ========================================
    printf("\nApplying text projection (2048 -> 1024)...\n");
    fflush(stdout);

    // Project text token hidden states (from layer 24)
    std::vector<float> projected_embeddings(n_text_tokens * n_embd_talker);
    // Project raw TTS special embeddings separately (per HuggingFace transformers)
    std::vector<float> tts_pad_embed(n_embd_talker, 0.0f);
    std::vector<float> tts_eos_embed(n_embd_talker, 0.0f);  // Signals end of text to Talker

    // Debug: print raw embedding norms before projection
    if (params.verbose) {
        printf("Raw embedding norms (before projection): ");
        for (int t = 0; t < n_text_tokens && t < 10; ++t) {
            float norm = 0.0f;
            for (int i = 0; i < n_embd_thinker; ++i) {
                float v = embeddings_copy[t * n_embd_thinker + i];
                norm += v * v;
            }
            printf("%.1f ", std::sqrt(norm));
        }
        if (n_text_tokens > 10) printf("...");
        printf("\n");
    }

    // Fix #32: Don't normalize token embeddings
    // HuggingFace uses tok_embd (layer 0) directly through text_projection without normalization.
    // The previous normalization was designed for layer 24 hidden states which have different magnitudes.
    // Token embeddings should be used as-is.
    printf("Using token embeddings without normalization (matching HuggingFace)\n");

    if (n_embd_thinker != n_embd_talker) {
        // Project text token hidden states
        if (!apply_text_projection(talker_model, embeddings_copy.data(),
                                    projected_embeddings.data(), n_text_tokens, params.verbose)) {
            fprintf(stderr, "Warning: Text projection failed, using truncated embeddings\n");
            for (int t = 0; t < n_text_tokens; ++t) {
                for (int i = 0; i < n_embd_talker; ++i) {
                    projected_embeddings[t * n_embd_talker + i] = embeddings_copy[t * n_embd_thinker + i];
                }
            }
        } else {
            printf("Text projection applied successfully\n");
        }

        // Project raw TTS_PAD embedding (uses input embedding, not hidden state)
        if (!apply_text_projection(talker_model, tts_pad_raw_embed.data(),
                                    tts_pad_embed.data(), 1, false)) {
            fprintf(stderr, "Warning: TTS_PAD projection failed\n");
        } else {
            printf("TTS_PAD embedding projected from raw tok_embd\n");
        }

        // Project raw TTS_EOS embedding - signals end of text to Talker
        if (!apply_text_projection(talker_model, tts_eos_raw_embed.data(),
                                    tts_eos_embed.data(), 1, false)) {
            fprintf(stderr, "Warning: TTS_EOS projection failed\n");
        } else {
            printf("TTS_EOS embedding projected from raw tok_embd\n");
        }

        // Fix #30: Don't scale special token embeddings
        // HuggingFace doesn't scale tts_pad/eos_embed after projection
        // They use the natural magnitude from text_projection
        // Scaling was causing special tokens to be too strong relative to expected
        if (n_text_tokens > 0) {
            float text_norm_sum = 0.0f;
            for (int t = 0; t < n_text_tokens; ++t) {
                float norm = 0.0f;
                for (int i = 0; i < n_embd_talker; ++i) {
                    float v = projected_embeddings[t * n_embd_talker + i];
                    norm += v * v;
                }
                text_norm_sum += std::sqrt(norm);
            }
            float avg_text_norm = text_norm_sum / n_text_tokens;

            float pad_norm = 0.0f;
            for (int i = 0; i < n_embd_talker; ++i) {
                pad_norm += tts_pad_embed[i] * tts_pad_embed[i];
            }
            pad_norm = std::sqrt(pad_norm);

            float eos_norm = 0.0f;
            for (int i = 0; i < n_embd_talker; ++i) {
                eos_norm += tts_eos_embed[i] * tts_eos_embed[i];
            }
            eos_norm = std::sqrt(eos_norm);

            printf("Text embedding avg norm: %.4f\n", avg_text_norm);
            printf("TTS_PAD embedding norm: %.4f (NOT scaled, matching HF)\n", pad_norm);
            printf("TTS_EOS embedding norm: %.4f (NOT scaled, matching HF)\n", eos_norm);
        }
    } else {
        // Dimensions match, just copy
        projected_embeddings.assign(embeddings_copy.begin(), embeddings_copy.begin() + n_text_tokens * n_embd_talker);
        tts_pad_embed.assign(tts_pad_raw_embed.begin(), tts_pad_raw_embed.begin() + n_embd_talker);
        tts_eos_embed.assign(tts_eos_raw_embed.begin(), tts_eos_raw_embed.begin() + n_embd_talker);
    }

    if (params.verbose) {
        printf("Projected text embeddings: [%d, %d]\n", n_text_tokens, n_embd_talker);
        printf("Text embed first 8: [");
        for (int i = 0; i < 8 && i < n_embd_talker; ++i) {
            printf("%.4f%s", projected_embeddings[i], i < 7 ? ", " : "");
        }
        printf("]\n");

        // Print norms of all text embeddings
        printf("Text embedding norms per token: ");
        for (int t = 0; t < n_text_tokens && t < 10; ++t) {
            float norm = 0.0f;
            for (int i = 0; i < n_embd_talker; ++i) {
                float v = projected_embeddings[t * n_embd_talker + i];
                norm += v * v;
            }
            printf("%.1f ", std::sqrt(norm));
        }
        if (n_text_tokens > 10) printf("...");
        printf("\n");

        printf("TTS_PAD embed first 8: [");
        for (int i = 0; i < 8 && i < n_embd_talker; ++i) {
            printf("%.4f%s", tts_pad_embed[i], i < 7 ? ", " : "");
        }
        printf("]\n");
    }

    // ========================================
    // Step 6: Autoregressive Talker generation
    // ========================================
    printf("\nRunning Talker autoregressive generation...\n");

    // Create Talker context with embeddings output for hidden state feedback
    llama_context_params talker_cparams = llama_context_default_params();
    talker_cparams.n_ctx = 2048;
    talker_cparams.n_batch = 512;
    talker_cparams.embeddings = true;  // Enable hidden state output

    llama_context * talker_ctx = llama_init_from_model(talker_model, talker_cparams);
    if (!talker_ctx) {
        fprintf(stderr, "Error: Failed to create Talker context\n");
        llama_model_free(talker_model);
        llama_free(thinker_ctx);
        llama_model_free(thinker_model);
        return 1;
    }

    // Enable debug layer outputs to preserve pre_norm_hidden tensor for Code Predictor
    // HuggingFace uses hidden_states[0][-1] which is BEFORE output_norm
    llama_set_debug_layer_outputs(talker_ctx, true);

    const llama_vocab * talker_vocab = llama_model_get_vocab(talker_model);
    int n_vocab = llama_vocab_n_tokens(talker_vocab);
    printf("Talker vocab size: %d\n", n_vocab);

    // Cache the token embedding matrix on CPU for residual injection
    // Get tok_embd tensor from model
    struct ggml_tensor * tok_embd = talker_model->tok_embd;
    if (!tok_embd) {
        fprintf(stderr, "Error: Talker model has no tok_embd tensor\n");
        llama_free(talker_ctx);
        llama_model_free(talker_model);
        llama_free(thinker_ctx);
        llama_model_free(thinker_model);
        return 1;
    }

    // tok_embd shape: [n_embd, n_vocab]
    int64_t embd_dim = tok_embd->ne[0];
    int64_t vocab_size = tok_embd->ne[1];
    printf("Talker tok_embd: [%lld, %lld] type=%d\n", (long long)embd_dim, (long long)vocab_size, tok_embd->type);

    if (embd_dim != n_embd_talker) {
        fprintf(stderr, "Warning: tok_embd dim %lld != n_embd_talker %d\n", (long long)embd_dim, n_embd_talker);
    }

    // Copy tok_embd to CPU for fast lookup during generation
    // Handle f16 tensors by reading raw bytes and converting
    std::vector<float> tok_embd_data(embd_dim * vocab_size);

    if (tok_embd->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tok_embd, tok_embd_data.data(), 0, ggml_nbytes(tok_embd));
    } else if (tok_embd->type == GGML_TYPE_F16) {
        // Read as f16 and convert to f32
        std::vector<ggml_fp16_t> tok_embd_f16(embd_dim * vocab_size);
        ggml_backend_tensor_get(tok_embd, tok_embd_f16.data(), 0, ggml_nbytes(tok_embd));
        for (size_t i = 0; i < tok_embd_f16.size(); ++i) {
            tok_embd_data[i] = ggml_fp16_to_fp32(tok_embd_f16[i]);
        }
    } else {
        fprintf(stderr, "Error: Unsupported tok_embd type %d\n", tok_embd->type);
        llama_free(talker_ctx);
        llama_model_free(talker_model);
        llama_free(thinker_ctx);
        llama_model_free(thinker_model);
        return 1;
    }

    // ========================================
    // Fix #31: Implement HuggingFace prefill structure
    // ========================================
    // HuggingFace builds a specific prefill sequence:
    // Position 0-2: text[0:3] + zeros             (just text embeddings)
    // Position 3:   tts_pad + nothink_embed       (codec token 2155)
    // Position 4:   tts_pad + think_bos_embed     (codec token 2156)
    // Position 5:   tts_pad + think_eos_embed     (codec token 2157)
    // Position 6:   tts_pad + speaker_id_embed    (ethan=2302, chelsie=2301, aiden=2303)
    // Position 7:   tts_bos + codec_pad_embed     (codec token 2148)
    // Position 8:   text[3] + codec_bos_embed     (codec token 2149)
    //
    // Then trailing_text_hidden = [text[4:], tts_eos_embed]
    // During generation step N, use trailing_text_hidden[N]

    // Speaker ID - hardcoded to Ethan for now
    const int SPEAKER_ID = 2302;  // ethan

    // Build prefill embeddings (9 positions)
    const int n_prefill = 9;
    std::vector<float> prefill_embeds(n_prefill * n_embd_talker, 0.0f);

    printf("Building prefill sequence (%d positions)...\n", n_prefill);

    // Helper to get codec embedding from tok_embd_data
    auto get_codec_embd = [&](int token_id, float * out) {
        if (token_id >= 0 && token_id < vocab_size) {
            for (int i = 0; i < n_embd_talker; ++i) {
                out[i] = tok_embd_data[token_id * embd_dim + i];
            }
        }
    };

    // Position 0-2: text[0:3] (just text embeddings, no codec)
    for (int p = 0; p < 3 && p < n_text_tokens; ++p) {
        float * dst = &prefill_embeds[p * n_embd_talker];
        for (int i = 0; i < n_embd_talker; ++i) {
            dst[i] = projected_embeddings[p * n_embd_talker + i];  // text embedding only
        }
    }

    // Position 3: tts_pad + nothink_embed
    {
        float * dst = &prefill_embeds[3 * n_embd_talker];
        std::vector<float> codec_embd(n_embd_talker);
        get_codec_embd(TALKER_CODEC_NOTHINK_ID, codec_embd.data());
        for (int i = 0; i < n_embd_talker; ++i) {
            dst[i] = tts_pad_embed[i] + codec_embd[i];
        }
    }

    // Position 4: tts_pad + think_bos_embed
    {
        float * dst = &prefill_embeds[4 * n_embd_talker];
        std::vector<float> codec_embd(n_embd_talker);
        get_codec_embd(TALKER_CODEC_THINK_BOS_ID, codec_embd.data());
        for (int i = 0; i < n_embd_talker; ++i) {
            dst[i] = tts_pad_embed[i] + codec_embd[i];
        }
    }

    // Position 5: tts_pad + think_eos_embed
    {
        float * dst = &prefill_embeds[5 * n_embd_talker];
        std::vector<float> codec_embd(n_embd_talker);
        get_codec_embd(TALKER_CODEC_THINK_EOS_ID, codec_embd.data());
        for (int i = 0; i < n_embd_talker; ++i) {
            dst[i] = tts_pad_embed[i] + codec_embd[i];
        }
    }

    // Position 6: tts_pad + speaker_id_embed
    {
        float * dst = &prefill_embeds[6 * n_embd_talker];
        std::vector<float> codec_embd(n_embd_talker);
        get_codec_embd(SPEAKER_ID, codec_embd.data());
        for (int i = 0; i < n_embd_talker; ++i) {
            dst[i] = tts_pad_embed[i] + codec_embd[i];
        }
    }

    // Position 7: tts_bos + codec_pad_embed
    // Extract and project TTS_BOS embedding from Thinker
    std::vector<float> tts_bos_embed(n_embd_talker, 0.0f);
    {
        struct ggml_tensor * thinker_tok_embd = thinker_model->tok_embd;
        if (thinker_tok_embd && TTS_BOS_TOKEN_ID < (int64_t)thinker_tok_embd->ne[1]) {
            std::vector<float> tts_bos_raw(n_embd_thinker);
            size_t row_offset = (size_t)TTS_BOS_TOKEN_ID * n_embd_thinker;
            size_t elem_size = ggml_type_size(thinker_tok_embd->type);
            size_t byte_offset = row_offset * elem_size;

            if (thinker_tok_embd->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(thinker_tok_embd, tts_bos_raw.data(),
                                        byte_offset, n_embd_thinker * sizeof(float));
            } else if (thinker_tok_embd->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> f16_buf(n_embd_thinker);
                ggml_backend_tensor_get(thinker_tok_embd, f16_buf.data(),
                                        byte_offset, n_embd_thinker * sizeof(ggml_fp16_t));
                for (int i = 0; i < n_embd_thinker; ++i) {
                    tts_bos_raw[i] = ggml_fp16_to_fp32(f16_buf[i]);
                }
            }

            // Project through text_projection MLP (same as tts_pad and tts_eos)
            if (!apply_text_projection(talker_model, tts_bos_raw.data(),
                                       tts_bos_embed.data(), 1, false)) {
                fprintf(stderr, "Warning: TTS_BOS projection failed\n");
            } else {
                printf("TTS_BOS embedding projected from raw tok_embd\n");
            }
        }
    }
    {
        float * dst = &prefill_embeds[7 * n_embd_talker];
        std::vector<float> codec_embd(n_embd_talker);
        get_codec_embd(TALKER_CODEC_PAD_ID, codec_embd.data());
        for (int i = 0; i < n_embd_talker; ++i) {
            dst[i] = tts_bos_embed[i] + codec_embd[i];
        }
    }

    // Position 8: text[3] + codec_bos_embed
    {
        float * dst = &prefill_embeds[8 * n_embd_talker];
        std::vector<float> codec_embd(n_embd_talker);
        get_codec_embd(TALKER_CODEC_BOS_ID, codec_embd.data());
        if (n_text_tokens > 3) {
            for (int i = 0; i < n_embd_talker; ++i) {
                dst[i] = projected_embeddings[3 * n_embd_talker + i] + codec_embd[i];
            }
        } else {
            // Fallback if not enough text tokens
            for (int i = 0; i < n_embd_talker; ++i) {
                dst[i] = tts_pad_embed[i] + codec_embd[i];
            }
        }
    }

    // Build trailing_text_hidden = [text[4:], tts_eos_embed]
    // This is used during autoregressive generation
    int n_trailing_text = (n_text_tokens > 4) ? (n_text_tokens - 4) : 0;
    int n_trailing = n_trailing_text + 1;  // +1 for tts_eos_embed at end
    std::vector<float> trailing_text_hidden(n_trailing * n_embd_talker);

    // Copy text[4:] to trailing_text_hidden
    for (int t = 0; t < n_trailing_text; ++t) {
        for (int i = 0; i < n_embd_talker; ++i) {
            trailing_text_hidden[t * n_embd_talker + i] =
                projected_embeddings[(t + 4) * n_embd_talker + i];
        }
    }
    // Append tts_eos_embed at the end
    for (int i = 0; i < n_embd_talker; ++i) {
        trailing_text_hidden[n_trailing_text * n_embd_talker + i] = tts_eos_embed[i];
    }

    printf("Trailing text hidden: %d positions (%d text + 1 eos)\n", n_trailing, n_trailing_text);

    // Dump tensors for comparison with HuggingFace (if --dump-tensors specified)
    if (!params.dump_tensors_path.empty()) {
        printf("\n=== Dumping tensors to %s ===\n", params.dump_tensors_path.c_str());

        // Create directory if needed
        std::string mkdir_cmd = "mkdir -p " + params.dump_tensors_path;
        system(mkdir_cmd.c_str());

        // Dump tts_bos_embed [1, 1024]
        save_tensor_hf_format(params.dump_tensors_path + "/tts_bos_embed.bin",
                              tts_bos_embed.data(), {1, (uint32_t)n_embd_talker});

        // Dump tts_eos_embed [1, 1024]
        save_tensor_hf_format(params.dump_tensors_path + "/tts_eos_embed.bin",
                              tts_eos_embed.data(), {1, (uint32_t)n_embd_talker});

        // Dump tts_pad_embed [1, 1024]
        save_tensor_hf_format(params.dump_tensors_path + "/tts_pad_embed.bin",
                              tts_pad_embed.data(), {1, (uint32_t)n_embd_talker});

        // Dump assistant_hidden (projected_embeddings) [n_text_tokens, 1024]
        save_tensor_hf_format(params.dump_tensors_path + "/assistant_hidden.bin",
                              projected_embeddings.data(),
                              {(uint32_t)n_text_tokens, (uint32_t)n_embd_talker});

        // Dump trailing_text_hidden [n_trailing, 1024]
        save_tensor_hf_format(params.dump_tensors_path + "/trailing_text_hidden.bin",
                              trailing_text_hidden.data(),
                              {(uint32_t)n_trailing, (uint32_t)n_embd_talker});

        // Dump prefill_embeds [n_prefill, 1024]
        save_tensor_hf_format(params.dump_tensors_path + "/prefill_embeds.bin",
                              prefill_embeds.data(),
                              {(uint32_t)n_prefill, (uint32_t)n_embd_talker});

        printf("=== Tensor dump complete ===\n\n");
    }

    // Run prefill through Talker
    printf("Running Talker prefill...\n");
    {
        llama_batch batch = llama_batch_init(n_prefill, n_embd_talker, 1);
        batch.n_tokens = n_prefill;
        memcpy(batch.embd, prefill_embeds.data(), n_prefill * n_embd_talker * sizeof(float));
        for (int p = 0; p < n_prefill; ++p) {
            batch.pos[p] = p;
            batch.n_seq_id[p] = 1;
            batch.seq_id[p][0] = 0;
            batch.logits[p] = (p == n_prefill - 1) ? 1 : 0;  // Only last position needs logits
        }

        int ret = llama_decode(talker_ctx, batch);
        llama_batch_free(batch);

        if (ret != 0) {
            fprintf(stderr, "Error: Talker prefill failed with code %d\n", ret);
            llama_free(talker_ctx);
            llama_model_free(talker_model);
            llama_free(thinker_ctx);
            llama_model_free(thinker_model);
            return 1;
        }
        printf("Prefill complete\n");
    }

    // Get initial hidden state from prefill
    // CRITICAL: Use pre-norm hidden state (BEFORE output_norm) for Code Predictor
    // HuggingFace uses hidden_states[0][-1] which is the last layer output BEFORE self.norm()
    std::vector<float> past_hidden(n_embd_talker, 0.0f);
    {
        // Debug: also get post-norm embeddings for comparison
        float * post_norm_emb = llama_get_embeddings(talker_ctx);
        if (post_norm_emb) {
            float post_sum_sq = 0.0f, post_min = post_norm_emb[0], post_max = post_norm_emb[0];
            for (int i = 0; i < n_embd_talker; ++i) {
                post_sum_sq += post_norm_emb[i] * post_norm_emb[i];
                if (post_norm_emb[i] < post_min) post_min = post_norm_emb[i];
                if (post_norm_emb[i] > post_max) post_max = post_norm_emb[i];
            }
            float post_l2 = sqrtf(post_sum_sq);
            fprintf(stderr, "DEBUG POST-NORM: L2=%.4f, min=%.4f, max=%.4f, first 4: [%.4f, %.4f, %.4f, %.4f]\n",
                    post_l2, post_min, post_max, post_norm_emb[0], post_norm_emb[1], post_norm_emb[2], post_norm_emb[3]);
        }

        // Extract POST-NORM hidden state (AFTER output_norm) for Code Predictor
        // mlx-vlm and HuggingFace both use post-norm: hidden_states = self.norm(hidden_states); return hidden_states
        // The Code Predictor input is: concat(past_hidden, last_id_hidden) where past_hidden is POST-NORM
        float * embeddings = llama_get_embeddings(talker_ctx);
        if (embeddings) {
            memcpy(past_hidden.data(), embeddings, n_embd_talker * sizeof(float));
        } else {
            fprintf(stderr, "Warning: Failed to get embeddings for past_hidden\n");
        }

        // Dump hidden state for debugging (post-norm, matching mlx-vlm)
        if (!params.dump_tensors_path.empty()) {
            std::string hidden_path = params.dump_tensors_path + "/hidden_post_norm.bin";
            printf("  Dumping post-norm hidden state to %s\n", hidden_path.c_str());
            std::ofstream ofs(hidden_path, std::ios::binary);
            if (ofs.is_open()) {
                uint32_t ndims = 1;
                uint32_t dim0 = n_embd_talker;
                ofs.write(reinterpret_cast<const char*>(&ndims), sizeof(uint32_t));
                ofs.write(reinterpret_cast<const char*>(&dim0), sizeof(uint32_t));
                ofs.write(reinterpret_cast<const char*>(past_hidden.data()), n_embd_talker * sizeof(float));
                ofs.close();
            }

            // Print stats
            float sum = 0, sum_sq = 0;
            for (int i = 0; i < n_embd_talker; ++i) {
                sum += past_hidden[i];
                sum_sq += past_hidden[i] * past_hidden[i];
            }
            float mean = sum / n_embd_talker;
            float std_dev = sqrtf(sum_sq / n_embd_talker - mean * mean);
            printf("  Post-norm hidden: mean=%.6f, std=%.6f, first 8: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
                   mean, std_dev,
                   past_hidden[0], past_hidden[1], past_hidden[2], past_hidden[3],
                   past_hidden[4], past_hidden[5], past_hidden[6], past_hidden[7]);
        }
    }

    // Sample first token from prefill logits
    // CRITICAL: Use llama_get_logits_ith(-1) to get the LAST token's logits
    // With embeddings=true, llama_get_logits() returns first token which is WRONG
    int last_token;
    {
        float * logits = llama_get_logits_ith(talker_ctx, -1);
        if (!logits) {
            fprintf(stderr, "Error: Failed to get logits from prefill\n");
            llama_free(talker_ctx);
            llama_model_free(talker_model);
            llama_free(thinker_ctx);
            llama_model_free(thinker_model);
            return 1;
        }

        // Dump prefill logits if requested
        if (!params.dump_tensors_path.empty()) {
            std::string logits_path = params.dump_tensors_path + "/prefill_logits.bin";
            printf("  Dumping prefill logits to %s\n", logits_path.c_str());
            std::ofstream ofs(logits_path, std::ios::binary);
            if (ofs.is_open()) {
                // Write dimension header
                uint32_t ndims = 1;
                uint32_t dim0 = n_vocab;
                ofs.write(reinterpret_cast<const char*>(&ndims), sizeof(uint32_t));
                ofs.write(reinterpret_cast<const char*>(&dim0), sizeof(uint32_t));
                ofs.write(reinterpret_cast<const char*>(logits), n_vocab * sizeof(float));
                ofs.close();
            }

            // Also print top-10 logits
            std::vector<std::pair<float, int>> top_logits;
            for (int i = 0; i < n_vocab; ++i) {
                top_logits.push_back({logits[i], i});
            }
            std::sort(top_logits.begin(), top_logits.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
            printf("  Top-10 logits before sampling:\n");
            for (int i = 0; i < 10 && i < (int)top_logits.size(); ++i) {
                printf("    [%d] token=%d, logit=%.4f\n", i, top_logits[i].second, top_logits[i].first);
            }
        }

        // Handle temperature=0 as greedy sampling
        if (params.temperature <= 0.0f) {
            // Greedy: find max logit among valid tokens
            float max_logit = -1e30f;
            last_token = 0;
            for (int i = 0; i < n_vocab; ++i) {
                // Only consider audio tokens (0-2047) and EOS (2150)
                if (i >= 2048 && i != TALKER_CODEC_EOS_ID) {
                    continue;
                }
                if (logits[i] > max_logit) {
                    max_logit = logits[i];
                    last_token = i;
                }
            }
            printf("Greedy sampling (temp=0): token=%d, logit=%.4f\n", last_token, max_logit);
        } else {
            std::mt19937 rng_prefill(42);
            std::vector<int> empty_recent;
            last_token = sample_token(logits, n_vocab, params.temperature, params.top_k,
                                      rng_prefill, empty_recent, 1.05f, 0.8f);
        }
        printf("First token from prefill: %d\n", last_token);
    }

    // Generate codec tokens with inline Code Predictor feedback
    // Reference: HuggingFace runs Code Predictor inside Talker loop, sums all 16 embeddings
    // inputs_embeds = sum([last_id_hidden] + mid_residual_hiddens + [last_residual_hidden]) + text_embed
    std::vector<int> codec_tokens;
    codec_tokens.push_back(last_token);  // Include first token from prefill

    std::vector<float> cur_embd(n_embd_talker);
    std::vector<float> last_id_hidden(n_embd_talker);
    int pos = n_prefill;  // Start after prefill positions

    // Storage for all codebook tokens (generated inline with Code Predictor)
    const int n_codebooks = 16;
    std::vector<std::vector<int>> all_codebook_tokens;
    all_codebook_tokens.reserve(params.max_codec_tokens);

    printf("Generating codec tokens with inline Code Predictor (max %d)...\n", params.max_codec_tokens);

    int ret;  // For llama_decode return codes in generation loop

    for (int step = 0; step < params.max_codec_tokens; ++step) {
        // Get codec token embedding (last_id_hidden)
        std::fill(last_id_hidden.begin(), last_id_hidden.end(), 0.0f);
        if (last_token >= 0 && last_token < vocab_size) {
            for (int i = 0; i < n_embd_talker; ++i) {
                last_id_hidden[i] = tok_embd_data[last_token * embd_dim + i];
            }
        }

        // Debug: print last_id_hidden L2 for comparison with HuggingFace
        if (params.verbose && step < 3) {
            float lid_l2 = 0.0f;
            for (int i = 0; i < n_embd_talker; ++i) {
                lid_l2 += last_id_hidden[i] * last_id_hidden[i];
            }
            lid_l2 = sqrtf(lid_l2);
            fprintf(stderr, "DEBUG Step %d: last_id_hidden L2=%.4f (token=%d), first 4: [%.4f, %.4f, %.4f, %.4f]\n",
                    step, lid_l2, last_token, last_id_hidden[0], last_id_hidden[1], last_id_hidden[2], last_id_hidden[3]);
        }

        // Sum all embeddings for Talker input
        // CRITICAL FIX: Run Code Predictor for ALL steps (including step 0)
        // HuggingFace runs Code Predictor at every step to get the 17-embedding sum
        std::fill(cur_embd.begin(), cur_embd.end(), 0.0f);

        // Run Code Predictor inline to get feedback embeddings
        // Returns 15 input embeddings + 1 last_residual_hidden = 16 total
        // Plus last_id_hidden = 17 embeddings total, matching HuggingFace
        std::vector<std::vector<float>> cp_hidden_states;
        std::vector<int> cp_codebook_tokens;

        if (run_code_predictor_inline(talker_model, past_hidden, last_id_hidden,
                                      cp_hidden_states, cp_codebook_tokens, rng,
                                      params.temperature, params.verbose && step < 3)) {
            // Sum: last_id_hidden + 16 embeddings from Code Predictor (15 input embeds + 1 last_residual)
            // Reference: codec_hiddens = [last_id_hidden] + mid_residual_hiddens + [last_residual_hidden]
            // inputs_embeds = codec_hiddens.sum(1, keepdim=True)
            // Total: 1 + 16 = 17 embeddings, matching HuggingFace
            for (int i = 0; i < n_embd_talker; ++i) {
                cur_embd[i] = last_id_hidden[i];
            }
            for (const auto & hs : cp_hidden_states) {
                for (int i = 0; i < n_embd_talker; ++i) {
                    cur_embd[i] += hs[i];
                }
            }

            // Store all codebook tokens for this frame
            std::vector<int> frame_tokens(n_codebooks);
            frame_tokens[0] = last_token;  // Codebook 0 from Talker
            for (size_t cb = 0; cb < cp_codebook_tokens.size() && cb + 1 < (size_t)n_codebooks; ++cb) {
                frame_tokens[cb + 1] = cp_codebook_tokens[cb];
            }
            all_codebook_tokens.push_back(frame_tokens);

            // Debug: print Code Predictor hidden states L2 and tokens
            if (params.verbose && step <= 2) {
                // Print L2 of each hidden state contribution
                float total_cp_l2 = 0.0f;
                for (size_t hi = 0; hi < cp_hidden_states.size(); ++hi) {
                    float hs_l2 = 0.0f;
                    for (int i = 0; i < n_embd_talker; ++i) {
                        hs_l2 += cp_hidden_states[hi][i] * cp_hidden_states[hi][i];
                    }
                    hs_l2 = sqrtf(hs_l2);
                    total_cp_l2 += hs_l2;
                    if (hi < 3 || hi == cp_hidden_states.size() - 1) {
                        fprintf(stderr, "    CP hs[%zu] L2=%.4f\n", hi, hs_l2);
                    } else if (hi == 3) {
                        fprintf(stderr, "    ... (%zu more hidden states) ...\n", cp_hidden_states.size() - 4);
                    }
                }
                fprintf(stderr, "  Total CP hidden states L2 sum=%.4f (count=%zu)\n", total_cp_l2, cp_hidden_states.size());

                printf("  CP tokens (CB1-5): [%d, %d, %d, %d, %d]\n",
                       cp_codebook_tokens.size() > 0 ? cp_codebook_tokens[0] : -1,
                       cp_codebook_tokens.size() > 1 ? cp_codebook_tokens[1] : -1,
                       cp_codebook_tokens.size() > 2 ? cp_codebook_tokens[2] : -1,
                       cp_codebook_tokens.size() > 3 ? cp_codebook_tokens[3] : -1,
                       cp_codebook_tokens.size() > 4 ? cp_codebook_tokens[4] : -1);
            }
        } else {
            // Fallback: just use last_id_hidden (should rarely happen)
            fprintf(stderr, "Warning: Code Predictor failed at step %d, using fallback\n", step);
            for (int i = 0; i < n_embd_talker; ++i) {
                cur_embd[i] = last_id_hidden[i];
            }
        }

        // ADD text embedding (residual injection)
        // Fix #31: Use trailing_text_hidden which contains [text[4:], tts_eos_embed]
        // - step < n_trailing: use trailing_text_hidden[step]
        // - step >= n_trailing: use tts_pad_embed
        float codec_norm = 0.0f, text_norm = 0.0f;
        for (int i = 0; i < n_embd_talker; ++i) {
            codec_norm += cur_embd[i] * cur_embd[i];
        }
        codec_norm = std::sqrt(codec_norm);

        if (step < n_trailing) {
            // Use trailing_text_hidden (text[4:] followed by tts_eos_embed at end)
            for (int i = 0; i < n_embd_talker; ++i) {
                float tv = trailing_text_hidden[step * n_embd_talker + i];
                text_norm += tv * tv;
                cur_embd[i] += tv;
            }
            text_norm = std::sqrt(text_norm);
            if (params.verbose && step == n_trailing - 1) {
                printf("  Step %d: Injecting tts_eos_embed (last element of trailing_text_hidden)\n", step);
            }
        } else {
            // Use tts_pad_embed for continued generation after text is exhausted
            for (int i = 0; i < n_embd_talker; ++i) {
                float tv = tts_pad_embed[i];
                text_norm += tv * tv;
                cur_embd[i] += tv;
            }
            text_norm = std::sqrt(text_norm);
        }

        if (params.verbose && step < 5) {
            printf("  Step %d: codec_norm=%.2f, text_norm=%.2f, combined[0:4]=[%.2f,%.2f,%.2f,%.2f]\n",
                   step, codec_norm, text_norm, cur_embd[0], cur_embd[1], cur_embd[2], cur_embd[3]);
        }

        // Create batch with embedding (not token)
        llama_batch batch = llama_batch_init(1, n_embd_talker, 1);
        batch.n_tokens = 1;
        memcpy(batch.embd, cur_embd.data(), n_embd_talker * sizeof(float));
        batch.pos[0] = pos++;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;

        ret = llama_decode(talker_ctx, batch);
        llama_batch_free(batch);

        if (ret != 0) {
            fprintf(stderr, "Error: Talker decode failed at step %d with code %d\n", step, ret);
            break;
        }

        // Debug: Check post-norm embeddings during decode
        if (params.verbose && step < 3) {
            float * post_norm = llama_get_embeddings(talker_ctx);
            if (post_norm) {
                float post_l2 = 0.0f;
                for (int i = 0; i < n_embd_talker; ++i) {
                    post_l2 += post_norm[i] * post_norm[i];
                }
                post_l2 = sqrtf(post_l2);
                fprintf(stderr, "DEBUG Step %d POST-NORM: L2=%.4f, first 4: [%.4f, %.4f, %.4f, %.4f]\n",
                        step, post_l2, post_norm[0], post_norm[1], post_norm[2], post_norm[3]);
            }
        }

        // Get hidden state from this step for next iteration
        // Use POST-NORM hidden state (AFTER output_norm) for Code Predictor
        // Reference: mlx-vlm line 765: past_hidden = hidden_states_list[-1][0][:, -1:]
        // where hidden_states is the output of model() which applies self.norm() before return
        float * embeddings = llama_get_embeddings(talker_ctx);
        if (embeddings) {
            memcpy(past_hidden.data(), embeddings, n_embd_talker * sizeof(float));
        }

        // Get logits and sample
        // CRITICAL: Use llama_get_logits_ith(-1) to get the LAST token's logits
        float * logits = llama_get_logits_ith(talker_ctx, -1);
        if (!logits) {
            fprintf(stderr, "Error: Failed to get logits at step %d\n", step);
            break;
        }

        // Get recent tokens for repetition penalty (last 32 tokens)
        std::vector<int> recent_tokens;
        size_t lookback = std::min(codec_tokens.size(), (size_t)32);
        for (size_t i = codec_tokens.size() - lookback; i < codec_tokens.size(); ++i) {
            recent_tokens.push_back(codec_tokens[i]);
        }

        // Fix #22: Match HuggingFace sampling parameters
        // HuggingFace Qwen3 Omni MoE: talker_top_p=1.0 (nucleus sampling disabled)
        // rep_penalty: 1.05, top_p: 1.0, temp: 0.9, top_k: 50
        int token = sample_token(logits, n_vocab, params.temperature, params.top_k, rng,
                                  recent_tokens, 1.05f, 1.0f);

        // Debug: show top logits before EOS check
        if (params.verbose && step <= 2) {
            std::vector<std::pair<float, int>> sorted_logits;
            for (int i = 0; i < n_vocab && i < 2048; ++i) {
                sorted_logits.push_back({logits[i], i});
            }
            // Also include EOS token
            sorted_logits.push_back({logits[TALKER_CODEC_EOS_ID], TALKER_CODEC_EOS_ID});
            std::sort(sorted_logits.begin(), sorted_logits.end(),
                [](const auto & a, const auto & b) { return a.first > b.first; });
            printf("  Step %d sampled token %d, Top-5: [%d:%.2f, %d:%.2f, %d:%.2f, %d:%.2f, %d:%.2f]\n",
                   step, token,
                   sorted_logits[0].second, sorted_logits[0].first,
                   sorted_logits[1].second, sorted_logits[1].first,
                   sorted_logits[2].second, sorted_logits[2].first,
                   sorted_logits[3].second, sorted_logits[3].first,
                   sorted_logits[4].second, sorted_logits[4].first);
        }

        // Fix #26: Check for EOS using correct token ID (2150)
        if (is_talker_eos(token) || token >= n_vocab) {
            printf("EOS token %d received at step %d\n", token, step);
            break;
        }

        codec_tokens.push_back(token);
        last_token = token;

        if (params.verbose && step < 10) {
            // Debug: show EOS logit rank and value
            float eos_logit = logits[TALKER_CODEC_EOS_ID];
            float max_logit = logits[0];
            int eos_rank = 0;
            for (int i = 0; i < n_vocab && i < 2048; ++i) {
                if (logits[i] > max_logit) max_logit = logits[i];
                if (logits[i] > eos_logit) eos_rank++;
            }
            printf("  Step %d: token %d, EOS_logit=%.2f (rank %d, max=%.2f)\n",
                   step, token, eos_logit, eos_rank, max_logit);

            // Extra debug at steps 0-2: dump key values for HuggingFace comparison
            if (step <= 2) {
                printf("  DEBUG Step %d details for HuggingFace comparison:\n", step);
                printf("    past_hidden[0:8]: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
                       past_hidden[0], past_hidden[1], past_hidden[2], past_hidden[3],
                       past_hidden[4], past_hidden[5], past_hidden[6], past_hidden[7]);
                printf("    last_id_hidden[0:8]: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
                       last_id_hidden[0], last_id_hidden[1], last_id_hidden[2], last_id_hidden[3],
                       last_id_hidden[4], last_id_hidden[5], last_id_hidden[6], last_id_hidden[7]);

                // Find top-5 logits
                std::vector<std::pair<float, int>> sorted_logits;
                for (int i = 0; i < n_vocab && i < 2048; ++i) {
                    sorted_logits.push_back({logits[i], i});
                }
                std::sort(sorted_logits.begin(), sorted_logits.end(),
                    [](const auto & a, const auto & b) { return a.first > b.first; });
                printf("    Top-5 logits: [%d:%.2f, %d:%.2f, %d:%.2f, %d:%.2f, %d:%.2f]\n",
                       sorted_logits[0].second, sorted_logits[0].first,
                       sorted_logits[1].second, sorted_logits[1].first,
                       sorted_logits[2].second, sorted_logits[2].first,
                       sorted_logits[3].second, sorted_logits[3].first,
                       sorted_logits[4].second, sorted_logits[4].first);
            }
        }

        // Progress indicator
        if ((step + 1) % 50 == 0) {
            printf("  Generated %d tokens...\n", step + 1);
        }
    }

    printf("Generated %zu codec tokens\n", codec_tokens.size());

    if (params.verbose) {
        printf("First 20 codec tokens: [");
        for (size_t i = 0; i < std::min(codec_tokens.size(), (size_t)20); ++i) {
            printf("%d%s", codec_tokens[i], i < std::min(codec_tokens.size(), (size_t)20) - 1 ? ", " : "");
        }
        printf("]\n");
    }

    // Save codec tokens
    std::string codec_path = params.output_path + ".codec.bin";
    {
        std::ofstream f(codec_path, std::ios::binary);
        uint32_t n = codec_tokens.size();
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));
        f.write(reinterpret_cast<const char*>(codec_tokens.data()), n * sizeof(int));
        printf("Saved %zu codec tokens to: %s\n", codec_tokens.size(), codec_path.c_str());
    }

    // ========================================
    // Step 7: Code Predictor (verify inline-generated codebooks)
    // ========================================
    printf("\nVerifying Code Predictor tokens (generated inline during Talker loop)...\n");
    fflush(stdout);

    // all_codebook_tokens was populated during the Talker generation loop
    // It contains frames for all steps starting from step 0 (Code Predictor runs at every step)
    // Note: The last sampled token doesn't have a frame (it's for the NEXT step which never runs)

    // Verify we have the right number of frames
    if (all_codebook_tokens.size() != codec_tokens.size()) {
        fprintf(stderr, "Warning: Codebook frames (%zu) != codec tokens (%zu), adjusting...\n",
                all_codebook_tokens.size(), codec_tokens.size());
        all_codebook_tokens.resize(codec_tokens.size());
        for (auto & frame : all_codebook_tokens) {
            if (frame.size() != (size_t)n_codebooks) {
                frame.resize(n_codebooks, 0);
            }
        }
    }

    printf("Verified %zu frames × %d codebooks\n", all_codebook_tokens.size(), n_codebooks);
    if (params.verbose && all_codebook_tokens.size() > 0) {
        printf("  Frame 0: cb0=%d", all_codebook_tokens[0][0]);
        for (int cb = 1; cb < n_codebooks && cb < 5; ++cb) {
            printf(", cb%d=%d", cb, all_codebook_tokens[0][cb]);
        }
        printf("...\n");

        // Detailed codebook analysis
        printf("\n=== Codebook Token Analysis ===\n");
        const size_t n_frames = all_codebook_tokens.size();
        for (int cb = 0; cb < n_codebooks; ++cb) {
            std::set<int> unique_tokens;
            int min_tok = INT_MAX, max_tok = INT_MIN;
            double sum = 0.0;
            for (size_t f = 0; f < n_frames; ++f) {
                int tok = all_codebook_tokens[f][cb];
                unique_tokens.insert(tok);
                if (tok < min_tok) min_tok = tok;
                if (tok > max_tok) max_tok = tok;
                sum += tok;
            }
            double mean = sum / n_frames;
            double variance = 0.0;
            for (size_t f = 0; f < n_frames; ++f) {
                double diff = all_codebook_tokens[f][cb] - mean;
                variance += diff * diff;
            }
            double std_dev = sqrt(variance / n_frames);
            printf("  cb%d: unique=%zu, range=[%d, %d], std=%.1f\n",
                   cb, unique_tokens.size(), min_tok, max_tok, std_dev);
        }
        printf("\n");
    }
    fflush(stdout);

    // Dump tokens if requested (for HuggingFace comparison)
    if (!params.dump_tokens_path.empty() && !all_codebook_tokens.empty()) {
        printf("Dumping codec tokens to %s...\n", params.dump_tokens_path.c_str());
        std::ofstream ofs(params.dump_tokens_path);
        if (ofs.is_open()) {
            // Format: 16 lines (one per codebook), space-separated token IDs
            const size_t n_frames = all_codebook_tokens.size();
            for (int cb = 0; cb < n_codebooks; ++cb) {
                for (size_t f = 0; f < n_frames; ++f) {
                    if (f > 0) ofs << " ";
                    ofs << all_codebook_tokens[f][cb];
                }
                ofs << "\n";
            }
            ofs.close();
            printf("  Wrote %zu frames × %d codebooks\n", n_frames, n_codebooks);
        } else {
            fprintf(stderr, "Warning: Could not open %s for writing\n", params.dump_tokens_path.c_str());
        }
    }

    // ========================================
    // Step 8: Code2Wav (convert to audio)
    // ========================================
    // Get model pointer for tensor access
    const struct llama_model * m = talker_model;

    // If --load-tokens is provided, load tokens from file (for debugging Code2Wav)
    if (!params.load_tokens_path.empty()) {
        auto loaded_tokens = load_codebook_tokens(params.load_tokens_path);
        if (!loaded_tokens.empty()) {
            all_codebook_tokens = loaded_tokens;
            printf("Using %zu frames from --load-tokens file (skipping Talker generation)\n",
                   all_codebook_tokens.size());
        }
    }

    if (params.skip_code2wav) {
        printf("\nSkipping Code2Wav (--skip-code2wav flag set)\n");
    } else {
        printf("\nRunning Code2Wav vocoder...\n");
        fflush(stdout);

        // Access Code2Wav tensors from model
        if (!m->c2w_code_embd || m->c2w_pre_layers.empty()) {
            fprintf(stderr, "Warning: Code2Wav tensors not found, generating placeholder audio\n");

            // Generate a simple sine wave as placeholder
            const int samples_per_frame = 480;  // 24kHz / 50 fps
            int n_samples = codec_tokens.size() * samples_per_frame;
            std::vector<float> audio_samples(n_samples);

            printf("Generating placeholder audio (%d samples)...\n", n_samples);

            // Generate simple tones based on codec values
            for (size_t frame = 0; frame < codec_tokens.size(); ++frame) {
                float freq = 200.0f + (codec_tokens[frame] % 1000) * 0.5f;  // 200-700 Hz
                for (int s = 0; s < samples_per_frame; ++s) {
                    int idx = frame * samples_per_frame + s;
                    float t = static_cast<float>(idx) / params.sample_rate;
                    audio_samples[idx] = 0.3f * sinf(2.0f * 3.14159f * freq * t);
                }
            }

            // Apply simple fade in/out
            int fade_samples = std::min(1000, n_samples / 4);
            for (int i = 0; i < fade_samples; ++i) {
                float fade = static_cast<float>(i) / fade_samples;
                audio_samples[i] *= fade;
                audio_samples[n_samples - 1 - i] *= fade;
            }

            // Write WAV file
            if (write_wav(params.output_path, audio_samples.data(), n_samples, params.sample_rate)) {
                printf("Wrote placeholder audio to: %s\n", params.output_path.c_str());
                printf("Duration: %.2f seconds\n", static_cast<float>(n_samples) / params.sample_rate);
            }
        } else {
            printf("Code2Wav has %zu pre-transformer layers\n", m->c2w_pre_layers.size());

            // Run Code2Wav vocoder using ggml graph execution
            std::vector<float> audio_samples = run_code2wav_ggml(m, all_codebook_tokens, params.verbose, params.dump_tensors_path, params.c2w_cpu_only);

            int n_samples = audio_samples.size();

            // Apply fade in/out
            int fade_samples = std::min(500, n_samples / 8);
            for (int i = 0; i < fade_samples; ++i) {
                float fade = static_cast<float>(i) / fade_samples;
                audio_samples[i] *= fade;
                audio_samples[n_samples - 1 - i] *= fade;
            }

            if (write_wav(params.output_path, audio_samples.data(), n_samples, params.sample_rate)) {
                printf("Wrote audio to: %s\n", params.output_path.c_str());
                printf("Duration: %.2f seconds\n", static_cast<float>(n_samples) / params.sample_rate);
            } else {
                fprintf(stderr, "Error: write_wav failed\n");
            }
        }
    }

    // ========================================
    // Pipeline Status
    // ========================================
    printf("\n=== Pipeline Status ===\n");
    printf("[x] Thinker: Layer-%d extraction (%d tokens)\n", params.n_layer_output, n_tokens);
    printf("[x] Text Projection: 2048 -> 1024\n");
    printf("[x] Talker: Generated %zu codec tokens\n", codec_tokens.size());
    printf("[%c] Code Predictor: %s\n",
           m->talker_cp_layers.empty() ? '~' : 'x',
           m->talker_cp_layers.empty() ? "Simplified expansion (tensors not found)" : "Expanded to 16 codebooks");
    printf("[%c] Code2Wav: %s\n",
           params.skip_code2wav ? '-' : (m->c2w_pre_layers.empty() ? '~' : 'x'),
           params.skip_code2wav ? "Skipped" : "HiFi-GAN vocoder synthesis");
    printf("[x] WAV Output: %s\n", params.output_path.c_str());

    // Cleanup
    llama_free(talker_ctx);
    llama_model_free(talker_model);
    llama_free(thinker_ctx);
    llama_model_free(thinker_model);
    llama_backend_free();

    printf("\nPipeline completed!\n");
    return 0;
}
