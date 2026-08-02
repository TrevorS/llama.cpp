#include "models.h"

#include "llama-impl.h"          // llama_mul_mat_hadamard (KV-cache rotation on the inject path)
#include "llama-kv-cache.h"      // complete llama_kv_cache_context for KV injection (cpy_k/cpy_v)
#include "llama-kv-cache-iswa.h" // complete llama_kv_cache_iswa_context for the sliding-window sub-cache

#include "llama-ext.h" // staging API: llama_dspark_markov_bias (host-side Markov bias for the draft driver)

#include "ggml-backend.h" // ggml_backend_tensor_get (Markov weight readback)

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

// ============================================================================
// DSpark draft head for DeepSeek-V4-Flash.
//
// This mirrors the DFlash speculative packaging (dflash.cpp): a feature-fusion
// encoder collapses the concatenated target-layer hidden states through `fc`,
// and a dual-mode decoder either injects those features into the draft KV cache
// (embd batch) or diffuses a noise block into draft tokens (token batch).
//
// The backbone differs from DFlash: it is a small (3-layer) DeepSeek-V4 stack —
// MLA-lite attention (q-LoRA + single KV latent + o-LoRA + attention sinks),
// sqrtsoftplus MoE with a shared expert, and hyper-connection (hc == 4)
// residual mixing. A low-rank Markov head biases block logits host-side.
//
// INTEGRATION NOTES (see the report to the lead):
//   * The target model (deepseek4.cpp) does not currently export per-layer input
//     embeddings, only `t_h_nextn`. The DFlash/DSpark driver's process() calls
//     llama_get_embeddings_layer_inp(ctx_tgt, [40,41,42]); the target side must
//     mean-pool the hc-stream hidden state to width n_embd and export it there.
//   * MLA here runs over the STANDARD non-causal KV cache with a single KV head
//     (n_head_kv must be 1). The DeepSeek-V4 `k_rot` NoPE hadamard is NOT applied
//     (it belongs to the bespoke dsv4 cache); this is the primary runtime-parity
//     risk to verify once the checkpoint exists.
// ============================================================================

// --- hparams ----------------------------------------------------------------

void llama_model_dspark::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func);
    if (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
        throw std::runtime_error("DSpark loader expects sqrtsoftplus MoE scoring");
    }

    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP, hparams.swiglu_clamp_exp, hparams.n_layer());
    if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP, hparams.swiglu_clamp_shexp, hparams.n_layer(), 0)) {
        hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
    }

    ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,      hparams.dsv4_o_group_count);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,        hparams.dsv4_o_lora_rank);
    // dspark draft blocks are raw-attention only; the converter intentionally
    // skips compress/indexer keys, and this value is unused in the dspark graph
    ml.get_key(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE, hparams.dsv4_compress_rope_base, false);
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);

    // target layers -> feature-fusion input width (concat of collapsed hidden states)
    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("DSpark model requires 'target_layers' in GGUF metadata");
    }
    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * hparams.n_embd;

    // Sliding-window attention. The reference DSparkAttention keeps a ring KV
    // cache of window_size entries and builds its index list as
    //     [ arange(min(win, start_pos+1)) , win + arange(block_size) ]
    // (inference/model.py get_dspark_topk_idxs) -- i.e. the last `win` history
    // positions plus the whole current draft block. That is a standard causal
    // sliding window of width `win`; DeepSeek-V4-Flash-0731 declares win = 128.
    // Without this the draft attends over the entire KV cache and conditions on
    // context it was never trained against, which changes every drafted logit.
    //
    // LLAMA_SWA_TYPE_STANDARD masks a (key p0, query p1) pair when
    // p1 - p0 >= n_swa (llama-hparams.h:400), which reproduces the reference
    // index set: intra-block pairs are at most block_size-1 == 4 apart so they
    // are never masked (and causal_attn is off for this draft, so the block
    // stays bidirectional), while history is clipped to the last 128 positions.
    // The reference anchors ONE window at the block start and shares it across the
    // whole block: sglang computes prefix_lens from the block's first token and
    // broadcasts the same swa_page_indices to all block_size queries, and
    // SpecForge's element-level mask spec derives anchor_pos from q_idx /
    // block_size only -- never from the query's offset within the block.
    //
    // LLAMA_SWA_TYPE_STANDARD slides per query (masked when p1 - p0 >= n_swa), so
    // with the anchor at position P a query at block offset j keeps history
    // [P+j-127, P-1] = 127-j keys where the reference keeps a constant 128 -- a
    // shortfall of j+1 keys at EVERY offset, which no choice of n_swa corrects
    // (129 fixes offset 0 but is still short at the last; a value large enough for
    // the last over-includes at offset 0). LLAMA_SWA_TYPE_BLOCK_ANCHORED anchors
    // the test on the sequence's first position in the batch instead, which
    // reproduces the reference set exactly.
    //
    // Getting the window wrong entirely is expensive: acceptance 41.5% -> 14.6% at
    // n=5 with LLAMA_DSPARK_NO_SWA=1.
    //
    // LLAMA_DSPARK_NO_SWA=1 forces the pre-window behaviour (draft attends the
    // whole cache). Kept as an A/B control: any acceptance number for this draft
    // is only interpretable against its paired no-window run.
    const bool no_swa = getenv("LLAMA_DSPARK_NO_SWA") && atoi(getenv("LLAMA_DSPARK_NO_SWA"));

    // LLAMA_DSPARK_SWA_STANDARD=1 falls back to the per-query sliding window as a
    // control for the block-anchored mask.
    const bool swa_std = getenv("LLAMA_DSPARK_SWA_STANDARD") && atoi(getenv("LLAMA_DSPARK_SWA_STANDARD"));

    if (!no_swa && ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false) && hparams.n_swa > 0) {
        hparams.swa_type = swa_std ? LLAMA_SWA_TYPE_STANDARD : LLAMA_SWA_TYPE_BLOCK_ANCHORED;
        // Every DSpark block is the same raw-attention block over the same
        // window -- there is no interleaved dense/SWA pattern like DFlash has,
        // so the pattern key is absent from the checkpoint. n_pattern == 0 is
        // the "every layer is SWA" sentinel (llama-hparams.cpp:8); note that
        // passing 1 would mark every layer DENSE, since `il % 1 < 0` is never
        // true, and llama-model.cpp:2280 then asserts on is_swa_any().
        // All-SWA leaves llama_kv_cache_iswa's base sub-cache with zero layers,
        // which is well-defined: it allocates cells but no K/V tensors.
        hparams.set_swa_pattern(0);
        ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer(), false);
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    }

    // Markov head rank (used to size markov_w1/w2 in load_arch_tensors).
    markov_rank = 256;
    ml.get_key(LLM_KV_DSPARK_MARKOV_RANK, markov_rank, false);

    type = LLM_TYPE_UNKNOWN;
}

// --- tensors ----------------------------------------------------------------

void llama_model_dspark::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t o_groups    = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult     = hparams.dsv4_hc_mult;
    const int64_t hc_dim      = hc_mult * n_embd;
    const int64_t hc_mix_dim  = (2 + hc_mult) * hc_mult;
    const int64_t n_embd_inp  = hparams.n_embd_inp_enc();

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    // feature-fusion encoder (shared naming with DFlash)
    fc              = create_tensor(tn(LLM_TENSOR_FC,               "weight"), {n_embd_inp, n_embd}, 0);
    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM,  "weight"), {n_embd}, 0);

    // final hyper-connection collapse head
    hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN,    "weight"), {hc_dim, hc_mult}, 0);
    hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE,  "weight"), {hc_mult}, 0);
    hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, 0);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, 0);
        layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, 0);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, 0);
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank * o_groups}, 0);
        layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, 0);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,   "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",  i), {n_expert}, 0);
        layer.ffn_norm        = create_tensor(tn(LLM_TENSOR_FFN_NORM,       "weight", i), {n_embd}, 0);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
    }

    // Markov head + confidence head. Names are suffix-less to match the converter
    // contract ("dspark.markov_w1", "dspark.markov_w2", "dspark.confidence_proj").
    markov_w1       = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W1), {(int64_t) markov_rank, n_vocab}, 0);
    markov_w2       = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2), {(int64_t) markov_rank, n_vocab}, 0);
    // confidence head input is hidden ⊕ markov features (4096 + 256 = 4352 in
    // the official checkpoint); unused in v1 but must still be consumed
    confidence_proj = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ), {n_embd + (int64_t) markov_rank, 1}, llama_model_loader::TENSOR_NOT_REQUIRED);
}

std::unique_ptr<llm_graph_context> llama_model_dspark::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}

// --- hyper-connection helpers (scalar port of deepseek4.cpp, hc == 4) --------

static ggml_tensor * dspark_hc_affine(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale, ggml_tensor * base) {
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, base);
    return x;
}

static ggml_tensor * dspark_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, ggml_row_size(t->type, i0));
}

static ggml_tensor * dspark_view_2d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], ggml_row_size(t->type, i0));
}

template <bool is_enc>
ggml_tensor * llama_model_dspark::graph<is_enc>::build_hc_weighted_sum(ggml_tensor * x, ggml_tensor * weights) const {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    ggml_tensor * acc = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, weights, 1, nt, weights->nb[1], ih*weights->nb[0]);

        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        acc = acc ? ggml_add(ctx0, acc, cur) : cur;
    }
    return acc;
}

template <bool is_enc>
ggml_tensor * llama_model_dspark::graph<is_enc>::build_hc_sinkhorn(ggml_tensor * comb) const {
    comb = ggml_soft_max(ctx0, comb);

    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx0, eps, hparams.dsv4_hc_eps);
    comb = ggml_add(ctx0, comb, eps);

    auto norm_cols = [&]() {
        ggml_tensor * comb_src_dst = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * col_sum = ggml_sum_rows(ctx0, comb_src_dst);
        col_sum = ggml_add(ctx0, col_sum, eps);
        col_sum = ggml_permute(ctx0, col_sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, col_sum);
    };
    auto norm_rows = [&]() {
        ggml_tensor * row_sum = ggml_sum_rows(ctx0, comb);
        row_sum = ggml_add(ctx0, row_sum, eps);
        comb = ggml_div(ctx0, comb, row_sum);
    };

    norm_cols();
    for (uint32_t i = 1; i < hparams.dsv4_hc_sinkhorn_iters; ++i) {
        norm_rows();
        norm_cols();
    }
    return comb;
}

template <bool is_enc>
ggml_tensor * llama_model_dspark::graph<is_enc>::build_hc_pre(
        ggml_tensor * x, ggml_tensor * hc_fn, ggml_tensor * hc_scale, ggml_tensor * hc_base,
        ggml_tensor ** post, ggml_tensor ** comb, int il) const {
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc*n_embd;
    const int64_t hc_mix_dim = (2 + hc)*hc;
    const int64_t nt         = x->ne[2];

    GGML_ASSERT(hc == 4);
    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_mixes", il);

    ggml_tensor * scale_pre  = dspark_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = dspark_view_1d(ctx0, hc_scale, 1, 1);
    ggml_tensor * scale_comb = dspark_view_1d(ctx0, hc_scale, 1, 2);

    ggml_tensor * base_pre  = dspark_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post = dspark_view_1d(ctx0, hc_base, hc, hc);
    ggml_tensor * base_comb = dspark_view_1d(ctx0, hc_base, hc*hc, 2*hc);

    ggml_tensor * pre = dspark_view_2d(ctx0, mixes, hc, nt, 0);
    pre = dspark_hc_affine(ctx0, pre, scale_pre, base_pre);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_pre", il);

    *post = dspark_view_2d(ctx0, mixes, hc, nt, hc);
    *post = dspark_hc_affine(ctx0, *post, scale_post, base_post);
    *post = ggml_sigmoid(ctx0, *post);
    *post = ggml_scale(ctx0, *post, 2.0f);
    cb(*post, "hc_post", il);

    *comb = dspark_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
    *comb = dspark_hc_affine(ctx0, *comb, scale_comb, base_comb);
    *comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
    *comb = build_hc_sinkhorn(*comb);
    cb(*comb, "hc_comb", il);

    return build_hc_weighted_sum(x, pre);
}

template <bool is_enc>
ggml_tensor * llama_model_dspark::graph<is_enc>::build_hc_post(
        ggml_tensor * x, ggml_tensor * residual, ggml_tensor * post, ggml_tensor * comb, int il) const {
    GGML_UNUSED(il);
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];

    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, x, post_dst);

        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], src*residual->nb[1]);
            ggml_tensor * comb_src_dst = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2], dst*comb->nb[0] + src*comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_src_dst));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }
    return out;
}

template <bool is_enc>
ggml_tensor * llama_model_dspark::graph<is_enc>::build_hc_head(
        ggml_tensor * x, ggml_tensor * hc_fn, ggml_tensor * hc_scale, ggml_tensor * hc_base) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_head_mixes", -1);

    ggml_tensor * pre = dspark_hc_affine(ctx0, mixes, hc_scale, hc_base);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_head_pre", -1);

    return build_hc_weighted_sum(x, pre);
}

// --- MLA-lite attention over the non-causal draft KV cache -------------------

template <bool is_enc>
ggml_tensor * llama_model_dspark::graph<is_enc>::build_attention(
        const llama_model & model,
        llm_graph_input_attn_kv      * inp_attn,
        llm_graph_input_attn_kv_iswa * inp_attn_iswa,
        ggml_tensor * cur, ggml_tensor * inp_pos, int il) const {
    GGML_ASSERT((inp_attn != nullptr) != (inp_attn_iswa != nullptr));

    const auto & layer = model.layers[il];

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t n_heads_group    = n_head / n_groups;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = n_heads_group*n_embd_head;
    const int64_t nt               = cur->ne[1];

    // dspark draft layers are uncompressed -> plain rope (no yarn), matching
    // deepseek4's ratio==0 path.
    const float   attn_factor_l = 1.0f;

    // query: q-LoRA + per-head norm + split rope
    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr_norm", il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
    q = ggml_rms_norm(ctx0, q, norm_rms_eps);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_nope, n_head, nt,
            ggml_row_size(q->type, n_embd_head),
            ggml_row_size(q->type, n_embd_head)*n_head, 0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_rope, n_head, nt,
            ggml_row_size(q->type, n_embd_head),
            ggml_row_size(q->type, n_embd_head)*n_head,
            ggml_row_size(q->type, n_embd_head_nope));
    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
            freq_base, 1.0f, 0.0f, attn_factor_l, 0.0f, 0.0f);
    q = ggml_concat(ctx0, q_nope, q_pe, 0);
    cb(q, "q", il);

    // single KV latent (MQA), split rope
    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);

    ggml_tensor * kv_nope = ggml_view_3d(ctx0, kv, n_embd_head_nope, 1, nt,
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head), 0);
    ggml_tensor * kv_pe = ggml_view_3d(ctx0, kv, n_embd_head_rope, 1, nt,
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head_nope));
    kv_pe = ggml_rope_ext(ctx0, kv_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
            freq_base, 1.0f, 0.0f, attn_factor_l, 0.0f, 0.0f);
    kv = ggml_concat(ctx0, kv_nope, kv_pe, 0);
    cb(kv, "kv", il);

    // attention over the cache (K == V == kv latent, non-causal); no output proj here
    ggml_tensor * out = inp_attn_iswa
        ? build_attn(inp_attn_iswa, nullptr, nullptr, nullptr,
                q, kv, kv, nullptr, layer.attn_sinks, nullptr,
                1.0f/sqrtf(float(n_embd_head)), il)
        : build_attn(inp_attn,      nullptr, nullptr, nullptr,
                q, kv, kv, nullptr, layer.attn_sinks, nullptr,
                1.0f/sqrtf(float(n_embd_head)), il);

    // de-rope the pe slice of the attention output (MLA absorption), matching deepseek4
    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    ggml_tensor * out_nope = ggml_view_3d(ctx0, out, n_embd_head_nope, n_head, nt,
            ggml_row_size(out->type, n_embd_head),
            ggml_row_size(out->type, n_embd_head)*n_head, 0);
    ggml_tensor * out_pe = ggml_view_3d(ctx0, out, n_embd_head_rope, n_head, nt,
            ggml_row_size(out->type, n_embd_head),
            ggml_row_size(out->type, n_embd_head)*n_head,
            ggml_row_size(out->type, n_embd_head_nope));
    out_pe = ggml_rope_ext_back(ctx0, out_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
            freq_base, 1.0f, 0.0f, attn_factor_l, 0.0f, 0.0f);
    out = ggml_concat(ctx0, out_nope, out_pe, 0);
    cb(out, "attn_derope", il);

    // grouped o-LoRA output projection
    out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, nt);
    out = ggml_permute(ctx0, out, 0, 2, 1, 3);
    ggml_tensor * oa = ggml_mul_mat(ctx0,
            ggml_reshape_3d(ctx0, layer.wo_a, layer.wo_a->ne[0], o_lora_rank, n_groups), out);
    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, nt);

    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "attn_out", il);
    return out;
}

// --- encoder graph: fuse target features -> h_nextn --------------------------

template <>
ggml_tensor * llama_model_dspark::graph<true>::build_inp_embd_enc() const {
    auto inp_target = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());

    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp_enc(), n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));
    return cur;
}

template <>
llama_model_dspark::graph<true>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd_enc();

    cur = build_lora_mm(model.fc, cur);
    // fc.weight ships as F16 and reduces 12288 terms. The target features here
    // are raw hidden states, and DeepSeek-V4's massive-activation tokens (BOS /
    // chat-template prefix) reach ~5e4, so the F16 accumulator saturates at
    // 65504 and returns inf. The following RMS norm then maps inf -> NaN, which
    // is injected into the draft KV cache and makes every subsequent draft logit
    // NaN for the rest of the sequence (silent: a NaN row still argmaxes, always
    // to the same low token id, so it reads as 0% acceptance rather than a
    // crash). The reference runs this projection in bf16, which has fp32
    // exponent range; forcing F32 accumulation is the equivalent here.
    // LLAMA_DSPARK_NO_FC_F32=1 drops the override, to check whether a checkpoint's
    // own fc dtype is safe on a stock runtime. A BF16 fc is (fp32 exponent range,
    // and ggml routes it through an F32 accumulator); an F16 one is not.
    if (!(getenv("LLAMA_DSPARK_NO_FC_F32") && atoi(getenv("LLAMA_DSPARK_NO_FC_F32")))) {
        ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
    }
    cb(cur, "fc_out", -1);

    cur = build_norm(cur, model.output_norm_enc, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "enc_norm_out", -1);

    ggml_set_output(cur);
    res->t_h_nextn = cur;
    // also publish as t_embd so cparams.embeddings (confidence-head runs) has a
    // result tensor on the encode path (build_pooling asserts otherwise)
    res->t_embd    = cur;

    ggml_build_forward_expand(gf, cur);
}

// --- decoder graph: dual-mode (KV inject / noise-block diffusion) -------------

template <>
ggml_tensor * llama_model_dspark::graph<false>::build_inp_embd_enc() const {
    GGML_ABORT("dspark decoder graph has no encoder input");
}

template <>
llama_model_dspark::graph<false>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t hc               = hparams.dsv4_hc_mult;

    ggml_tensor * inp_pos  = build_inp_pos();

    // DSpark's trained window (128) makes the draft cache interleaved-SWA; the
    // plain path stays for checkpoints without `attention.sliding_window`.
    const bool use_iswa = hparams.swa_type != LLAMA_SWA_TYPE_NONE;

    llm_graph_input_attn_kv      * inp_attn      = nullptr;
    llm_graph_input_attn_kv_iswa * inp_attn_iswa = nullptr;
    if (use_iswa) {
        inp_attn_iswa = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    // -------- mode 1: inject fused target features into the draft KV cache ----
    if (ubatch.embd) {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->embd);

        ggml_tensor * inp_g = inp->embd;
        cb(inp_g, "inp_g_embeddings", -1);
        res->add_input(std::move(inp));

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];

            // single KV latent from the injected features
            ggml_tensor * kv = build_lora_mm(layer.wkv, inp_g);
            kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
            kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_tokens);

            ggml_tensor * kv_nope = ggml_view_3d(ctx0, kv, n_embd_head_nope, 1, n_tokens,
                    ggml_row_size(kv->type, n_embd_head),
                    ggml_row_size(kv->type, n_embd_head), 0);
            ggml_tensor * kv_pe = ggml_view_3d(ctx0, kv, n_embd_head_rope, 1, n_tokens,
                    ggml_row_size(kv->type, n_embd_head),
                    ggml_row_size(kv->type, n_embd_head),
                    ggml_row_size(kv->type, n_embd_head_nope));
            kv_pe = ggml_rope_ext(ctx0, kv_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
                    freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            kv = ggml_concat(ctx0, kv_nope, kv_pe, 0);
            cb(kv, "kv_injected", il);

            // This path writes straight into the cache and so bypasses build_attn(),
            // which is where the cache's Hadamard rotation would normally be applied
            // (llama-graph.cpp:2665 / :2903). Apply it here or a quantized draft KV
            // cache (attn_rot_k, head dim 512 % 64 == 0) stores unrotated K/V that
            // the attention pass then reads as rotated.
            const llama_kv_cache_context * kvc    = nullptr;
            ggml_tensor                  * k_idxs = nullptr;
            ggml_tensor                  * v_idxs = nullptr;
            ggml_tensor                  * k_rot  = nullptr;
            ggml_tensor                  * v_rot  = nullptr;

            if (use_iswa) {
                // every dspark layer is SWA (set_swa_pattern(0)), but route by
                // is_swa(il) anyway so a future mixed pattern stays correct
                const bool is_swa = hparams.is_swa(il);

                kvc    = is_swa ? inp_attn_iswa->mctx->get_swa()      : inp_attn_iswa->mctx->get_base();
                k_idxs = is_swa ? inp_attn_iswa->get_k_idxs_swa()     : inp_attn_iswa->get_k_idxs();
                v_idxs = is_swa ? inp_attn_iswa->get_v_idxs_swa()     : inp_attn_iswa->get_v_idxs();
                k_rot  = is_swa ? inp_attn_iswa->self_k_rot_swa       : inp_attn_iswa->self_k_rot;
                v_rot  = is_swa ? inp_attn_iswa->self_v_rot_swa       : inp_attn_iswa->self_v_rot;
            } else {
                kvc    = inp_attn->mctx;
                k_idxs = inp_attn->get_k_idxs();
                v_idxs = inp_attn->get_v_idxs();
                k_rot  = inp_attn->self_k_rot;
                v_rot  = inp_attn->self_v_rot;
            }

            // K and V are the same latent here (MQA, n_head_kv == 1), but the two
            // rotations are independent, so keep separate rotated copies.
            ggml_tensor * k_inj = k_rot ? llama_mul_mat_hadamard(ctx0, kv, k_rot) : kv;
            ggml_tensor * v_inj = v_rot ? llama_mul_mat_hadamard(ctx0, kv, v_rot) : kv;

            ggml_build_forward_expand(gf, kvc->cpy_k(ctx0, k_inj, k_idxs, il));
            ggml_build_forward_expand(gf, kvc->cpy_v(ctx0, v_inj, v_idxs, il));
        }

        res->t_embd = inp_g;
        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    // -------- mode 2: noise-block diffusion -> draft tokens -------------------
    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->tok_embd != nullptr && "DSpark decoder requires the target token embeddings");
        tok_embd = model_other->tok_embd;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    ggml_tensor * inp_tokens = inp->tokens;

    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    cb(inpL, "inp_noise_embd", -1);
    res->add_input(std::move(inp));

    // broadcast into hc streams
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        ggml_tensor * cur = build_hc_pre(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, &post, &comb, il);
        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention(model, inp_attn, inp_attn_iswa, cur, inp_pos, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_hc_pre(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, &post, &comb, il);
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, hparams.n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp,   nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "l_out", il);
    }

    // collapse hc streams and project to vocab
    ggml_tensor * cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "hc_head", -1);

    // The confidence head consumes the PRE-norm hc_head output. Reference
    // (inference/model.py DSparkBlock::forward_head):
    //     x       = self.hc_head(x, ...)
    //     logits  = self.head(self.norm(x), ...)      <- norm only for logits
    //     conf    = self.confidence_head(x, markov_embed)   <- pre-norm x
    // RMS-norm rescales each row to unit RMS times a learned gain, so feeding
    // the post-norm tensor to a projection trained on unnormalized activations
    // yields a wrong confidence logit. Keep a handle to x before normalizing.
    ggml_tensor * h_prenorm = cur;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    // Publish the pre-norm rows so the driver can run the confidence head. The
    // reference applies confidence_head(x, markov_embed(prev)) where prev is the
    // token chained from the previous block position, so the head cannot be
    // finished inside this graph unless the whole sampling chain runs on-device
    // -- the ids do not exist yet here. Exporting x lets the driver call
    // llama_dspark_confidence_logit() with the prev id it just sampled.
    //
    // n_embd_out() is the hc-expanded width, so these rows are [n_embd, n_tok]
    // and the driver's `h + idx*n_embd_dec` indexing lands on row idx.
    // Without this the decoder left t_h_nextn unset and llama_get_embeddings_nextn()
    // still returned the previous ENCODE's fused features -- p_min then truncated
    // on a stale hidden-state component instead of a confidence logit.
    ggml_set_output(h_prenorm);
    res->t_h_nextn = h_prenorm;
    ggml_build_forward_expand(gf, h_prenorm);

    // -------- in-graph semi-autoregressive draft chain (ds4.c-style) ----------
    // Markov bias + greedy argmax + confidence head per proposal row, chained
    // on-device so the driver reads only K (id, conf_logit) pairs instead of
    // pulling full logit rows and running the Markov GEMV on the host.
    // Gate: enabled by the driver, single-seq token ubatch shaped like a noise
    // block (rows 1.. all carry the same noise token). Chained greedily — the
    // host sampler is greedy too (temp 0, top-k first candidate), so ids match.
    do {
        const auto * m = dynamic_cast<const llama_model_dspark *>(&model);
        if (!cparams.dspark_draft_chain || m == nullptr || m->markov_w1 == nullptr || m->markov_w2 == nullptr) {
            break;
        }
        // A DSpark block is at most block_size wide (5 on DeepSeek-V4-Flash-0731),
        // so anything wider is not a draft block -- the bound also excludes
        // reserve-time worst-case ubatches (n_ubatch wide, dummy all-equal tokens)
        // which would otherwise build a huge chain. The lower bound must stay at 2:
        // the best measured operating point is n_max = 2, and a `< 3` gate silently
        // dropped the chain there and fell back to the host GEMV.
        if (ubatch.embd || ubatch.token == nullptr || n_tokens < 2 || n_tokens > 8 || ubatch.n_seqs_unq != 1) {
            break;
        }
        bool noise_block = true;
        for (uint32_t i = 2; i < ubatch.n_tokens; ++i) {
            noise_block &= ubatch.token[i] == ubatch.token[1];
        }
        if (!noise_block) {
            break;
        }

        // Every row of a DSpark block is a prediction, including the anchor row:
        // the block is [anchor, noise x (block_size-1)] and yields block_size drafts.
        // `n_tokens - 1` is the DFlash 1+N convention (mask tokens sit AT the
        // predicted position, so the anchor row predicts nothing) and leaking it
        // here silently drops the last draft of every block. The reference loops the
        // full block width -- forward_head does `for i in range(self.block_size)` --
        // and so does the host driver in common/speculative.cpp.
        const int64_t K       = n_tokens;
        const int64_t n_vocab = res->t_logits->ne[0];

        ggml_tensor * prev = ggml_view_1d(ctx0, inp_tokens, 1, 0); // anchor token = id_last

        ggml_tensor * meta = nullptr;
        for (int64_t k = 0; k < K; ++k) {
            ggml_tensor * w1p = ggml_get_rows(ctx0, m->markov_w1, prev); // [rank, 1] f32
            ggml_tensor * bias = ggml_mul_mat(ctx0, m->markov_w2, w1p);  // [vocab, 1]

            ggml_tensor * lrow = ggml_view_2d(ctx0, res->t_logits, n_vocab, 1,
                    res->t_logits->nb[1], (size_t) k * res->t_logits->nb[1]);
            ggml_tensor * biased = ggml_add(ctx0, lrow, bias);

            ggml_tensor * id  = ggml_argmax(ctx0, biased); // I32 [1]
            ggml_tensor * idf = ggml_cpy(ctx0, id, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1));

            ggml_tensor * conf;
            if (m->confidence_proj != nullptr) {
                // pre-norm hc_head output, per the reference confidence head
                ggml_tensor * hrow = ggml_view_2d(ctx0, h_prenorm, h_prenorm->ne[0], 1,
                        h_prenorm->nb[1], (size_t) k * h_prenorm->nb[1]);
                ggml_tensor * feat = ggml_concat(ctx0, hrow, w1p, 0); // [n_embd + rank, 1]
                conf = ggml_reshape_1d(ctx0, ggml_mul_mat(ctx0, m->confidence_proj, feat), 1);
            } else {
                conf = ggml_scale(ctx0, idf, 0.0f); // no head -> logit 0 (sigmoid 0.5)
            }

            ggml_tensor * pair = ggml_concat(ctx0, idf, conf, 0); // [2]
            meta = meta ? ggml_concat(ctx0, meta, pair, 0) : pair;

            prev = id;
        }

        cb(meta, "dspark_draft_meta", -1);
        res->t_dspark_meta = meta;
        ggml_build_forward_expand(gf, meta);
    } while (false);
}

// --- Markov head bias (host-side, consumed by the draft driver) --------------
//
// B[prev] = W2 @ W1[prev], with W1 [rank, vocab] (token->rank embed) and
// W2 [rank, vocab] (rank->vocab). Writes a length-vocab bias into `out`.
// W2 is cached to host on first use per model; W1[prev] is read per call.

static bool dspark_type_readable(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

static float dspark_read_f32(const void * base, ggml_type type, int64_t i) {
    switch (type) {
        case GGML_TYPE_F32:  return ((const float *) base)[i];
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(((const ggml_fp16_t *) base)[i]);
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(((const ggml_bf16_t *) base)[i]);
        default:             return 0.0f; // unsupported (e.g. quantized) -> no bias
    }
}

void llama_dspark_markov_bias(const llama_model * model, llama_token prev, float * out) {
    const auto * m = dynamic_cast<const llama_model_dspark *>(model);
    if (m == nullptr || m->markov_w1 == nullptr || m->markov_w2 == nullptr) {
        return; // not a dspark draft model -> leave `out` untouched (additive identity)
    }

    ggml_tensor * w1 = m->markov_w1; // [rank, vocab]
    ggml_tensor * w2 = m->markov_w2; // [rank, vocab]
    const int64_t rank  = w1->ne[0];
    const int64_t vocab = w1->ne[1];

    for (int64_t v = 0; v < vocab; ++v) {
        out[v] = 0.0f;
    }
    if (prev < 0 || prev >= vocab) {
        return;
    }
    if (!dspark_type_readable(w1->type)) {
        // Quantized markov weights are unsupported on this host path. Say so once
        // rather than silently returning a zero bias -- an unbiased chain collapses
        // the block to its anchor (per-position survival 26/1.8/0/0/0 vs 57/17/2/0/0),
        // which reads as a quality problem, not a fault.
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: markov_w1 type %s is not host-readable -- Markov bias DISABLED\n",
                    __func__, ggml_type_name(w1->type));
            warned = true;
        }
        return;
    }

    // read W1[prev] (rank floats)
    std::vector<float> rvec(rank);
    {
        std::vector<uint8_t> row(ggml_row_size(w1->type, rank));
        ggml_backend_tensor_get(w1, row.data(), (size_t) prev * ggml_row_size(w1->type, rank), row.size());
        for (int64_t r = 0; r < rank; ++r) {
            rvec[r] = dspark_read_f32(row.data(), w1->type, r);
        }
    }

    // host cache of W2 (rank*vocab floats), populated once per model
    static std::mutex mtx;
    static std::unordered_map<const llama_model *, std::vector<float>> cache;
    std::lock_guard<std::mutex> lk(mtx);

    auto it = cache.find(model);
    if (it == cache.end()) {
        std::vector<float> w2f((size_t) rank * vocab);
        std::vector<uint8_t> raw(ggml_nbytes(w2));
        ggml_backend_tensor_get(w2, raw.data(), 0, raw.size());
        for (size_t i = 0; i < (size_t) rank * vocab; ++i) {
            w2f[i] = dspark_read_f32(raw.data(), w2->type, (int64_t) i);
        }
        it = cache.emplace(model, std::move(w2f)).first;
    }
    const std::vector<float> & w2f = it->second;

    // out[v] = sum_r W2[r, v] * rvec[r] — the columns are contiguous in the
    // cache, so the inner dot autovectorizes; split the vocab across threads
    // (this sits on the per-slot draft hot path).
    const float * rv  = rvec.data();
    const float * w2p = w2f.data();
    auto gemv_range = [rank, rv, w2p, out](int64_t v0, int64_t v1) {
        for (int64_t v = v0; v < v1; ++v) {
            const float * col = w2p + (size_t) v * rank;
            float acc = 0.0f;
            for (int64_t r = 0; r < rank; ++r) {
                acc += col[r] * rv[r];
            }
            out[v] = acc;
        }
    };

    const int64_t n_th = std::clamp<int64_t>((int64_t) std::thread::hardware_concurrency(), 1, 8);
    if (n_th <= 1 || vocab < 4096) {
        gemv_range(0, vocab);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(n_th);
    const int64_t chunk = (vocab + n_th - 1) / n_th;
    for (int64_t t = 0; t < n_th; ++t) {
        const int64_t v0 = t * chunk;
        const int64_t v1 = std::min(vocab, v0 + chunk);
        if (v0 >= v1) {
            break;
        }
        workers.emplace_back(gemv_range, v0, v1);
    }
    for (auto & w : workers) {
        w.join();
    }
}

// --- Confidence head (host-side, consumed by the draft driver) ---------------
//
// logit = W_conf . [h (n_embd) ; W1[prev] (rank)] with W_conf [n_embd+rank, 1].
// The reference (DeepSpec _confident_prefix_length) truncates the proposal at
// the first row whose sigmoid(logit) falls below the confidence threshold.
// The official DS4-Flash checkpoint carries no bias for this projection.

bool llama_dspark_confidence_logit(const llama_model * model, llama_token prev, const float * h, float * out) {
    const auto * m = dynamic_cast<const llama_model_dspark *>(model);
    if (m == nullptr || m->confidence_proj == nullptr || m->markov_w1 == nullptr || h == nullptr) {
        return false;
    }

    ggml_tensor * w1 = m->markov_w1;      // [rank, vocab]
    ggml_tensor * wc = m->confidence_proj; // [n_embd + rank, 1]
    const int64_t rank   = w1->ne[0];
    const int64_t n_in   = wc->ne[0];
    const int64_t n_embd = n_in - rank;
    if (prev < 0 || prev >= w1->ne[1] || n_embd <= 0) {
        return false;
    }
    if (!dspark_type_readable(w1->type) || !dspark_type_readable(wc->type)) {
        return false;
    }

    // host cache of the projection column (n_in floats), populated once per model
    static std::mutex mtx;
    static std::unordered_map<const llama_model *, std::vector<float>> cache;
    std::lock_guard<std::mutex> lk(mtx);

    auto it = cache.find(model);
    if (it == cache.end()) {
        std::vector<float> wcf((size_t) n_in);
        std::vector<uint8_t> raw(ggml_nbytes(wc));
        ggml_backend_tensor_get(wc, raw.data(), 0, raw.size());
        for (int64_t i = 0; i < n_in; ++i) {
            wcf[i] = dspark_read_f32(raw.data(), wc->type, i);
        }
        it = cache.emplace(model, std::move(wcf)).first;
    }
    const std::vector<float> & wcf = it->second;

    // read W1[prev] (rank floats)
    std::vector<uint8_t> row(ggml_row_size(w1->type, rank));
    ggml_backend_tensor_get(w1, row.data(), (size_t) prev * ggml_row_size(w1->type, rank), row.size());

    float acc = 0.0f;
    for (int64_t i = 0; i < n_embd; ++i) {
        acc += wcf[i] * h[i];
    }
    for (int64_t r = 0; r < rank; ++r) {
        acc += wcf[n_embd + r] * dspark_read_f32(row.data(), w1->type, r);
    }

    *out = acc;
    return true;
}
