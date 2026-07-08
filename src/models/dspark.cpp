#include "models.h"

#include "llama-kv-cache.h" // complete llama_kv_cache_context for KV injection (cpy_k/cpy_v)

#include "llama-ext.h" // staging API: llama_dspark_markov_bias (host-side Markov bias for the draft driver)

#include "ggml-backend.h" // ggml_backend_tensor_get (Markov weight readback)

#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
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
    ml.get_key(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE, hparams.dsv4_compress_rope_base);
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);

    // target layers -> feature-fusion input width (concat of collapsed hidden states)
    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("DSpark model requires 'target_layers' in GGUF metadata");
    }
    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * hparams.n_embd;

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
    confidence_proj = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ), {n_embd, 1}, llama_model_loader::TENSOR_NOT_REQUIRED);
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

// --- MLA-lite attention over the standard non-causal KV cache ----------------

template <bool is_enc>
ggml_tensor * llama_model_dspark::graph<is_enc>::build_attention(
        const llama_model & model, llm_graph_input_attn_kv * inp_attn,
        ggml_tensor * cur, ggml_tensor * inp_pos, int il) const {
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
    ggml_tensor * out = build_attn(inp_attn, nullptr, nullptr, nullptr,
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
    cb(cur, "fc_out", -1);

    cur = build_norm(cur, model.output_norm_enc, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "enc_norm_out", -1);

    ggml_set_output(cur);
    res->t_h_nextn = cur;

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
    llm_graph_input_attn_kv * inp_attn = build_attn_inp_kv();

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

            ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));
            ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, kv, inp_attn->get_v_idxs(), il));
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

        cur = build_attention(model, inp_attn, cur, inp_pos, il);

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

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// --- Markov head bias (host-side, consumed by the draft driver) --------------
//
// B[prev] = W2 @ W1[prev], with W1 [rank, vocab] (token->rank embed) and
// W2 [rank, vocab] (rank->vocab). Writes a length-vocab bias into `out`.
// W2 is cached to host on first use per model; W1[prev] is read per call.

static float dspark_read_f32(const void * base, ggml_type type, int64_t i) {
    switch (type) {
        case GGML_TYPE_F32: return ((const float *) base)[i];
        case GGML_TYPE_F16: return ggml_fp16_to_fp32(((const ggml_fp16_t *) base)[i]);
        default:            return 0.0f; // unsupported (e.g. quantized) -> no bias
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
    if (w1->type != GGML_TYPE_F32 && w1->type != GGML_TYPE_F16) {
        return; // quantized markov weights unsupported on this host path (see report)
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

    // out[v] = sum_r W2[r, v] * rvec[r]
    for (int64_t v = 0; v < vocab; ++v) {
        const float * col = w2f.data() + (size_t) v * rank;
        float acc = 0.0f;
        for (int64_t r = 0; r < rank; ++r) {
            acc += col[r] * rvec[r];
        }
        out[v] = acc;
    }
}
