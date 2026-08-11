#include "llama-hparams.h"
#include "models.h"

#include "llama-kv-cache-dsv4.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

static float dsv4_rope_attn_factor(float freq_scale, float ext_factor) {
    if (ext_factor == 0.0f) {
        return 1.0f;
    }

    return 1.0f / (1.0f + 0.1f*logf(1.0f/freq_scale));
}

void llama_model_deepseek4::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    if (hparams.n_layer_nextn > 0 && hparams.n_layer_nextn < hparams.n_layer_all) {
        const uint32_t n_layer_main = hparams.n_layer_all - hparams.n_layer_nextn;
        const std::string mtp_probe = "blk." + std::to_string(n_layer_main) + ".nextn.eh_proj.weight";
        if (ml.get_weight(mtp_probe.c_str()) == nullptr) {
            hparams.n_layer_nextn = 0;
        }
    }
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < block_count");

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,     hparams.swiglu_clamp_exp,   hparams.n_layer_all);
    if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP,   hparams.swiglu_clamp_shexp, hparams.n_layer_all, 0)) {
        hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
    }

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

    ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         hparams.dsv4_o_group_count);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           hparams.dsv4_o_lora_rank);
    ml.get_key(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,    hparams.dsv4_compress_rope_base);
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    ml.get_key(LLM_KV_HASH_LAYER_COUNT,                     hparams.dsv4_hash_layer_count);

    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    uint32_t n_compress_ratios = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_COMPRESS_RATIOS, n_compress_ratios);
    if (n_compress_ratios < hparams.n_layer_all) {
        throw std::runtime_error("DeepSeek-V4 compress_ratios is shorter than block_count");
    }
    GGML_ASSERT(n_compress_ratios <= LLAMA_MAX_LAYERS);
    ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios);

    ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func);
    if (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
        throw std::runtime_error("DeepSeek-V4 loader currently expects sqrtsoftplus MoE scoring");
    }
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.set_swa_pattern(0);
    for (uint32_t il = hparams.n_layer(); il < hparams.n_layer_all; ++il) {
        hparams.is_swa_impl[il] = true;
    }

    // nextn/MTP export width: the DS4 MTP head consumes the flattened
    // hc-stream state, not the collapsed hidden state (see graph tail).
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_deepseek4::load_arch_tensors(llama_model_loader & ml) {
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

    const bool mtp_only = (n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const int trunk_flags = mtp_only    ? TENSOR_NOT_REQUIRED : 0;
    const int mtp_flags   = ml.load_mtp ? 0 : TENSOR_SKIP;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN, "weight"),    {hc_dim, hc_mult}, 0);
    hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE, "weight"),  {hc_mult}, 0);
    hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1}, 0);

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];
        const int flags = i < n_layer ? trunk_flags : mtp_flags;

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, flags);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, flags);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, flags);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, flags);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, flags);
        layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, flags);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, flags);
        // for wo_a, the shape in the file is (n_head * n_embd_head / o_groups, o_lora_rank*o_groups)
        // so we reshape here, to avoid reshaping the tensor in the graph
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, flags | TENSOR_ALLOW_RESHAPE);
        layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, flags);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, flags);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, flags);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, flags);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, flags);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, flags);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, flags);

        const int64_t ratio = hparams.dsv4_compress_ratios[i];
        if (ratio != 0) {
            const int64_t coff = ratio == 4 ? 2 : 1;

            layer.attn_comp_wkv   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,   "weight", i), {n_embd, coff * n_embd_head}, flags);
            layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, coff * n_embd_head}, flags);
            layer.attn_comp_ape   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_APE,   "weight", i), {coff * n_embd_head, ratio}, flags);
            layer.attn_comp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM,  "weight", i), {n_embd_head}, flags);

            if (ratio == 4) {
                const int64_t n_embd_indexer = hparams.indexer_head_size;

                layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, flags);
                layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * n_embd_indexer}, flags);

                layer.indexer_comp_wkv   = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_WKV,   "weight", i), {n_embd, 2 * n_embd_indexer}, flags);
                layer.indexer_comp_wgate = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_WGATE, "weight", i), {n_embd, 2 * n_embd_indexer}, flags);
                layer.indexer_comp_ape   = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_APE,   "weight", i), {2 * n_embd_indexer, ratio}, flags);
                layer.indexer_comp_norm  = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_NORM,  "weight", i), {n_embd_indexer}, flags);
            } else if (ratio != 128) {
                throw std::runtime_error("DeepSeek-V4 loader only supports compression ratios 0, 4, and 128");
            }
        }

        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, flags);
        if ((uint32_t) i < hparams.dsv4_hash_layer_count) {
            layer.ffn_gate_tid2eid = create_tensor(tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", i), {n_expert_used, n_vocab}, flags);
        } else {
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, flags);
        }
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, flags);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, flags);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek4::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

static size_t dsv4_elem_offset(const ggml_tensor * t, int64_t i) {
    return ggml_row_size(t->type, i);
}

static ggml_tensor * dsv4_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, dsv4_elem_offset(t, i0));
}

static ggml_tensor * dsv4_view_2d(
        ggml_context * ctx,
        ggml_tensor  * t,
        int64_t        ne0,
        int64_t        ne1,
        int64_t        i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], dsv4_elem_offset(t, i0));
}

static ggml_tensor * dsv4_append_zero_row(ggml_context * ctx, ggml_tensor * t, bool neg_inf) {
    ggml_tensor * row = ggml_view_1d(ctx, t, t->ne[0], 0);
    row = neg_inf ? ggml_scale_bias(ctx, row, 0.0f, -INFINITY) : ggml_scale(ctx, row, 0.0f);
    row = ggml_reshape_2d(ctx, row, t->ne[0], 1);

    return ggml_concat(ctx, t, row, 1);
}

struct dsv4_state_tensors {
    ggml_tensor * kv;
    ggml_tensor * score;
};

static dsv4_state_tensors dsv4_build_state_restore(
        ggml_context * ctx,
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_dsv4_comp_state * state,
        int32_t il) {
    dsv4_state_tensors restored = {
        state->get_kv_all(ctx, il),
        state->get_score_all(ctx, il),
    };

    if (inp.state_restore_src_idxs == nullptr || inp.state_restore_dst_idxs == nullptr) {
        return restored;
    }

    ggml_tensor * kv_rows = ggml_get_rows(ctx, restored.kv, inp.state_restore_src_idxs);
    restored.kv = state->cpy_kv(ctx, kv_rows, inp.state_restore_dst_idxs, il);

    ggml_tensor * score_rows = ggml_get_rows(ctx, restored.score, inp.state_restore_src_idxs);
    restored.score = state->cpy_score(ctx, score_rows, inp.state_restore_dst_idxs, il);

    return restored;
}

static dsv4_state_tensors dsv4_build_state_snapshot(
        ggml_context * ctx,
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_dsv4_comp_state * state,
        ggml_tensor * source_kv,
        ggml_tensor * source_score,
        int32_t il) {
    if (inp.state_snapshot_src_idxs == nullptr || inp.state_snapshot_dst_idxs == nullptr ||
            source_kv == nullptr || source_score == nullptr) {
        return {};
    }

    ggml_tensor * kv_rows = ggml_get_rows(ctx, source_kv, inp.state_snapshot_src_idxs);
    ggml_tensor * kv = state->cpy_kv(ctx, kv_rows, inp.state_snapshot_dst_idxs, il);

    ggml_tensor * score_rows = ggml_get_rows(ctx, source_score, inp.state_snapshot_src_idxs);
    ggml_tensor * score = state->cpy_score(ctx, score_rows, inp.state_snapshot_dst_idxs, il);

    return { kv, score };
}

static constexpr int64_t DSV4_CSA_RATIO  = 4;
static constexpr int64_t DSV4_HCA_RATIO  = 128;

// mean over the hyper-connection streams: [n_embd, hc, n_tokens] -> [n_embd, n_tokens]
static ggml_tensor * dsv4_hc_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }
    return ggml_scale(ctx, acc, 1.0f/hc);
}

static ggml_tensor * dsv4_hc_affine(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * scale,
        ggml_tensor  * base) {
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, base);
    return x;
}

// Env-gated fusion of the DeepSeek-V4 hyper-connection (HC) residual mixing.
// The scalar per-stream loops in build_hc_weighted_sum / build_hc_post emit
// ~87 small elementwise launches per layer (the k_bin_bcast "storm", #2 GPU
// consumer at depth). When LLAMA_DSV4_HC_FUSED=1 they are replaced by the
// GGML_OP_DSV4_HC_FUSED op: a single traffic-minimal kernel that reads each
// operand once and writes the output once, accumulating in the same order as
// the loops (bit-identical). Default OFF -> unchanged scalar graph.
//
// (An earlier LLAMA_DSV4_HC_BATCH graph-op restructure — broadcast-mul + a
// batched mul_mat over the hc axis — regressed -7.8% because this box is
// bandwidth-bound and the transposes/repeat it needed added traffic; it was
// replaced by the fused op. See PROGRESS.md.)
static bool dsv4_hc_fused_enabled() {
    // default ON as of 2026-07-14: weighted_sum/post were greedy-token-identical
    // vs the scalar graph (PROGRESS gate 2) and sinkhorn (mode 2) collapses the
    // unrolled ~85-node chain per call. Set LLAMA_DSV4_HC_FUSED=0 to disable.
    static const bool enabled = []() {
        const char * e = getenv("LLAMA_DSV4_HC_FUSED");
        return !e || e[0] != '0';
    }();
    return enabled;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_weighted_sum(
        ggml_tensor * x,
        ggml_tensor * weights) const {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    if (dsv4_hc_fused_enabled()) {
        return ggml_dsv4_hc_weighted_sum(ctx0, x, weights);
    }

    ggml_tensor * acc = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, weights, 1, nt, weights->nb[1], ih*weights->nb[0]);

        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        acc = acc ? ggml_add(ctx0, acc, cur) : cur;
    }

    return acc;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_sinkhorn(
        ggml_tensor * comb,
        int           il) const {
    GGML_UNUSED(il);

    // comb is [dst_hc, src_hc, n_tokens]. Sinkhorn follows the reference:
    // row softmax over dst, one column normalization, then repeated row/column normalization.
    // Fused: the unrolled chain is ~85 nodes of 1-2us launches on a [hc,hc,nt]
    // tensor, ~30% of decode GPU time at depth (PROGRESS iter 20d) — one
    // kernel replaces softmax + all normalization iterations.
    if (dsv4_hc_fused_enabled()) {
        return ggml_dsv4_hc_sinkhorn(ctx0, comb,
                (int) hparams.dsv4_hc_sinkhorn_iters, hparams.dsv4_hc_eps);
    }

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

ggml_tensor * llama_model_deepseek4::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        ggml_tensor ** post,
        ggml_tensor ** comb,
        int il) const {
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

    ggml_tensor * scale_pre  = dsv4_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = dsv4_view_1d(ctx0, hc_scale, 1, 1);
    ggml_tensor * scale_comb = dsv4_view_1d(ctx0, hc_scale, 1, 2);

    ggml_tensor * base_pre  = dsv4_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post = dsv4_view_1d(ctx0, hc_base, hc, hc);
    ggml_tensor * base_comb = dsv4_view_1d(ctx0, hc_base, hc*hc, 2*hc);

    ggml_tensor * pre = dsv4_view_2d(ctx0, mixes, hc, nt, 0);
    pre = dsv4_hc_affine(ctx0, pre, scale_pre, base_pre);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_pre", il);

    *post = dsv4_view_2d(ctx0, mixes, hc, nt, hc);
    *post = dsv4_hc_affine(ctx0, *post, scale_post, base_post);
    *post = ggml_sigmoid(ctx0, *post);
    *post = ggml_scale(ctx0, *post, 2.0f);
    cb(*post, "hc_post", il);

    *comb = dsv4_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
    *comb = dsv4_hc_affine(ctx0, *comb, scale_comb, base_comb);
    *comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
    *comb = build_hc_sinkhorn(*comb, il);
    cb(*comb, "hc_comb", il);

    return build_hc_weighted_sum(x, pre);
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        ggml_tensor * comb,
        int il) const {
    GGML_UNUSED(il);

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];

    if (dsv4_hc_fused_enabled()) {
        return ggml_dsv4_hc_fused_post(ctx0, x, residual, post, comb);
    }

    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, x, post_dst);

        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], src*residual->nb[1]);
            ggml_tensor * comb_src_dst = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2],
                    dst*comb->nb[0] + src*comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_src_dst));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_head(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_head_mixes", -1);

    ggml_tensor * pre = dsv4_hc_affine(ctx0, mixes, hc_scale, hc_base);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_head_pre", -1);

    return build_hc_weighted_sum(x, pre);
}

ggml_tensor * llama_model_deepseek4::graph::build_compressed_kv_reduce_finish(
        ggml_tensor * values,
        ggml_tensor * scores,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t n_embd_head,
        const char * name,
        int il) const {
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;

    values = ggml_cont(ctx0, ggml_permute(ctx0, values, 1, 0, 2, 3));
    scores = ggml_cont(ctx0, ggml_permute(ctx0, scores, 1, 0, 2, 3));

    ggml_tensor * weights = ggml_soft_max(ctx0, scores);
    ggml_tensor * comp = ggml_mul(ctx0, values, weights);
    comp = ggml_sum_rows(ctx0, comp);
    comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
    cb(comp, name, il);

    comp = build_norm(comp, norm, nullptr, LLM_NORM_RMS, il);
    cb(comp, name, il);

    comp = ggml_rope_ext(ctx0, comp, comp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    comp = ggml_rope_set_offset(comp, n_embd_head_nope);
    cb(comp, name, il);

    return comp;
}

ggml_tensor * llama_model_deepseek4::graph::build_hca_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t n_embd_head,
        const char * name,
        int il) const {
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_blocks         = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    GGML_ASSERT(state_read_idxs->ne[0] == DSV4_HCA_RATIO*n_blocks);
    GGML_ASSERT(kv_state->ne[0] == n_embd_head);
    GGML_ASSERT(score_state->ne[0] == n_embd_head);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    ggml_tensor * kv = ggml_get_rows(ctx0, kv_state, state_read_idxs);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, DSV4_HCA_RATIO, n_blocks);
    cb(kv, name, il);

    ggml_tensor * score = ggml_get_rows(ctx0, score_state, state_read_idxs);
    score = ggml_reshape_3d(ctx0, score, n_embd_head, DSV4_HCA_RATIO, n_blocks);
    cb(score, name, il);

    return build_compressed_kv_reduce_finish(kv, score, comp_pos, norm, n_embd_head, name, il);
}

void llama_model_deepseek4::graph::persist_comp_state(
        ggml_tensor * state_kv,
        ggml_tensor * state_score,
        const llama_dsv4_comp_state * comp_state,
        const llm_graph_input_dsv4::comp_input & inp,
        int il) const {
    ggml_tensor * persist_kv    = ggml_get_rows(ctx0, state_kv,    inp.state_persist_src_idxs);
    ggml_tensor * persist_score = ggml_get_rows(ctx0, state_score, inp.state_persist_src_idxs);

    ggml_tensor * out_kv    = comp_state->cpy_kv   (ctx0, persist_kv,    inp.state_persist_dst_idxs, il);
    ggml_tensor * out_score = comp_state->cpy_score(ctx0, persist_score, inp.state_persist_dst_idxs, il);

    ggml_build_forward_expand(gf, out_kv);
    ggml_build_forward_expand(gf, out_score);
}

ggml_tensor * llama_model_deepseek4::graph::build_overlap_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t ratio,
        int64_t n_embd_head,
        const char * name,
        int il) const {
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_blocks         = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    GGML_ASSERT(state_read_idxs->ne[0] == 2*ratio*n_blocks);
    GGML_ASSERT(kv_state->ne[0] == 2*n_embd_head);
    GGML_ASSERT(score_state->ne[0] == 2*n_embd_head);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    kv_state    = dsv4_append_zero_row(ctx0, kv_state,    false);
    score_state = dsv4_append_zero_row(ctx0, score_state, true);

    const int64_t n_read = ratio*n_blocks;

    ggml_tensor * kv_rows    = ggml_get_rows(ctx0, kv_state,    state_read_idxs);
    ggml_tensor * score_rows = ggml_get_rows(ctx0, score_state, state_read_idxs);

    ggml_tensor * kv_prev = ggml_cont(ctx0,
            ggml_view_2d(ctx0, kv_rows, n_embd_head, n_read, kv_rows->nb[1], 0));
    kv_prev = ggml_reshape_3d(ctx0, kv_prev, n_embd_head, ratio, n_blocks);
    cb(kv_prev, name, il);

    ggml_tensor * score_prev = ggml_cont(ctx0,
            ggml_view_2d(ctx0, score_rows, n_embd_head, n_read, score_rows->nb[1], 0));
    score_prev = ggml_reshape_3d(ctx0, score_prev, n_embd_head, ratio, n_blocks);
    cb(score_prev, name, il);

    ggml_tensor * kv_cur = ggml_cont(ctx0,
            ggml_view_2d(ctx0, kv_rows, n_embd_head, n_read, kv_rows->nb[1],
                n_read*kv_rows->nb[1] + ggml_row_size(kv_rows->type, n_embd_head)));
    kv_cur = ggml_reshape_3d(ctx0, kv_cur, n_embd_head, ratio, n_blocks);

    ggml_tensor * score_cur = ggml_cont(ctx0,
            ggml_view_2d(ctx0, score_rows, n_embd_head, n_read, score_rows->nb[1],
                n_read*score_rows->nb[1] + ggml_row_size(score_rows->type, n_embd_head)));
    score_cur = ggml_reshape_3d(ctx0, score_cur, n_embd_head, ratio, n_blocks);

    ggml_tensor * values = ggml_concat(ctx0, kv_prev, kv_cur, 1);
    ggml_tensor * scores = ggml_concat(ctx0, score_prev, score_cur, 1);

    return build_compressed_kv_reduce_finish(values, scores, comp_pos, norm, n_embd_head, name, il);
}

ggml_tensor * llama_model_deepseek4::graph::build_lid_top_k(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];
    const auto & inp_lid = inp_dsv4->get_lid();
    const int64_t n_embd_indexer_head      = hparams.indexer_head_size;
    const int64_t n_embd_indexer_head_rope = hparams.n_rot();
    const int64_t n_embd_indexer_head_nope = n_embd_indexer_head - n_embd_indexer_head_rope;
    const int64_t n_indexer_head           = hparams.indexer_n_head;
    const int64_t nt                       = cur->ne[1];

    GGML_ASSERT(inp_lid.kq_mask);
    GGML_ASSERT(inp_lid.k_rot);
    GGML_ASSERT(n_embd_indexer_head >= n_embd_indexer_head_rope);

    ggml_tensor * indexer_k = inp_dsv4->mctx->get_lid()->get_k(ctx0, il);
    const int64_t n_lid = inp_lid.kq_mask->ne[0];
    GGML_ASSERT(n_lid > 0);
    GGML_ASSERT(n_lid <= indexer_k->ne[2]);

    // Identity shortcut (default ON; LLAMA_DSV4_LID_SHORTCUT=0 disables): with
    // n_lid <= top_k, top-k over exactly n_lid candidates selects every row, so
    // the selection mask reduces to the plain CSA mask (set_rows writes 0 into
    // ALL rows; x + 0 == x, and invalid cells keep their -inf from the mask
    // itself). Skip the whole indexer chain — q/proj matmuls, rope, hadamard,
    // score, top-k — and return null; the consumer attends the full CSA window
    // directly. Bit-exact by that algebra, not an approximation. The n_lid
    // boundary crossing (once per sequence, monotone) lands on a 256-pad step
    // that already rebuilds the graph, so no extra recaptures.
    static const bool dsv4_lid_shortcut = []() {
        const char * e = getenv("LLAMA_DSV4_LID_SHORTCUT");
        return !e || e[0] != '0';
    }();
    if (dsv4_lid_shortcut && n_lid <= (int64_t) hparams.indexer_top_k) {
        return nullptr;
    }

    ggml_tensor * indexer_q = build_lora_mm(layer.indexer_attn_q_b, qr);
    indexer_q = ggml_reshape_3d(ctx0, indexer_q, n_embd_indexer_head, n_indexer_head, nt);
    cb(indexer_q, "lid_q", il);

    indexer_q = ggml_rope_ext(ctx0, indexer_q, inp_pos, nullptr, n_embd_indexer_head_rope,
            rope_type, n_ctx_orig, hparams.dsv4_compress_rope_base, freq_scale,
            ext_factor, dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    indexer_q = ggml_rope_set_offset(indexer_q, n_embd_indexer_head_nope);
    cb(indexer_q, "lid_q_rope", il);

    indexer_q = llama_mul_mat_hadamard(ctx0, indexer_q, inp_lid.k_rot);
    cb(indexer_q, "lid_q_rot", il);

    ggml_tensor * indexer_weights = build_lora_mm(layer.indexer_proj, cur);
    indexer_weights = ggml_scale(ctx0, indexer_weights, 1.0f/sqrtf(float(n_embd_indexer_head*n_indexer_head)));
    cb(indexer_weights, "lid_weights", il);

    indexer_k = ggml_view_4d(ctx0, indexer_k,
            indexer_k->ne[0], indexer_k->ne[1], n_lid, indexer_k->ne[3],
            indexer_k->nb[1], indexer_k->nb[2], indexer_k->nb[3], 0);
    cb(indexer_k, "lid_k", il);

    // Fused lightning-indexer score+top-k (default ON; LLAMA_DSV4_FUSED_LID=0
    // disables). Replaces the 8-op chain below with a single op whose working
    // set is O(chunk x n_tokens) instead of O(n_ctx x n_tokens x n_head), so
    // long contexts stay allocatable — the unfused chain OOMs >= d65536 ub2048.
    static const bool dsv4_fused_lid = []() {
        const char * e = getenv("LLAMA_DSV4_FUSED_LID");
        return !e || e[0] != '0';
    }();
    // Decode (nt==1): at shallow depth the unfused chain's top-k is faster at
    // batch 1, so it stays unfused. But its launch count scales with context
    // (~11k single-block launches/token at d32768 — launch-bound, GPU ~45%
    // idle; see PROGRESS.md iteration 9), so past a depth threshold the fused
    // op wins. Threshold override: LLAMA_DSV4_FUSED_LID_TG_DEPTH (tokens;
    // 0 = always fuse decode, very large = never, i.e. pre-iteration-9).
    //
    // PRECEDENCE: this threshold is NOT consulted in the default config. The
    // guard below is `fused_lid && (nt > 1 || n_lid >= tg_depth || mxfp4)`, and
    // the MXFP4 container is default ON, so the third disjunct short-circuits
    // and decode always fuses; with LLAMA_DSV4_FUSED_LID=0 the whole branch is
    // skipped instead. The only way to reach the threshold is to turn the
    // container off explicitly (LLAMA_DSV4_LID_CACHE_MXFP4=0), which is a
    // diagnostic configuration. Kept for exactly that case.
    static const int64_t dsv4_lid_tg_depth = []() {
        const char * e = getenv("LLAMA_DSV4_FUSED_LID_TG_DEPTH");
        return e ? atoll(e) : (long long) 4096;
    }();
    // Packed MXFP4 lid cache (P3b, default ON; shared gate — see
    // llama_dsv4_lid_cache_mxfp4). When the fused path is off the container is
    // off too, keeping the shallow-decode unfused shortcut / baseline-repro
    // escape hatch working instead of asserting.
    const bool dsv4_lid_cache_mxfp4 = llama_dsv4_lid_cache_mxfp4();
    if (dsv4_fused_lid && (nt > 1 || n_lid >= dsv4_lid_tg_depth || dsv4_lid_cache_mxfp4)) {
        ggml_tensor * fq = ggml_cont(ctx0, indexer_q);       // [d_idx, n_head, nt]
        ggml_tensor * fw = ggml_cont(ctx0, indexer_weights); // [n_head, nt]
        // Our fused lid_topk op requires an F32 mask. Since upstream's fused
        // lightning-indexer (cparams.fused_lid, default on) makes the lid mask
        // F16, cast it back for our path — the two fused indexers coexist.
        ggml_tensor * lid_mask = inp_lid.kq_mask->type == GGML_TYPE_F32
            ? inp_lid.kq_mask
            : ggml_cast(ctx0, inp_lid.kq_mask, GGML_TYPE_F32);
        const uint32_t n_top_k = n_lid < (int64_t) hparams.indexer_top_k ? (uint32_t) n_lid : hparams.indexer_top_k;
        ggml_tensor * top_k = ggml_dsv4_lid_topk(ctx0, fq, indexer_k, fw, lid_mask, n_top_k);
        cb(top_k, "lid_top_k", il);
        return top_k;
    }

    const int64_t n_stream = indexer_k->ne[3];
    indexer_q = ggml_view_4d(ctx0, indexer_q,
            indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2]/n_stream, n_stream,
            indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3]/n_stream, 0);
    indexer_weights = ggml_view_4d(ctx0, indexer_weights,
            indexer_weights->ne[0], indexer_weights->ne[1]/n_stream, indexer_weights->ne[2], n_stream,
            indexer_weights->nb[1], indexer_weights->nb[2]/n_stream, indexer_weights->nb[3]/n_stream, 0);

    ggml_tensor * indexer_score = nullptr;
    if (cparams.fused_lid) {
        indexer_score = ggml_lightning_indexer(ctx0, indexer_q, indexer_k, indexer_weights, inp_lid.kq_mask);
        cb(indexer_score, "lid_score_masked", il);
        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, indexer_score, il});
    } else {
        indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);
        cb(indexer_q, "lid_q", il);
        indexer_k = ggml_permute(ctx0, indexer_k, 0, 2, 1, 3);
        cb(indexer_k, "lid_k", il);

        ggml_tensor * indexer_kq = ggml_mul_mat(ctx0, indexer_k, indexer_q);
        cb(indexer_kq, "lid_kq", il);

        indexer_kq = ggml_cont(ctx0, ggml_permute(ctx0, indexer_kq, 2, 1, 0, 3));
        cb(indexer_kq, "lid_kq", il);

        indexer_score = ggml_relu(ctx0, indexer_kq);
        indexer_score = ggml_mul(ctx0, indexer_score, indexer_weights);
        indexer_score = ggml_sum_rows(ctx0, indexer_score);
        indexer_score = ggml_cont(ctx0, ggml_permute(ctx0, indexer_score, 2, 1, 0, 3));
        cb(indexer_score, "lid_score", il);

        indexer_score = ggml_add(ctx0, indexer_score, inp_lid.kq_mask);
        cb(indexer_score, "lid_score_masked", il);
    }

    const uint32_t n_top_k = indexer_score->ne[0] < hparams.indexer_top_k ? indexer_score->ne[0] : hparams.indexer_top_k;
    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
    cb(top_k, "lid_top_k", il);

    return top_k;
}

ggml_tensor * llama_model_deepseek4::graph::build_top_k_mask(
        ggml_tensor * kq_mask,
        ggml_tensor * top_k,
        const char * name,
        int il) const {
    GGML_ASSERT(kq_mask);
    GGML_ASSERT(top_k);

    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3],
            kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1,
            top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k,
            kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3],
            kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);
    cb(kq_mask_top_k, name, il);

    return kq_mask_top_k;
}

ggml_tensor * llama_model_deepseek4::graph::build_csa_lid_attention(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    const auto & inp_csa = inp_dsv4->get_csa();
    GGML_ASSERT(inp_csa.kq_mask);

    ggml_tensor * top_k = build_lid_top_k(model, inp_dsv4, qr, cur, inp_pos, il);

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
    cb(raw_k, "csa_raw_k", il);

    ggml_tensor * csa_k = inp_dsv4->mctx->get_csa()->get_k(ctx0, il);
    const int64_t n_csa = inp_csa.kq_mask->ne[0];
    GGML_ASSERT(n_csa > 0);
    GGML_ASSERT(n_csa <= csa_k->ne[2]);

    csa_k = ggml_view_4d(ctx0, csa_k,
            csa_k->ne[0], csa_k->ne[1], n_csa, csa_k->ne[3],
            csa_k->nb[1], csa_k->nb[2], csa_k->nb[3], 0);
    cb(csa_k, "csa_comp_k", il);

    // Decode-side gathered sparse attention (LLAMA_DSV4_CSA_GATHER): instead of
    // a dense [n_csa] top-k mask + FA over all CSA cells, gather the n_top_k
    // selected CSA rows and attend only those + the raw window. Depth-flat CSA
    // attention (cost ~ n_top_k, not n_csa). Only for nt_s==1 (one query per
    // stream) — per-token index divergence within a tile needs the union path.
    static const bool dsv4_csa_gather = []() {
        // Default ON; LLAMA_DSV4_CSA_GATHER=0 disables.
        const char * e = getenv("LLAMA_DSV4_CSA_GATHER");
        return !e || e[0] != '0';
    }();
    // top_k == nullptr: identity regime (n_lid <= indexer_top_k, every CSA row
    // selected — see the shortcut in build_lid_top_k)
    const int64_t nt_s     = top_k ? top_k->ne[1] : 0;
    const int64_t n_top_k  = top_k ? top_k->ne[0] : 0;
    const int64_t n_stream = top_k ? top_k->ne[3] : 0;
    // B2 per-tile union gather (LLAMA_DSV4_CSA_TILE=W): split the ubatch into
    // T = nt_s/W tiles of W consecutive tokens; per tile, gather the union of
    // the tokens' top-k CSA cells (padded to u_cap) and attend raw window +
    // per-tile union via a dim-3-batched FA (tiles ride the stream mechanism
    // of build_attn_mha). Exact when the tile union fits u_cap (overflow drops
    // highest-index cells). Only pays off at depth (small-nb FA runs at ~12 vs
    // 41 TFLOPS): gate on n_csa >= LLAMA_DSV4_CSA_TILE_MIN (default 12288).
    static const int64_t dsv4_tile_w = []() {
        const char * e = getenv("LLAMA_DSV4_CSA_TILE");
        // default ON (W=16) as of 2026-07-13: gates passed (PPL identical,
        // passkey 5/5 at 42k incl past-raw-window keys, deterministic,
        // +12.2% pp@d65k / +32.1% pp@d131k on IQ3). Set 0 to disable.
        return e ? atoll(e) : (long long) 16;
    }();
    static const int64_t dsv4_tile_ucap = []() {
        const char * e = getenv("LLAMA_DSV4_CSA_TILE_UCAP");
        // 4096 covers the measured W=16 union mean+tail at d65k-d131k (mean
        // ~2400, max ~5200; only the tail few % of tiles truncate). Must keep
        // n_raw+u_cap 256-aligned for CUDA FA.
        return e ? atoll(e) : (long long) 4096;
    }();
    static const int64_t dsv4_tile_min = []() {
        const char * e = getenv("LLAMA_DSV4_CSA_TILE_MIN");
        return e ? atoll(e) : (long long) 12288;
    }();
    static const bool dsv4_fa_split = []() {
        const char * e = getenv("LLAMA_DSV4_FA_SPLIT");
        return !e || e[0] != '0';
    }();
    ggml_tensor * k_all     = nullptr;
    ggml_tensor * kq_mask   = nullptr;
    ggml_tensor * split_out = nullptr; // set by the split-attention path (bypasses build_attn_mha)
    ggml_tensor * raw_mask = inp_attn->get_kq_mask();
    const int64_t n_raw = raw_k->ne[2];
    if (top_k && dsv4_tile_w > 0 && nt_s > dsv4_tile_w && n_stream == 1 &&
            nt_s % dsv4_tile_w == 0 && n_csa >= dsv4_tile_min && dsv4_tile_ucap < n_csa &&
            (n_raw + dsv4_tile_ucap) % 256 == 0 && raw_mask->ne[1] == nt_s) {
        const int64_t W_t   = dsv4_tile_w;
        const int64_t T_t   = nt_s / W_t;
        const int64_t u_cap = dsv4_tile_ucap;

        const bool use_split = dsv4_fa_split && cparams.flash_attn && T_t <= raw_k->ne[0];

        ggml_tensor * uni  = ggml_dsv4_lid_union(ctx0, top_k, n_csa, u_cap, W_t); // [u_cap,T,1,1]
        cb(uni, "csa_union_idx", il);
        ggml_tensor * memb = ggml_dsv4_lid_memb(ctx0, top_k, uni, n_csa); // [u_cap,nt_s,1,1] f32

        // gather all per-tile unions with flat ids into the shared CSA cache
        ggml_tensor * csa_src = ggml_view_4d(ctx0, csa_k,
                csa_k->ne[0], n_csa, 1, 1,
                csa_k->nb[2], csa_k->nb[3], csa_k->nb[3], 0);
        ggml_tensor * uidx = ggml_reshape_4d(ctx0, uni, u_cap*T_t, 1, 1, 1);
        ggml_tensor * gathered = ggml_get_rows(ctx0, csa_src, uidx); // [hd, u_cap*T, 1, 1]
        if (gathered->type != raw_k->type) {
            gathered = ggml_cast(ctx0, gathered, raw_k->type);
        }
        gathered = ggml_reshape_4d(ctx0, gathered, csa_k->ne[0], 1, u_cap, T_t);
        cb(gathered, "csa_tile_k", il);

        if (memb->type != raw_mask->type) {
            memb = ggml_cast(ctx0, memb, raw_mask->type);
        }
        memb = ggml_reshape_4d(ctx0, memb, u_cap, W_t, 1, T_t);

        // Split-attention + LSE merge (LLAMA_DSV4_FA_SPLIT, default ON): the
        // raw window is shared by all tiles, so instead of physically
        // replicating it T times (repeat_4d defeats L2 and the tiled FA runs
        // at small-nb occupancy), run (i) one dense FA over the raw window at
        // full batch width (ne3=1) and (ii) a tiled FA over ONLY the per-tile
        // unions, both emitting per-row LSE, then merge. Exact same math as
        // the concat path (softmax over the disjoint KV union); the attention
        // sink lives in the raw half's LSE. T <= hd keeps the LSE tail slice
        // within the FA result (constructor constraint DV >= ne3).
        if (use_split) {
            // dense raw half: all queries vs the shared raw window, ne3 = 1
            ggml_tensor * q_full = ggml_permute(ctx0, q, 0, 2, 1, 3);     // [hd, nt_s, n_head, 1]
            ggml_tensor * k_raw  = ggml_permute(ctx0, raw_k, 0, 2, 1, 3); // [hd, n_raw, 1, 1]
            ggml_tensor * v_raw  = ggml_permute(ctx0, raw_k, 0, 2, 1, 3);
            ggml_tensor * fa_raw = ggml_flash_attn_ext_with_lse(ctx0, q_full, k_raw, v_raw,
                    raw_mask, kq_scale, 0.0f, 0.0f);
            res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, fa_raw, il});
            ggml_flash_attn_ext_add_sinks(fa_raw, sinks);
            ggml_flash_attn_ext_set_prec (fa_raw, GGML_PREC_F32);
            cb(fa_raw, "csa_fa_raw", il);

            // union remainder half: per-tile disjoint unions, tiles ride ne3
            ggml_tensor * q_tiles = ggml_view_4d(ctx0, q,
                    q->ne[0], q->ne[1], W_t, T_t,
                    q->nb[1], q->nb[2], q->nb[3]/T_t, 0);
            q_tiles = ggml_permute(ctx0, q_tiles, 0, 2, 1, 3);               // [hd, W, n_head, T]
            ggml_tensor * k_uni  = ggml_permute(ctx0, gathered, 0, 2, 1, 3); // [hd, u_cap, 1, T]
            ggml_tensor * v_uni  = ggml_permute(ctx0, gathered, 0, 2, 1, 3);
            ggml_tensor * fa_uni = ggml_flash_attn_ext_with_lse(ctx0, q_tiles, k_uni, v_uni,
                    memb, kq_scale, 0.0f, 0.0f);
            res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, fa_uni, il});
            ggml_flash_attn_ext_set_prec(fa_uni, GGML_PREC_F32);
            cb(fa_uni, "csa_fa_uni", il);

            // rows align between the halves (g = t*W + iq): [hd,n_head,nt_s,1+1]
            // and [hd,n_head,W,T+1] share the same row order and tail offset
            ggml_tensor * merged = ggml_dsv4_fa_merge(ctx0, fa_raw, fa_uni); // [hd, n_head, nt_s, 1]
            cb(merged, "csa_fa_merge", il);

            split_out = ggml_reshape_2d(ctx0, merged, merged->ne[0]*merged->ne[1], merged->ne[2]*merged->ne[3]);
        } else {
            // raw window is shared by all tiles -> repeat along the tile dim
            ggml_tensor * raw_rep = ggml_repeat_4d(ctx0, raw_k,
                    raw_k->ne[0], raw_k->ne[1], n_raw, T_t);
            k_all = ggml_concat(ctx0, raw_rep, gathered, 2); // [hd, 1, n_raw+u_cap, T]

            ggml_tensor * raw_mask_t = ggml_reshape_4d(ctx0, raw_mask,
                    raw_mask->ne[0], W_t, 1, T_t);
            kq_mask = ggml_concat(ctx0, raw_mask_t, memb, 0); // [n_raw+u_cap, W, 1, T]
        }
    } else if (top_k && dsv4_csa_gather && nt_s == 1 && n_csa > n_top_k) {
        // present csa_k [hd,1,n_csa,n_stream] as [hd, n_csa, n_stream, 1] and
        // the indices [n_top_k,1,1,n_stream] as [n_top_k, n_stream, 1, 1].
        ggml_tensor * csa_src = ggml_view_4d(ctx0, csa_k,
                csa_k->ne[0], n_csa, n_stream, 1,
                csa_k->nb[2], csa_k->nb[3], csa_k->nb[3]*n_stream, 0);
        ggml_tensor * idx = ggml_view_4d(ctx0, top_k,
                n_top_k, n_stream, 1, 1,
                top_k->nb[3], top_k->nb[3]*n_stream, top_k->nb[3]*n_stream, 0);
        ggml_tensor * gathered = ggml_get_rows(ctx0, csa_src, idx); // [hd, n_top_k, n_stream, 1] f32
        if (gathered->type != raw_k->type) {
            gathered = ggml_cast(ctx0, gathered, raw_k->type);
        }
        gathered = ggml_reshape_4d(ctx0, gathered, csa_k->ne[0], 1, n_top_k, n_stream);
        cb(gathered, "csa_gathered_k", il);

        k_all = ggml_concat(ctx0, raw_k, gathered, 2);

        // selected cells are valid by construction (invalid cells have -inf
        // score and are never in the top-k at depth) -> zero mask for the
        // gathered block.
        ggml_tensor * csa_mask = ggml_new_tensor_4d(ctx0, raw_mask->type,
                n_top_k, raw_mask->ne[1], raw_mask->ne[2], raw_mask->ne[3]);
        csa_mask = ggml_fill(ctx0, csa_mask, 0.0f);
        kq_mask = ggml_concat(ctx0, raw_mask, csa_mask, 0);
    } else if (top_k) {
        k_all = ggml_concat(ctx0, raw_k, csa_k, 2);

        ggml_tensor * csa_mask = build_top_k_mask(inp_csa.kq_mask, top_k, "csa_top_k_mask", il);
        kq_mask = ggml_concat(ctx0, raw_mask, csa_mask, 0);
    } else {
        // identity regime: every CSA row is selected, the top-k mask reduces to
        // the plain CSA mask (see build_lid_top_k shortcut). Re-check the
        // predicate on the CONSUMER's width — the skip decision was made on
        // n_lid, and n_lid == n_csa is an invariant (same reserve plan, same
        // 256-pad), not a guarantee.
        GGML_ASSERT(n_csa <= (int64_t) hparams.indexer_top_k);
        GGML_ASSERT(inp_dsv4->get_lid().kq_mask->ne[0] == n_csa);

        k_all   = ggml_concat(ctx0, raw_k, csa_k, 2);
        kq_mask = ggml_concat(ctx0, raw_mask, inp_csa.kq_mask, 0);
    }
    ggml_tensor * out;
    if (split_out) {
        out = split_out;
    } else {
        cb(k_all, "csa_k_all", il);
        cb(kq_mask, "csa_lid_kq_mask", il);

        out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, sinks, nullptr, kq_scale, il);
    }
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_csa_lid", il);

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_hca_attention(
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    const auto & inp_hca = inp_dsv4->get_hca();
    GGML_ASSERT(inp_hca.kq_mask);

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
    cb(raw_k, "hca_raw_k", il);

    ggml_tensor * hca_k = inp_dsv4->mctx->get_hca()->get_k(ctx0, il);
    const int64_t n_hca = inp_hca.kq_mask->ne[0];
    GGML_ASSERT(n_hca > 0);
    GGML_ASSERT(n_hca <= hca_k->ne[2]);

    hca_k = ggml_view_4d(ctx0, hca_k,
            hca_k->ne[0], hca_k->ne[1], n_hca, hca_k->ne[3],
            hca_k->nb[1], hca_k->nb[2], hca_k->nb[3], 0);
    cb(hca_k, "hca_comp_k", il);

    ggml_tensor * k_all = ggml_concat(ctx0, raw_k, hca_k, 2);
    cb(k_all, "hca_k_all", il);

    ggml_tensor * raw_mask = inp_attn->get_kq_mask();
    ggml_tensor * hca_mask = inp_hca.kq_mask;

    ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, hca_mask, 0);
    cb(kq_mask, "hca_kq_mask", il);

    ggml_tensor * out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, sinks, nullptr, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_hca", il);

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_raw_attention(
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * sinks,
        float kq_scale,
        int il,
        int row) const {
    GGML_ASSERT(hparams.is_swa(il));

    ggml_tensor * k_rot = inp_attn->self_k_rot;

    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_cur = inp_attn->mctx;

    ggml_tensor * k_idxs = inp_attn->get_k_idxs();
    if (row >= 0) {
        k_idxs = ggml_view_1d(ctx0, k_idxs, 1, (size_t) row * k_idxs->nb[0]);
    }

    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, kv, k_idxs, il));

    ggml_tensor * kq_mask = inp_attn->get_kq_mask();
    if (row >= 0) {
        // single-row slice of the full-ubatch mask (column `row`)
        kq_mask = ggml_view_4d(ctx0, kq_mask, kq_mask->ne[0], 1, kq_mask->ne[2], kq_mask->ne[3],
                kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3], (size_t) row * kq_mask->nb[1]);
    }

    ggml_tensor * k = mctx_cur->get_k(ctx0, il);

    ggml_tensor * out = build_attn_mha(q, k, k, nullptr, kq_mask, sinks, nullptr, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_raw", il);

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_attention(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il,
        int row) const {
    return build_attention_impl(model, inp_dsv4, nullptr, cur, inp_pos, il, row);
}

ggml_tensor * llama_model_deepseek4::graph::build_attention(
        const llama_model & model,
        llm_graph_input_attn_k_iswa * inp_mtp,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    return build_attention_impl(model, nullptr, inp_mtp, cur, inp_pos, il, -1);
}

ggml_tensor * llama_model_deepseek4::graph::build_attention_impl(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_attn_k_iswa * inp_mtp,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il,
        int row) const {
    GGML_ASSERT((inp_dsv4 == nullptr) != (inp_mtp == nullptr));

    const auto & layer = model.layers[il];
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4 ? inp_dsv4->get_raw() : nullptr;

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t n_heads_group    = n_head / n_groups;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = n_heads_group*n_embd_head;
    const int64_t nt               = cur->ne[1];

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % n_groups == 0);

    const bool use_compress_rope = hparams.dsv4_compress_ratios[il] != 0;
    const float freq_base_l      = use_compress_rope ? hparams.dsv4_compress_rope_base : freq_base;
    const float freq_scale_l     = use_compress_rope ? freq_scale : 1.0f;
    const float ext_factor_l     = use_compress_rope ? ext_factor : 0.0f;
    const float attn_factor_l    = dsv4_rope_attn_factor(freq_scale_l, ext_factor_l);
    const float beta_fast_l      = use_compress_rope ? beta_fast : 0.0f;
    const float beta_slow_l      = use_compress_rope ? beta_slow : 0.0f;
    const int32_t n_ctx_orig_l   = use_compress_rope ? n_ctx_orig : 0;

    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    cb(qr, "qr", il);

    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr_norm", il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
    q = ggml_rms_norm(ctx0, q, norm_rms_eps);
    cb(q, "q_norm", il);

    q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    q = ggml_rope_set_offset(q, n_embd_head_nope);
    cb(q, "q", il);

    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    cb(kv, "kv_norm", il);

    kv = ggml_rope_ext(ctx0, kv, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    kv = ggml_rope_set_offset(kv, n_embd_head_nope);
    cb(kv, "kv", il);

    const int64_t ratio = hparams.dsv4_compress_ratios[il];
    GGML_ASSERT(inp_dsv4 || ratio == 0);

    ggml_tensor * hca_state_kv    = nullptr;
    ggml_tensor * hca_state_score = nullptr;
    ggml_tensor * hca_source_kv   = nullptr;
    ggml_tensor * hca_source_score = nullptr;
    if (ratio == DSV4_HCA_RATIO && inp_dsv4->get_hca().state_pos) {
        hca_state_kv = build_lora_mm(layer.attn_comp_wkv, cur);
        cb(hca_state_kv, "hca_state_kv", il);

        hca_state_score = build_lora_mm(layer.attn_comp_wgate, cur);
        cb(hca_state_score, "hca_state_score", il);

        ggml_tensor * ape = layer.attn_comp_ape;

        ggml_tensor * ape_rows = ggml_get_rows(ctx0, ape, inp_dsv4->get_hca().state_pos);
        hca_state_score = ggml_add(ctx0, hca_state_score, ape_rows);
        cb(hca_state_score, "hca_state_score_ape", il);

    }

    if (ratio == DSV4_CSA_RATIO && inp_dsv4->get_csa().state_pos) {
        ggml_tensor * csa_state_kv = build_lora_mm(layer.attn_comp_wkv, cur);
        cb(csa_state_kv, "csa_state_kv", il);

        ggml_tensor * csa_state_score = build_lora_mm(layer.attn_comp_wgate, cur);
        cb(csa_state_score, "csa_state_score", il);

        ggml_tensor * csa_ape = layer.attn_comp_ape;

        ggml_tensor * csa_ape_rows = ggml_get_rows(ctx0, csa_ape, inp_dsv4->get_csa().state_pos);
        csa_state_score = ggml_add(ctx0, csa_state_score, csa_ape_rows);
        cb(csa_state_score, "csa_state_score_ape", il);

        GGML_ASSERT(inp_dsv4->get_csa().state_write_idxs);

        const auto * csa_state = inp_dsv4->mctx->get_csa_state();
        const dsv4_state_tensors csa_restored = dsv4_build_state_restore(
                ctx0, inp_dsv4->get_csa(), csa_state, il);
        ggml_tensor * csa_base_kv = dsv4_view_2d(
                ctx0, csa_restored.kv, csa_restored.kv->ne[0], csa_state->get_n_rows(), 0);
        ggml_tensor * csa_base_score = dsv4_view_2d(
                ctx0, csa_restored.score, csa_restored.score->ne[0], csa_state->get_n_rows(), 0);

        ggml_tensor * csa_source_kv = ggml_concat(ctx0, csa_base_kv, csa_state_kv, 1);
        ggml_tensor * csa_source_score = ggml_concat(ctx0, csa_base_score, csa_state_score, 1);

        ggml_tensor * kv_comp_csa_state = build_overlap_compressed_kv_from_state(
                csa_source_kv,
                csa_source_score,
                inp_dsv4->get_csa().state_read_idxs,
                inp_dsv4->get_csa().state_write_pos,
                layer.attn_comp_norm,
                DSV4_CSA_RATIO,
                n_embd_head,
                "csa_state_compress",
                il);

        if (inp_dsv4->get_csa().k_rot) {
            kv_comp_csa_state = llama_mul_mat_hadamard(ctx0, kv_comp_csa_state, inp_dsv4->get_csa().k_rot);
            cb(kv_comp_csa_state, "csa_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_csa()->cpy_k(ctx0,
                    kv_comp_csa_state, inp_dsv4->get_csa().state_write_idxs, il));

        ggml_tensor * csa_snapshot_source_kv = ggml_concat(ctx0,
                csa_restored.kv, csa_state_kv, 1);
        ggml_tensor * csa_snapshot_source_score = ggml_concat(ctx0,
                csa_restored.score, csa_state_score, 1);

        const dsv4_state_tensors csa_snapshot = dsv4_build_state_snapshot(
                ctx0, inp_dsv4->get_csa(), csa_state, csa_snapshot_source_kv, csa_snapshot_source_score, il);
        if (csa_snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, csa_snapshot.kv);
        }
        if (csa_snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, csa_snapshot.score);
        }

        persist_comp_state(csa_state_kv, csa_state_score,
                inp_dsv4->mctx->get_csa_state(), inp_dsv4->get_csa(), il);

        ggml_tensor * lid_state_kv = build_lora_mm(layer.indexer_comp_wkv, cur);
        cb(lid_state_kv, "lid_state_kv", il);

        ggml_tensor * lid_state_score = build_lora_mm(layer.indexer_comp_wgate, cur);
        cb(lid_state_score, "lid_state_score", il);

        ggml_tensor * lid_ape = layer.indexer_comp_ape;

        ggml_tensor * lid_ape_rows = ggml_get_rows(ctx0, lid_ape, inp_dsv4->get_lid().state_pos);
        lid_state_score = ggml_add(ctx0, lid_state_score, lid_ape_rows);
        cb(lid_state_score, "lid_state_score_ape", il);

        GGML_ASSERT(inp_dsv4->get_lid().state_write_idxs);

        const auto * lid_state = inp_dsv4->mctx->get_lid_state();
        const dsv4_state_tensors lid_restored = dsv4_build_state_restore(
                ctx0, inp_dsv4->get_lid(), lid_state, il);
        ggml_tensor * lid_base_kv = dsv4_view_2d(
                ctx0, lid_restored.kv, lid_restored.kv->ne[0], lid_state->get_n_rows(), 0);
        ggml_tensor * lid_base_score = dsv4_view_2d(
                ctx0, lid_restored.score, lid_restored.score->ne[0], lid_state->get_n_rows(), 0);

        ggml_tensor * lid_source_kv = ggml_concat(ctx0, lid_base_kv, lid_state_kv, 1);
        ggml_tensor * lid_source_score = ggml_concat(ctx0, lid_base_score, lid_state_score, 1);

        ggml_tensor * kv_comp_lid_state = build_overlap_compressed_kv_from_state(
                lid_source_kv,
                lid_source_score,
                inp_dsv4->get_lid().state_read_idxs,
                inp_dsv4->get_lid().state_write_pos,
                layer.indexer_comp_norm,
                DSV4_CSA_RATIO,
                hparams.indexer_head_size,
                "lid_state_compress",
                il);

        if (inp_dsv4->get_lid().k_rot) {
            kv_comp_lid_state = llama_mul_mat_hadamard(ctx0, kv_comp_lid_state, inp_dsv4->get_lid().k_rot);
            cb(kv_comp_lid_state, "lid_state_compress_rot", il);
        }

        if (llama_dsv4_lid_cache_mxfp4()) {
            // packed container: the official e2m1 QAT rounding folds into the
            // scatter itself (GGML_OP_DSV4_QAT_SET_ROWS)
            ggml_build_forward_expand(gf, inp_dsv4->mctx->get_lid()->cpy_k_qat(ctx0,
                        kv_comp_lid_state, inp_dsv4->get_lid().state_write_idxs, il));
        } else {
            ggml_build_forward_expand(gf, inp_dsv4->mctx->get_lid()->cpy_k(ctx0,
                        kv_comp_lid_state, inp_dsv4->get_lid().state_write_idxs, il));
        }

        ggml_tensor * lid_snapshot_source_kv = ggml_concat(ctx0,
                lid_restored.kv, lid_state_kv, 1);
        ggml_tensor * lid_snapshot_source_score = ggml_concat(ctx0,
                lid_restored.score, lid_state_score, 1);

        const dsv4_state_tensors lid_snapshot = dsv4_build_state_snapshot(
                ctx0, inp_dsv4->get_lid(), lid_state, lid_snapshot_source_kv, lid_snapshot_source_score, il);
        if (lid_snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, lid_snapshot.kv);
        }
        if (lid_snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, lid_snapshot.score);
        }

        persist_comp_state(lid_state_kv, lid_state_score,
                inp_dsv4->mctx->get_lid_state(), inp_dsv4->get_lid(), il);
    }

    const llama_dsv4_comp_state * hca_state = nullptr;
    dsv4_state_tensors hca_restored = {};
    if (ratio == DSV4_HCA_RATIO && inp_dsv4->get_hca().state_write_idxs) {
        GGML_ASSERT(hca_state_kv);
        GGML_ASSERT(hca_state_score);

        hca_state = inp_dsv4->mctx->get_hca_state();
        hca_restored = dsv4_build_state_restore(ctx0, inp_dsv4->get_hca(), hca_state, il);
        ggml_tensor * hca_base_kv = dsv4_view_2d(
                ctx0, hca_restored.kv, hca_restored.kv->ne[0], hca_state->get_n_rows(), 0);
        ggml_tensor * hca_base_score = dsv4_view_2d(
                ctx0, hca_restored.score, hca_restored.score->ne[0], hca_state->get_n_rows(), 0);

        hca_source_kv = ggml_concat(ctx0, hca_base_kv, hca_state_kv, 1);
        hca_source_score = ggml_concat(ctx0, hca_base_score, hca_state_score, 1);

        ggml_tensor * kv_comp_hca = build_hca_compressed_kv_from_state(
                hca_source_kv,
                hca_source_score,
                inp_dsv4->get_hca().state_read_idxs,
                inp_dsv4->get_hca().state_write_pos,
                layer.attn_comp_norm,
                n_embd_head,
                "hca_state_compress",
                il);

        if (inp_dsv4->get_hca().k_rot) {
            kv_comp_hca = llama_mul_mat_hadamard(ctx0, kv_comp_hca, inp_dsv4->get_hca().k_rot);
            cb(kv_comp_hca, "hca_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_hca()->cpy_k(ctx0,
                    kv_comp_hca, inp_dsv4->get_hca().state_write_idxs, il));
    }

    if (ratio == DSV4_HCA_RATIO && inp_dsv4->get_hca().state_pos) {
        GGML_ASSERT(hca_state_kv);
        GGML_ASSERT(hca_state_score);

        if (hca_state == nullptr) {
            hca_state = inp_dsv4->mctx->get_hca_state();
        }
        if (hca_restored.kv == nullptr) {
            hca_restored = dsv4_build_state_restore(ctx0, inp_dsv4->get_hca(), hca_state, il);
        }
        if (hca_source_kv == nullptr || hca_source_score == nullptr) {
            ggml_tensor * hca_base_kv = dsv4_view_2d(
                    ctx0, hca_restored.kv, hca_restored.kv->ne[0], hca_state->get_n_rows(), 0);
            ggml_tensor * hca_base_score = dsv4_view_2d(
                    ctx0, hca_restored.score, hca_restored.score->ne[0], hca_state->get_n_rows(), 0);

            hca_source_kv = ggml_concat(ctx0, hca_base_kv, hca_state_kv, 1);
            hca_source_score = ggml_concat(ctx0, hca_base_score, hca_state_score, 1);
        }

        ggml_tensor * hca_snapshot_source_kv = ggml_concat(ctx0,
                hca_restored.kv, hca_state_kv, 1);
        ggml_tensor * hca_snapshot_source_score = ggml_concat(ctx0,
                hca_restored.score, hca_state_score, 1);

        const dsv4_state_tensors hca_snapshot = dsv4_build_state_snapshot(
                ctx0, inp_dsv4->get_hca(), hca_state, hca_snapshot_source_kv, hca_snapshot_source_score, il);
        if (hca_snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, hca_snapshot.kv);
        }
        if (hca_snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, hca_snapshot.score);
        }

        persist_comp_state(hca_state_kv, hca_state_score,
                inp_dsv4->mctx->get_hca_state(), inp_dsv4->get_hca(), il);
    }

    GGML_ASSERT(row < 0 || ratio == 0); // row slicing only on raw-attention layers

    ggml_tensor * out = nullptr;
    if (inp_mtp) {
        out = build_attn(inp_mtp,
                nullptr, nullptr, nullptr,
                q, kv, kv,
                nullptr, layer.attn_sinks, nullptr,
                1.0f/sqrtf(float(n_embd_head)), il);
        cb(out, "attn_raw", il);
    } else if (ratio == DSV4_CSA_RATIO &&
            inp_dsv4->get_csa().kq_mask &&
            inp_dsv4->get_lid().kq_mask &&
            inp_dsv4->get_lid().k_rot) {
        out = build_csa_lid_attention(model, inp_dsv4, inp_attn, q, kv, qr, cur, inp_pos, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il);
    } else if (ratio == DSV4_HCA_RATIO &&
            inp_dsv4->get_hca().kq_mask) {
        out = build_hca_attention(inp_dsv4, inp_attn, q, kv, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il);
    } else {
        out = build_raw_attention(inp_attn, q, kv, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il, row);
    }

    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    out = ggml_rope_ext_back(ctx0, out, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    out = ggml_rope_set_offset(out, n_embd_head_nope);
    cb(out, "attn_derope", il);

    out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, nt);
    out = ggml_permute(ctx0, out, 0, 2, 1, 3);
    ggml_tensor * oa = ggml_mul_mat(ctx0, layer.wo_a, out);
    cb(oa, "attn_wo_a", il);
    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, nt);

    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "attn_out", il);

    return out;
}

llama_model_deepseek4::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    ggml_tensor * cur;

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();
    ggml_build_forward_expand(gf, inp_attn->self_kq_mask);

    const int64_t hc = hparams.dsv4_hc_mult;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    // extract_layer_inputs() expects [n_embd, n_tokens]; collapse the hc
    // streams by mean-pooling, matching the layer-input reference taps. Index il
    // is the input of layer il; il == n_layer is the final backbone output.
    auto build_layer_inp_tap = [&](int il) {
        if (il >= (int) cparams.embeddings_layer_inp.size() || !cparams.embeddings_layer_inp[il]) {
            return;
        }
        ggml_tensor * x = ggml_is_contiguous(inpL) ? inpL : ggml_cont(ctx0, inpL);
        ggml_tensor * tap = ggml_view_2d(ctx0, x, n_embd, n_tokens, x->nb[2], 0);
        for (int h = 1; h < (int) hc; ++h) {
            tap = ggml_add(ctx0, tap, ggml_view_2d(ctx0, x, n_embd, n_tokens, x->nb[2], (size_t) h*x->nb[1]));
        }
        tap = ggml_scale(ctx0, tap, 1.0f/(float) hc);
        cb(tap, "layer_inp_hc_mean", il);
        ggml_build_forward_expand(gf, tap);
        res->t_layer_inp[il] = tap;
    };

    // ds4-style refusal steering: project the direction out of ffn_out (and,
    // unless LLAMA_CVEC_FFN_ONLY, also attn_out) before its HC-post.
    // ds4's SHIPPED, validated recipe is ffn_out ONLY at scale 2.0-2.5
    // (refusal-ffn.json: flip 0.68@2.0 / 0.89@2.5, broken 0). Adding attn
    // steering was an untested assumption that under-performs — set
    // LLAMA_CVEC_FFN_ONLY=1 to match ds4 exactly.
    static const bool cvec_at_ffn  = getenv("LLAMA_CVEC_AT_FFN") != nullptr;
    static const bool cvec_ffn_only = getenv("LLAMA_CVEC_FFN_ONLY") != nullptr;
    static const bool cvec_at_attn = cvec_at_ffn && !cvec_ffn_only;

    for (int il = 0; il < n_layer; ++il) {
        build_layer_inp_tap(il);

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        cur = build_hc_pre(inpL,
                model.layers[il].hc_attn_fn,
                model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base,
                &post, &comb, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention(model, inp_dsv4, cur, inp_pos, il);

        // attn_out steering, pre-HC (skipped in ds4's ffn-only recipe).
        if (cvec_at_attn) {
            cur = build_cvec(cur, il);
        }

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_hc_pre(inpL,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &post, &comb, il);
        cb(cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        const auto & layer = model.layers[il];
        ggml_tensor * selected_experts = nullptr;
        ggml_tensor * exp_probs_b = layer.ffn_exp_probs_b;
        if ((uint32_t) il < hparams.dsv4_hash_layer_count) {
            selected_experts = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, res->t_inp_tokens);
            exp_probs_b = nullptr;
        }

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, hparams.n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                selected_experts);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        // ffn_out steering (paired with the attn_out steering above) — ds4
        // applies the direction at both, pre-HC. Default (no env) = post-HC l_last.
        if (cvec_at_ffn) {
            cur = build_cvec(cur, il);
        }

        inpL = build_hc_post(cur, residual, post, comb, il);
        if (!cvec_at_ffn) {
            inpL = build_cvec(inpL, il);
        }
        cb(inpL, "l_last", il);
    }

    build_layer_inp_tap(n_layer);

    // Flattened post-final-layer HC state [n_embd*hc, nt]. This is what the
    // DS4 MTP head consumes (prev_hc in ds4.c terms), so it doubles as the
    // nextn embedding export; row selection order mirrors qwen35 (masked:
    // output rows only; unmasked: every token row, MTP needs them all).
    {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);

        if (inp_out_ids && cparams.embeddings_nextn_masked) {
            flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        }

        if (cparams.embeddings_nextn) {
            cb(flat, "h_nextn", -1);
            res->t_h_nextn = flat;
        }

        if (inp_out_ids && !cparams.embeddings_nextn_masked) {
            flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        }

        const int64_t nt_out = inp_out_ids ? n_outputs : n_tokens;
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, nt_out);
    }

    cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "hc_head", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}


llama_model_deepseek4::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "DEEPSEEK4 MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "DEEPSEEK4 MTP currently only supports a single MTP block");
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
            cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
            "nextn_layer_offset out of range [0, n_layer_nextn)");
    GGML_ASSERT(ubatch.token && "DEEPSEEK4 MTP requires token input");

    const int64_t hc = hparams.dsv4_hc_mult;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) (n_embd*hc) && "DEEPSEEK4 MTP hidden width mismatch");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_out());

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    ggml_tensor * tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h_state = ggml_reshape_3d(ctx0, inp->h, n_embd, hc, n_tokens);
    cb(h_state, "mtp_h_state", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_attn_k_iswa * inp_attn = build_attn_inp_k_iswa();

    ggml_tensor * h_norm = build_norm(h_state, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens);
    e_norm = ggml_repeat_4d(ctx0, e_norm, n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * inpL = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(inpL, "mtp_eh_proj", il);

    ggml_tensor * residual = inpL;
    ggml_tensor * post = nullptr;
    ggml_tensor * comb = nullptr;

    ggml_tensor * cur = build_hc_pre(inpL,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            &post, &comb, il);
    cb(cur, "mtp_hc_attn_pre", il);

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    cur = build_attention(model, inp_attn, cur, inp_pos, il);

    inpL = build_hc_post(cur, residual, post, comb, il);
    cb(inpL, "mtp_hc_attn_post", il);

    residual = inpL;
    cur = build_hc_pre(inpL,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            &post, &comb, il);
    cb(cur, "mtp_hc_ffn_pre", il);

    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    GGML_ASSERT((uint32_t) il >= hparams.dsv4_hash_layer_count && "DEEPSEEK4 MTP does not support hash-routed MTP blocks");
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
    cb(moe_out, "mtp_ffn_moe_out", il);

    ggml_tensor * ffn_shexp = build_ffn(cur,
            layer.ffn_up_shexp, nullptr, nullptr,
            layer.ffn_gate_shexp, nullptr, nullptr,
            layer.ffn_down_shexp, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(ffn_shexp, "mtp_ffn_shexp", il);

    cur = ggml_add(ctx0, moe_out, ffn_shexp);
    cb(cur, "mtp_ffn_out", il);

    inpL = build_hc_post(cur, residual, post, comb, il);
    inpL = build_cvec(inpL, il);
    cb(inpL, "mtp_l_out", il);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
    ggml_tensor * h_nextn = ggml_get_rows(ctx0, flat, inp_out_ids);
    cb(h_nextn, "h_nextn", -1);
    res->t_h_nextn = h_nextn;

    inpL = ggml_reshape_3d(ctx0, h_nextn, n_embd, hc, n_outputs);

    cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "mtp_hc_head", -1);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm ? layer.nextn.shared_head_norm : model.output_norm;
    GGML_ASSERT(head_norm_w && "DEEPSEEK4 MTP missing shared head norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "mtp_shared_head_norm", -1);
    res->t_embd = cur;

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    GGML_ASSERT(head_w && "DEEPSEEK4 MTP missing LM head");
    cur = ggml_mul_mat(ctx0, head_w, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
