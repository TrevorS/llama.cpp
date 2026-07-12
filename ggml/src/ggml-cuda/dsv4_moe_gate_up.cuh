#include "common.cuh"

// DeepSeek-V4 MoE prefill gate+up+activation fused tile op.
// See ggml_dsv4_moe_gate_up() in ggml.h for the semantics.
//
// Replaces the (gate mul_mat_id) + (up mul_mat_id) + clamp + swiglu_split chain
// for IQ2_XXS expert weights during prefill (n_tokens > 1) with the ds4.c
// expert-tile kernel, producing the activated "mid" tensor
//   mid[t, slot, r] = silu(clamp(gate)) * clamp(up)          (NO routing weight)
// which the existing down mul_mat_id then consumes unchanged.
void ggml_cuda_op_dsv4_moe_gate_up(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
