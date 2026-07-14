#include "common.cuh"

// DeepSeek-V4 hyper-connection (HC) residual mixing, fused.
//
// Replaces the per-stream elementwise mul/add loops in build_hc_weighted_sum /
// build_hc_post (the k_bin_bcast "storm") with a single traffic-minimal kernel:
// each operand is read once and the output written once, with the src/hc sums
// accumulated in the SAME left-to-right order as the scalar ggml graph (using
// __fmul_rn/__fadd_rn to avoid FMA contraction) so the result is bit-identical.
//
// Two modes, selected via op_params[0]:
//   MODE_WEIGHTED_SUM: out[e,t]     = sum_ih x[e,ih,t] * w[ih,t]
//                      src0 x [n_embd,hc,nt], src1 w [hc,nt] -> [n_embd,nt]
//   MODE_POST:         out[e,dst,t] = x[e,t]*post[dst,t]
//                                     + sum_src res[e,src,t]*comb[dst,src,t]
//                      src0 x [n_embd,nt], src1 res [n_embd,hc,nt],
//                      src2 post [hc,nt], src3 comb [dst,src,nt] -> [n_embd,hc,nt]
//   MODE_SINKHORN:     softmax along dst, +eps, col-normalize, then
//                      (iters-1) x (row-, col-normalize) — the whole unrolled
//                      ~85-node chain (per call, on a [hc,hc,nt] tensor) in one
//                      launch. op_params[1] = iters, op_params[2] = eps (f32).
//                      src0 comb [dst,src,nt] -> [dst,src,nt]
// See ggml_dsv4_hc_weighted_sum() / ggml_dsv4_hc_post() in ggml.h.

#define GGML_DSV4_HC_MODE_WEIGHTED_SUM 0
#define GGML_DSV4_HC_MODE_POST         1
#define GGML_DSV4_HC_MODE_SINKHORN     2
#define GGML_DSV4_HC_MAX               8   // max supported hc_mult

void ggml_cuda_op_dsv4_hc_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
