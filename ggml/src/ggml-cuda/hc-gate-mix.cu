#include "hc-gate-mix.cuh"

// dst[e,t] = scale * sum_c x[e + c*n_embd, t] * gate[e + c*n_embd, t]
//
// Replaces mul + cont + (hc-1) adds over strided views + scale. The win is the gated
// intermediate: unfused, x*gate is written as a full [hc*n_embd, n_tokens] tensor and
// read straight back by the stream-collapse adds. Here it stays in registers.
// hc is small (4 for qwen4exp) so the reduction is a plain unrolled loop, and e is the
// fastest-varying index so all loads stay coalesced.
static __global__ void k_hc_gate_mix_f32(
        const float * __restrict__ x,
        const float * __restrict__ gate,
        float       * __restrict__ dst,
        const int n_embd, const int hc, const float scale) {

    const int t = blockIdx.y;
    const long long base = (long long) t * hc * n_embd;

    for (int e = blockIdx.x*blockDim.x + threadIdx.x; e < n_embd; e += gridDim.x*blockDim.x) {
        float acc = 0.0f;
        for (int c = 0; c < hc; ++c) {
            const long long i = base + (long long) c*n_embd + e;
            acc += x[i] * gate[i];
        }
        dst[(long long) t*n_embd + e] = acc * scale;
    }
}

void ggml_cuda_op_hc_gate_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x    = dst->src[0];
    const ggml_tensor * gate = dst->src[1];

    GGML_ASSERT(x->type    == GGML_TYPE_F32);
    GGML_ASSERT(gate->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(gate));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int hc = ggml_get_op_params_i32(dst, 0);
    const float scale = ggml_get_op_params_f32(dst, 1);

    const int n_embd   = dst->ne[0];
    const int n_tokens = dst->ne[1];

    GGML_ASSERT(x->ne[0] == (int64_t) hc*n_embd);

    const int block_size = 256;
    const int nblk_x     = (n_embd + block_size - 1) / block_size;

    dim3 grid(nblk_x, n_tokens, 1);
    k_hc_gate_mix_f32<<<grid, block_size, 0, ctx.stream()>>>(
        (const float *) x->data, (const float *) gate->data, (float *) dst->data,
        n_embd, hc, scale);
}
