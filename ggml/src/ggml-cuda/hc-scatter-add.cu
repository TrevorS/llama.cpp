#include "hc-scatter-add.cuh"

// dst[e,c,t] = residual[e,c,t] + block[e,t] * 2*sigmoid(inject[c,t] * inv_hc)
//
// Replaces sigmoid + 2x scale + repeat_4d + mul + add. The repeat is the expensive part:
// it materialises an [n_embd, hc, n_tokens] copy of block that is read once and thrown away.
// Here block is re-read per stream straight from L2 instead.
static __global__ void k_hc_scatter_add_f32(
        const float * __restrict__ residual,
        const float * __restrict__ block,
        const float * __restrict__ inject,
        float       * __restrict__ dst,
        const int n_embd, const int hc, const float inv_hc) {

    const int c = blockIdx.y;            // hyper-connection stream
    const int t = blockIdx.z;            // token

    const float inj = inject[t*hc + c];
    const float w   = 2.0f / (1.0f + expf(-(inj * inv_hc)));

    const long long off_r = ((long long) t*hc + c) * n_embd;
    const long long off_b = ((long long) t) * n_embd;

    for (int e = blockIdx.x*blockDim.x + threadIdx.x; e < n_embd; e += gridDim.x*blockDim.x) {
        dst[off_r + e] = residual[off_r + e] + block[off_b + e] * w;
    }
}

void ggml_cuda_op_hc_scatter_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * residual = dst->src[0];
    const ggml_tensor * block    = dst->src[1];
    const ggml_tensor * inject   = dst->src[2];

    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(block->type    == GGML_TYPE_F32);
    GGML_ASSERT(inject->type   == GGML_TYPE_F32);
    GGML_ASSERT(dst->type      == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(residual));
    GGML_ASSERT(ggml_is_contiguous(block));
    GGML_ASSERT(ggml_is_contiguous(inject));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int n_embd   = residual->ne[0];
    const int hc       = residual->ne[1];
    const int n_tokens = residual->ne[2];

    float inv_hc;
    memcpy(&inv_hc, dst->op_params, sizeof(float));

    const int block_size = 256;
    const int nblk_x     = (n_embd + block_size - 1) / block_size;

    dim3 grid(nblk_x, hc, n_tokens);
    k_hc_scatter_add_f32<<<grid, block_size, 0, ctx.stream()>>>(
        (const float *) residual->data, (const float *) block->data,
        (const float *) inject->data,   (float *) dst->data,
        n_embd, hc, inv_hc);
}
