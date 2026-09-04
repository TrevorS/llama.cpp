#include "scale-silu.cuh"

// The qwen4exp hyper-connection mixer emits scale -> silu on the low-rank path 97 times per
// token; the two launches read and write the same tensor twice. One kernel, same math.
static __global__ void scale_silu_f32(const float * x, float * dst, const float scale, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    const float v = scale * x[i];
    dst[i] = v / (1.0f + expf(-v));
}

void ggml_cuda_op_scale_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src) {
    const ggml_tensor * src0 = src->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    float scale;
    memcpy(&scale, (const float *) src->op_params + 0, sizeof(float));

    const int k = (int) ggml_nelements(src0);
    const int num_blocks = (k + CUDA_SCALE_SILU_BLOCK_SIZE - 1) / CUDA_SCALE_SILU_BLOCK_SIZE;

    scale_silu_f32<<<num_blocks, CUDA_SCALE_SILU_BLOCK_SIZE, 0, ctx.stream()>>>((const float *) src0->data, (float *) dst->data, scale, k);
}
