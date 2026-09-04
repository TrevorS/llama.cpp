#include "common.cuh"

#define CUDA_SCALE_SILU_BLOCK_SIZE 256

// fused GGML_OP_SCALE + GGML_UNARY_OP_SILU: dst = silu(scale * x)
void ggml_cuda_op_scale_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src);
