#include "common.cuh"

#define DSV4_TOPK_SORT_N 4096u   // bitonic block / chunk size (must be >= top_k)

// DeepSeek-V4 lightning-indexer fused score + top-k.
// See ggml_dsv4_lid_topk() in ggml.h for the semantics.
void ggml_cuda_op_dsv4_lid_topk(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
