#include "common.cuh"

#define DSV4_TOPK_SORT_N 4096u   // bitonic block / chunk size (must be >= top_k)

// DeepSeek-V4 lightning-indexer fused score + top-k.
// See ggml_dsv4_lid_topk() in ggml.h for the semantics.
void ggml_cuda_op_dsv4_lid_topk(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// B2 sparse-CSA union + membership. See ggml_dsv4_lid_union/_memb in ggml.h.
void ggml_cuda_op_dsv4_lid_union(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_fp4_rt(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_lid_memb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_qat_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_fa_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
