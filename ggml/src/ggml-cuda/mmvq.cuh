#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

// [TAG_MMID_BATCH_INVARIANT]
// MUL_MAT_ID takes two different kernels either side of one token: the dedicated
// mul_mat_vec_q_moe above ncols_dst == 1, the general mul_mat_vec_q at exactly 1. They
// cannot agree bitwise -- the MoE kernel walks K with blocks_per_iter = vdr*warp_size/qi
// and reduces warp-locally, while the general kernel folds nwarps partials through
// tmp_shared. With 512 experts at top-10 that is enough to select a different expert set,
// which is a different computation rather than a different rounding.
//
// Building with -DGGML_CUDA_MMID_BATCH_INVARIANT=1 sends every width, including 1, to the
// MoE kernel, which is already invariant in ncols_dst (nothing but the bounds check and
// the token index depend on it). The price is the fused up+gate+SwiGLU, which only ever
// applied at ncols_dst == 1 and has no counterpart in the MoE kernel -- so the fusion must
// be refused as well, or the gate would be silently dropped.
#ifndef GGML_CUDA_MMID_BATCH_INVARIANT
#define GGML_CUDA_MMID_BATCH_INVARIANT 0
#endif

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
