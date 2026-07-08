#include "dsv4_lid_topk.cuh"

#include <cstdint>

// DSV4_TOPK_SORT_N is defined in dsv4_lid_topk.cuh (shared with supports_op).

// ============================================================================
// DeepSeek-V4 lightning-indexer fused score + top-k
//
// Replaces the 8-op ggml chain
//   k*q matmul -> permute -> relu -> mul(weights) -> sum_rows -> permute
//                -> add(mask) -> top_k
// with two kernels operating on a per-stream scores buffer that lives in the
// CUDA memory pool (NOT the graph compute buffer), so long contexts stay
// allocatable. The [n_ctx x n_tokens x n_head] intermediate is never
// materialized; only a [n_tokens x n_lid] scores buffer is.
//
// score(t,j) = sum_h( relu(q_th . k_j) * weights[h,t] ) + mask[j, t_local, s]
// output     = per-token top-k indices into the context (descending by score,
//              lower-index tie-break), matching ggml_top_k's selection set.
// ============================================================================

// ---------------------------------------------------------------------------
// score kernel
// ---------------------------------------------------------------------------

static __device__ __forceinline__ float dsv4_ldk(const float * p) { return *p; }
static __device__ __forceinline__ float dsv4_ldk(const half  * p) { return __half2float(*p); }

template <typename KT>
static __global__ void dsv4_score_kernel(
        float       * __restrict__ scores,   // [nt, n_lid]
        const float * __restrict__ q,        // [d_idx, n_head, nt] contiguous
        const float * __restrict__ weights,  // [n_head, nt] contiguous
        const KT    * __restrict__ k,        // strided view
        const float * __restrict__ mask,     // strided view
        int64_t nbk2, int64_t nbk3,          // k element strides (per j, per stream)
        int64_t nbm1, int64_t nbm3,          // mask element strides (per t_local, per stream)
        int nt, int nt_s, int n_lid, int d_idx, int n_head, int j_tile) {
    extern __shared__ float smem[];
    float * sq = smem;                 // [d_idx * n_head]
    float * sw = smem + d_idx*n_head;  // [n_head]

    const int t = blockIdx.x;
    if (t >= nt) return;
    const int s       = t / nt_s;
    const int t_local = t % nt_s;

    const int qh_elems = d_idx * n_head;
    for (int i = threadIdx.x; i < qh_elems; i += blockDim.x) {
        sq[i] = q[(int64_t)t*qh_elems + i];
    }
    for (int i = threadIdx.x; i < n_head; i += blockDim.x) {
        sw[i] = weights[(int64_t)t*n_head + i];
    }
    __syncthreads();

    const int warp   = threadIdx.x >> 5;
    const int lane   = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;

    const int j0 = blockIdx.y * j_tile;
    for (int jj = warp; jj < j_tile; jj += nwarps) {
        const int j = j0 + jj;
        if (j >= n_lid) continue;
        const KT * kj = k + (int64_t)s*nbk3 + (int64_t)j*nbk2;
        float acc = 0.0f;
        for (int h = 0; h < n_head; h++) {
            const float * qh = sq + h*d_idx;
            float dot = 0.0f;
            for (int d = lane; d < d_idx; d += 32) {
                dot += qh[d] * dsv4_ldk(kj + d);
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, o);
            dot = __shfl_sync(0xffffffffu, dot, 0);
            acc += fmaxf(dot, 0.0f) * sw[h];
        }
        if (lane == 0) {
            const float mv = mask[(int64_t)s*nbm3 + (int64_t)t_local*nbm1 + (int64_t)j];
            scores[(int64_t)t*n_lid + j] = acc + mv;
        }
    }
}

// ---------------------------------------------------------------------------
// top-k (bitonic, descending, lower-index tie-break) — ported from ds4_cuda.cu
// ---------------------------------------------------------------------------

static __device__ __forceinline__ bool dsv4_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

template <uint32_t SORT_N>
static __device__ void dsv4_bitonic_sort(float * vals, uint32_t * idxs) {
    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = threadIdx.x; i < SORT_N; i += blockDim.x) {
                const uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float    av = vals[i],    bv = vals[other];
                    const uint32_t ai = idxs[i],    bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half ? dsv4_better(bv, bi, av, ai)
                                                : dsv4_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv; idxs[i] = bi;
                        vals[other] = av; idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// single-block-per-token, n_lid <= SORT_N
template <uint32_t SORT_N>
static __global__ void dsv4_topk_single_kernel(
        uint32_t * selected, const float * scores,
        int n_lid, int n_tokens, int top_k) {
    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    __shared__ float    vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];
    const float * row = scores + (int64_t)t * n_lid;
    for (uint32_t i = threadIdx.x; i < SORT_N; i += blockDim.x) {
        if ((int)i < n_lid) { vals[i] = row[i]; idxs[i] = i; }
        else                { vals[i] = -INFINITY; idxs[i] = UINT32_MAX; }
    }
    __syncthreads();
    dsv4_bitonic_sort<SORT_N>(vals, idxs);
    for (int i = threadIdx.x; i < top_k; i += blockDim.x) {
        selected[(int64_t)t*top_k + i] = idxs[i];
    }
}

// per (token, chunk): emit top_k candidate indices for this chunk
template <uint32_t SORT_N>
static __global__ void dsv4_topk_chunk_kernel(
        uint32_t * candidates, const float * scores,
        int n_lid, int n_tokens, int top_k, int candidate_stride) {
    const int t     = blockIdx.x;
    const int chunk = blockIdx.y;
    if (t >= n_tokens) return;
    const uint32_t chunk_start = (uint32_t)chunk * SORT_N;
    if ((int)chunk_start >= n_lid) return;
    const uint32_t chunk_n = ((int)(n_lid - chunk_start) < (int)SORT_N) ? (uint32_t)(n_lid - chunk_start) : SORT_N;
    __shared__ float    vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];
    const float * row = scores + (int64_t)t * n_lid;
    for (uint32_t i = threadIdx.x; i < SORT_N; i += blockDim.x) {
        if (i < chunk_n) { vals[i] = row[chunk_start + i]; idxs[i] = chunk_start + i; }
        else             { vals[i] = -INFINITY; idxs[i] = UINT32_MAX; }
    }
    __syncthreads();
    dsv4_bitonic_sort<SORT_N>(vals, idxs);
    uint32_t * out = candidates + (int64_t)t*candidate_stride + (int64_t)chunk*top_k;
    for (int i = threadIdx.x; i < top_k; i += blockDim.x) out[i] = idxs[i];
}

// merge `set_count` candidate sets (per group) -> top_k, re-reading scores
template <uint32_t SORT_N>
static __global__ void dsv4_topk_merge_kernel(
        uint32_t * out, const uint32_t * candidates, const float * scores,
        int n_lid, int n_tokens, int top_k,
        int n_sets, int merge_group, int candidate_stride, int out_stride,
        int final) {
    const int t     = blockIdx.x;
    const int group = blockIdx.y;
    if (t >= n_tokens) return;
    const int set0 = group * merge_group;
    if (set0 >= n_sets) return;
    int set_count = n_sets - set0;
    if (set_count > merge_group) set_count = merge_group;
    const int candidate_count = set_count * top_k;

    __shared__ float    vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];
    const float    * row  = scores + (int64_t)t * n_lid;
    const uint32_t * cand = candidates + (int64_t)t*candidate_stride + (int64_t)set0*top_k;
    for (uint32_t i = threadIdx.x; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float    v   = -INFINITY;
        if ((int)i < candidate_count) {
            idx = cand[i];
            if (idx < (uint32_t) n_lid) v = row[idx];
        }
        vals[i] = v; idxs[i] = idx;
    }
    __syncthreads();
    dsv4_bitonic_sort<SORT_N>(vals, idxs);
    uint32_t * dst = final ? (out + (int64_t)t*top_k)
                           : (out + (int64_t)t*out_stride + (int64_t)group*top_k);
    for (int i = threadIdx.x; i < top_k; i += blockDim.x) dst[i] = idxs[i];
}

// ---------------------------------------------------------------------------
// host launcher
// ---------------------------------------------------------------------------

static void dsv4_topk_launch(
        ggml_cuda_pool & pool, uint32_t * selected, const float * scores,
        int n_lid, int n_tokens, int top_k, cudaStream_t stream) {
    constexpr uint32_t SORT_N = DSV4_TOPK_SORT_N;
    GGML_ASSERT((uint32_t) top_k <= SORT_N);
    const int block = 1024;

    if ((uint32_t) n_lid <= SORT_N) {
        dsv4_topk_single_kernel<SORT_N><<<n_tokens, block, 0, stream>>>(
                selected, scores, n_lid, n_tokens, top_k);
        return;
    }

    const int n_chunks       = (n_lid + SORT_N - 1) / SORT_N;
    const int merge_group    = SORT_N / (uint32_t) top_k; // >= 1; merge_group*top_k <= SORT_N
    GGML_ASSERT(merge_group >= 2);

    // scratch: candidate tree (per token), sized for all reduction levels.
    int candidate_stride = n_chunks * top_k;
    int n_sets = n_chunks;
    int64_t scratch_per_token = candidate_stride;
    for (int sets = n_chunks; sets > merge_group; ) {
        sets = (sets + merge_group - 1) / merge_group;
        scratch_per_token += (int64_t) sets * top_k;
    }
    ggml_cuda_pool_alloc<uint32_t> scratch_alloc(pool, (size_t) n_tokens * scratch_per_token);
    uint32_t * scratch = scratch_alloc.get();

    // level 0: per-chunk candidates
    uint32_t * cur = scratch;
    int cur_stride = candidate_stride;
    dim3 grid_chunks(n_tokens, n_chunks, 1);
    dsv4_topk_chunk_kernel<SORT_N><<<grid_chunks, block, 0, stream>>>(
            cur, scores, n_lid, n_tokens, top_k, candidate_stride);

    // tree merges until n_sets <= merge_group
    while (n_sets > merge_group) {
        const int next_sets   = (n_sets + merge_group - 1) / merge_group;
        const int next_stride = next_sets * top_k;
        uint32_t * next = cur + (int64_t) n_tokens * cur_stride;
        dim3 grid_merge(n_tokens, next_sets, 1);
        dsv4_topk_merge_kernel<SORT_N><<<grid_merge, block, 0, stream>>>(
                next, cur, scores, n_lid, n_tokens, top_k,
                n_sets, merge_group, cur_stride, next_stride, /*final=*/0);
        cur = next;
        n_sets = next_sets;
        cur_stride = next_stride;
    }

    // final merge -> selected
    dim3 grid_final(n_tokens, 1, 1);
    dsv4_topk_merge_kernel<SORT_N><<<grid_final, block, 0, stream>>>(
            selected, cur, scores, n_lid, n_tokens, top_k,
            n_sets, merge_group, cur_stride, top_k, /*final=*/1);
}

void ggml_cuda_op_dsv4_lid_topk(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q       = dst->src[0];
    const ggml_tensor * k       = dst->src[1];
    const ggml_tensor * weights = dst->src[2];
    const ggml_tensor * mask    = dst->src[3];

    GGML_ASSERT(q->type       == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(mask->type    == GGML_TYPE_F32);
    GGML_ASSERT(dst->type     == GGML_TYPE_I32);
    GGML_ASSERT(k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(k->nb[0] == ggml_type_size(k->type)); // head dim contiguous

    const int d_idx    = q->ne[0];
    const int n_head   = q->ne[1];
    const int n_stream = k->ne[3];
    const int n_lid    = k->ne[2];
    const int nt_s     = mask->ne[1];
    const int nt       = q->ne[2];
    const int n_top_k  = dst->ne[0];

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();

    // scores buffer lives in the pool, not the graph compute buffer.
    ggml_cuda_pool_alloc<float> scores_alloc(pool, (size_t) nt * n_lid);
    float * scores = scores_alloc.get();

    const int64_t nbk2 = k->nb[2] / ggml_type_size(k->type);
    const int64_t nbk3 = k->nb[3] / ggml_type_size(k->type);
    const int64_t nbm1 = mask->nb[1] / sizeof(float);
    const int64_t nbm3 = mask->nb[3] / sizeof(float);

    const int j_tile      = 512;
    const int block       = 256;
    const size_t smem     = ((size_t) d_idx*n_head + n_head) * sizeof(float);
    dim3 grid_score(nt, (n_lid + j_tile - 1) / j_tile, 1);

    const float * q_d = (const float *) q->data;
    const float * w_d = (const float *) weights->data;
    const float * m_d = (const float *) mask->data;

    if (k->type == GGML_TYPE_F16) {
        dsv4_score_kernel<half><<<grid_score, block, smem, stream>>>(
                scores, q_d, w_d, (const half *) k->data, m_d,
                nbk2, nbk3, nbm1, nbm3, nt, nt_s, n_lid, d_idx, n_head, j_tile);
    } else {
        dsv4_score_kernel<float><<<grid_score, block, smem, stream>>>(
                scores, q_d, w_d, (const float *) k->data, m_d,
                nbk2, nbk3, nbm1, nbm3, nt, nt_s, n_lid, d_idx, n_head, j_tile);
    }

    // output is contiguous [n_top_k, nt_s, 1, n_stream] == flat [nt * n_top_k]
    dsv4_topk_launch(pool, (uint32_t *) dst->data, scores, n_lid, nt, n_top_k, stream);
}
