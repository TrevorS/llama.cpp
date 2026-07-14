#include "dsv4_lid_topk.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <mma.h>

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

// ---------------------------------------------------------------------------
// fp4 QAT fake-quant (LLAMA_DSV4_LID_FP4) — e2m1 block-32 round trip matching
// ds4.c dsv4_fp4_act_quantize_row_inplace_cpu. The indexer q/k arrive already
// hadamard-rotated (deepseek4.cpp), so we apply the e2m1 simulation only. This
// is the model's official (QAT) indexer numeric; the CPU reference in
// ggml-cpu/ops.cpp mirrors it under the same env gate.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ float dsv4_e2m1_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax   = fminf(fabsf(x), 6.0f);
    const float lv[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    int   best = 0;
    float bd   = fabsf(ax - lv[0]);
#pragma unroll
    for (int i = 1; i < 8; i++) {
        const float d = fabsf(ax - lv[i]);
        // nearest, even-index tie-break (matches ds4.c device dequant)
        if (d < bd || (d == bd && (i & 1) == 0 && (best & 1) != 0)) { best = i; bd = d; }
    }
    return sign * lv[best];
}

// One block per d_idx-wide row (blockDim.x == d_idx, a multiple of 32); each
// thread owns one component, block-32 amax via warp shuffle. Writes a
// contiguous f32 [d_idx, n_rows] fake-quantized copy.
template <typename KT>
static __global__ void dsv4_fp4_quant_kernel(
        float * __restrict__ out, const KT * __restrict__ in,
        int64_t row_stride0, int64_t row_stride1, int rows_per_group, int d_idx) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int64_t g = row / rows_per_group;
    const int64_t r = row % rows_per_group;
    const KT * ir = in + g * row_stride1 + r * row_stride0;
    float v = dsv4_ldk(ir + tid);
    float a = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    a = fmaxf(a, 7.052966104933725e-38f);
    const float scale = exp2f(ceilf(log2f(a / 6.0f)));
    const float q = fminf(6.0f, fmaxf(-6.0f, v / scale));
    out[(int64_t) row * d_idx + tid] = dsv4_e2m1_dequant(q) * scale;
}

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
// tensor-core score kernel (head_dim == 128, one CUDA stream-group per launch)
//
// Ported from ds4_cuda.cu indexer_scores_wmma128_kernel. Computes, for a 16-token
// x 128-comp tile, score(t,j) = sum_h relu(q_th . k_j) * weights[h,t] + mask[j],
// using 16x16x16 fp16 wmma fragments (8 warps, each owning one 16-comp sub-tile).
// q is rounded to fp16 for the matmul; the running-topk stage re-reads these
// scores, so selection matches the CPU reference to within fp16 boundary noise.
//
// One CUDA-stream-group (s) per launch: pointers are pre-offset so the kernel
// addresses tokens/comps/mask locally. This keeps a 16-token tile within a single
// stream even when nt_s is not a multiple of 16.
// ---------------------------------------------------------------------------

template <typename KT>
static __global__ void dsv4_score_wmma128_kernel(
        float       * __restrict__ scores,   // [nt_s, n_lid] for this stream
        const float * __restrict__ q,        // [(t*n_head + h)*128 + d] for this stream
        const float * __restrict__ weights,  // [t*n_head + h] for this stream
        const KT    * __restrict__ k,        // [j*nbk2 + d] for this stream
        const float * __restrict__ mask,     // [t*nbm1 + j] for this stream
        int64_t nbk2, int64_t nbm1,
        int n_tokens, int n_lid, int n_head) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 128u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid    = threadIdx.x;
    const uint32_t warp   = tid >> 5u;

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[128 * 128];
    __shared__ float  c_sh[8 * 16 * 16];

    float acc[8];
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) acc[i] = 0.0f;

    // load this comp-tile's k rows into b_sh (b_sh[d + c*128], col-major for wmma)
    for (uint32_t i = tid; i < 128u * 128u; i += 256u) {
        const uint32_t c = i >> 7u;      // comp within tile
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < (uint32_t) n_lid) v = dsv4_ldk(k + (int64_t) comp * nbk2 + d);
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (int h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 256u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < (uint32_t) n_tokens) {
                v = q[((int64_t) token * n_head + h) * 128 + d];
            }
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + col0 * 128u + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh + warp * 16u * 16u, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        const uint32_t local0 = tid & 255u;
        const uint32_t token0 = tile_t + (local0 >> 4u);
        const float w0 = token0 < (uint32_t) n_tokens ? weights[(int64_t) token0 * n_head + h] : 0.0f;
        uint32_t slot = 0;
        for (uint32_t i = tid; i < 8u * 16u * 16u; i += 256u, slot++) {
            const uint32_t wtile = i >> 8u;
            const uint32_t local = i & 255u;
            const uint32_t r = local >> 4u;
            const uint32_t c = local & 15u;
            const uint32_t token = tile_t + r;
            const uint32_t comp  = tile_c + wtile * 16u + c;
            if (token < (uint32_t) n_tokens && comp < (uint32_t) n_lid) {
                acc[slot] += fmaxf(c_sh[i], 0.0f) * w0;
            }
        }
        __syncthreads();
    }

    uint32_t slot = 0;
    for (uint32_t i = tid; i < 8u * 16u * 16u; i += 256u, slot++) {
        const uint32_t wtile = i >> 8u;
        const uint32_t local = i & 255u;
        const uint32_t r = local >> 4u;
        const uint32_t c = local & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp  = tile_c + wtile * 16u + c;
        if (token < (uint32_t) n_tokens && comp < (uint32_t) n_lid) {
            const float mv = mask[(int64_t) token * nbm1 + comp];
            scores[(int64_t) token * n_lid + comp] = acc[slot] + mv;
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// int8 dp4a score kernel (head_dim == 128) — LLAMA_DSV4_LID_INT8
//
// The wmma fp16 score kernel is L1/shared-memory-bandwidth bound (ncu: 81% mem,
// 80% L1, 14% compute): the tensor cores sit idle while K is re-read from smem
// once per head (K is shared across heads, MLA-style). Storing K int8 in smem
// (16 KB vs 32 KB) halves that L1 traffic and lifts occupancy. dp4a runs on the
// idle int ALU. Per-row symmetric int8 quant of q and k; scales applied after
// the int32 accumulation. One CUDA-stream-group per launch (pointers pre-offset
// like the wmma path). 256 threads, one (16-token x 128-comp) tile per block:
// thread owns 1 token x 8 comps.
// ---------------------------------------------------------------------------

template <typename KT>
static __global__ void dsv4_score_int8_kernel(
        float       * __restrict__ scores,   // [n_tokens, n_lid] for this stream
        const float * __restrict__ q,        // [(t*n_head + h)*128 + d]
        const float * __restrict__ weights,  // [t*n_head + h]
        const KT    * __restrict__ k,        // [j*nbk2 + d]
        const float * __restrict__ mask,     // [t*nbm1 + j]
        int64_t nbk2, int64_t nbm1,
        int n_tokens, int n_lid, int n_head) {
    const int tile_c = blockIdx.x * 128;
    const int tile_t = blockIdx.y * 16;
    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int nwarps = blockDim.x >> 5;

    __shared__ int8_t ks[128 * 128];  // K tile int8 [comp][dim]      16 KB
    __shared__ int8_t qs[16  * 128];  // q tile int8 [token][dim]      2 KB
    __shared__ float  sk[128];        // per-comp  scale
    __shared__ float  sq[16];         // per-token scale

    // --- quantize K tile once (shared across heads) ---
    for (int c = warp; c < 128; c += nwarps) {
        const int comp = tile_c + c;
        const KT * krow = k + (int64_t) comp * nbk2;
        float v[4];
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int d = lane + i * 32;
            v[i] = comp < n_lid ? dsv4_ldk(krow + d) : 0.0f;
            amax = fmaxf(amax, fabsf(v[i]));
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        if (lane == 0) sk[c] = amax > 0.0f ? amax / 127.0f : 0.0f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int d = lane + i * 32;
            ks[c * 128 + d] = (int8_t) __float2int_rn(v[i] * inv);
        }
    }

    const int my_t     = tid & 15;            // 0..15 token within tile
    const int my_c0    = (tid >> 4) * 8;      // 0,8,..,120 comp base
    const int token    = tile_t + my_t;
    const bool tok_ok  = token < n_tokens;
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;

    for (int h = 0; h < n_head; h++) {
        __syncthreads();
        // quantize q[16 tokens] for this head
        for (int tk = warp; tk < 16; tk += nwarps) {
            const int tokn = tile_t + tk;
            const float * qrow = q + ((int64_t) tokn * n_head + h) * 128;
            float v[4];
            float amax = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int d = lane + i * 32;
                v[i] = tokn < n_tokens ? qrow[d] : 0.0f;
                amax = fmaxf(amax, fabsf(v[i]));
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
            if (lane == 0) sq[tk] = amax > 0.0f ? amax / 127.0f : 0.0f;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int d = lane + i * 32;
                qs[tk * 128 + d] = (int8_t) __float2int_rn(v[i] * inv);
            }
        }
        __syncthreads();

        const float w = tok_ok ? weights[(int64_t) token * n_head + h] : 0.0f;
        const float sqt = sq[my_t];
        const int * qp = (const int *) (qs + my_t * 128);
#pragma unroll
        for (int cc = 0; cc < 8; cc++) {
            const int * kp = (const int *) (ks + (my_c0 + cc) * 128);
            int dot = 0;
#pragma unroll
            for (int c = 0; c < 32; c++) dot = __dp4a(qp[c], kp[c], dot);
            const float d = (float) dot * sqt * sk[my_c0 + cc];
            acc[cc] += fmaxf(d, 0.0f) * w;
        }
    }

    if (tok_ok) {
#pragma unroll
        for (int cc = 0; cc < 8; cc++) {
            const int comp = tile_c + my_c0 + cc;
            if (comp < n_lid) {
                scores[(int64_t) token * n_lid + comp] = acc[cc] + mask[(int64_t) token * nbm1 + comp];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// dedicated decode score kernel (nt==1, head_dim==128) — LLAMA_DSV4_LID_DEC
//
// At decode the wmma/int8 path pads 1 token to a 16-token tile (15/16 wasted)
// and the scalar path launches too few blocks. This kernel is warp-per-comp:
// q (all heads) is quantized int8 once into smem and reused; each warp streams
// one comp's K (coalesced, 4 dims/lane), int8-quantizes it, and reduces the
// 128-dim dot per head via warp shuffle. Grid-strided over comps for high
// block count. One stream-group per launch (pointers pre-offset).
// ---------------------------------------------------------------------------

template <typename KT>
static __global__ void dsv4_score_decode_kernel(
        float       * __restrict__ scores,   // [n_lid]
        const float * __restrict__ q,        // [h*128 + d]  (single token)
        const float * __restrict__ weights,  // [h]
        const KT    * __restrict__ k,        // [j*nbk2 + d]
        const float * __restrict__ mask,     // [j]
        int64_t nbk2, int n_lid, int n_head) {
    extern __shared__ char smem_dec[];
    int8_t * qs   = (int8_t *) smem_dec;                 // [n_head*128]
    float  * sq   = (float *) (qs + (size_t) n_head * 128); // [n_head]
    float  * sw   = sq + n_head;                          // [n_head]

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int nwarps = blockDim.x >> 5;

    // quantize q (all heads) into smem once
    for (int h = warp; h < n_head; h += nwarps) {
        const float * qh = q + (int64_t) h * 128;
        float v[4];
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; i++) { v[i] = qh[lane * 4 + i]; amax = fmaxf(amax, fabsf(v[i])); }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        if (lane == 0) { sq[h] = amax > 0.0f ? amax / 127.0f : 0.0f; sw[h] = weights[h]; }
#pragma unroll
        for (int i = 0; i < 4; i++) qs[h * 128 + lane * 4 + i] = (int8_t) __float2int_rn(v[i] * inv);
    }
    __syncthreads();

    // grid-stride over comps, one warp per comp
    for (int j = blockIdx.x * nwarps + warp; j < n_lid; j += gridDim.x * nwarps) {
        const KT * kj = k + (int64_t) j * nbk2;
        int8_t kq[4];
        float amax = 0.0f;
        float kv[4];
#pragma unroll
        for (int i = 0; i < 4; i++) { kv[i] = dsv4_ldk(kj + lane * 4 + i); amax = fmaxf(amax, fabsf(kv[i])); }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float ksc = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float kinv = amax > 0.0f ? 127.0f / amax : 0.0f;
#pragma unroll
        for (int i = 0; i < 4; i++) kq[i] = (int8_t) __float2int_rn(kv[i] * kinv);
        const int kpack = *(const int *) kq;

        float acc = 0.0f;
        for (int h = 0; h < n_head; h++) {
            const int qpack = *(const int *) (qs + h * 128 + lane * 4);
            int dot = __dp4a(qpack, kpack, 0);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
            const float d = (float) dot * sq[h] * ksc;
            acc += fmaxf(d, 0.0f) * sw[h];
        }
        if (lane == 0) scores[j] = acc + mask[j];
    }
}

// ---------------------------------------------------------------------------
// B2 union + membership (one block per stream, smem bitmap over n_csa)
// ---------------------------------------------------------------------------

static __global__ void dsv4_union_kernel(
        int32_t * __restrict__ out, const int32_t * __restrict__ top_k,
        int64_t nb1_tk, int64_t nb3_tk, int64_t nb3_out,
        int n_top_k, int nt_s, int n_csa, int u_max) {
    const int s = blockIdx.x;
    extern __shared__ uint32_t bm[];
    const int n_words = (n_csa + 31) / 32;
    for (int i = threadIdx.x; i < n_words; i += blockDim.x) bm[i] = 0;
    __syncthreads();
    const int total = n_top_k * nt_s;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int t = idx / n_top_k, i = idx % n_top_k;
        const int c = top_k[(int64_t) i + t * nb1_tk + (int64_t) s * nb3_tk];
        if (c >= 0 && c < n_csa) atomicOr(&bm[c >> 5], 1u << (c & 31));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int32_t * o = out + (int64_t) s * nb3_out;
        int pos = 0;
        for (int w = 0; w < n_words && pos < u_max; w++) {
            uint32_t word = bm[w];
            while (word && pos < u_max) {
                const int b = __ffs(word) - 1;
                o[pos++] = w * 32 + b;
                word &= word - 1;
            }
        }
        for (; pos < u_max; pos++) o[pos] = n_csa - 1;
    }
}

// One block per (token-tile, stream). Uses union_idx directly (binary search)
// so the membership rank is GUARANTEED consistent with the gather order.
#define DSV4_MEMB_TPB 8
static __global__ void dsv4_memb_kernel(
        float * __restrict__ memb, const int32_t * __restrict__ top_k,
        const int32_t * __restrict__ union_idx,
        int64_t nb1_tk, int64_t nb3_tk, int64_t nb1_m, int64_t nb3_m, int64_t nb3_u,
        int n_top_k, int nt_s, int n_csa, int u_max) {
    const int s      = blockIdx.y;
    const int t_base = blockIdx.x * DSV4_MEMB_TPB;
    extern __shared__ int32_t uni_s[]; // [u_max] union_idx for this stream
    for (int i = threadIdx.x; i < u_max; i += blockDim.x) uni_s[i] = union_idx[(int64_t) i + (int64_t) s * nb3_u];
    __syncthreads();

    for (int lt = 0; lt < DSV4_MEMB_TPB; lt++) {
        const int t = t_base + lt;
        if (t >= nt_s) break;
        float * mrow = memb + (int64_t) t * nb1_m + (int64_t) s * nb3_m;
        for (int u = threadIdx.x; u < u_max; u += blockDim.x) mrow[u] = -INFINITY;
        __syncthreads();
        const int32_t * tk = top_k + (int64_t) t * nb1_tk + (int64_t) s * nb3_tk;
        for (int i = threadIdx.x; i < n_top_k; i += blockDim.x) {
            const int c = tk[i];
            if (c < 0 || c >= n_csa) continue;
            // lower_bound(uni_s, c)
            int lo = 0, hi = u_max;
            while (lo < hi) { const int mid = (lo + hi) >> 1; if (uni_s[mid] < c) lo = mid + 1; else hi = mid; }
            if (lo < u_max && uni_s[lo] == c) mrow[lo] = 0.0f;
        }
        __syncthreads();
    }
}

void ggml_cuda_op_dsv4_lid_union(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * top_k = dst->src[0];
    GGML_ASSERT(top_k->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32);
    const int n_csa   = ggml_get_op_params_i32(dst, 0);
    const int u_max   = dst->ne[0];
    const int n_top_k = top_k->ne[0];
    const int nt_s    = top_k->ne[1];
    const int n_stream = top_k->ne[3];
    const int64_t nb1_tk  = top_k->nb[1] / sizeof(int32_t);
    const int64_t nb3_tk  = top_k->nb[3] / sizeof(int32_t);
    const int64_t nb3_out = dst->nb[3] / sizeof(int32_t);
    const size_t smem = (size_t) ((n_csa + 31) / 32) * sizeof(uint32_t);
    dsv4_union_kernel<<<n_stream, 256, smem, ctx.stream()>>>(
        (int32_t *) dst->data, (const int32_t *) top_k->data,
        nb1_tk, nb3_tk, nb3_out, n_top_k, nt_s, n_csa, u_max);
}

void ggml_cuda_op_dsv4_lid_memb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * top_k = dst->src[0];
    const ggml_tensor * uni   = dst->src[1];
    GGML_ASSERT(top_k->type == GGML_TYPE_I32 && uni->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    const int n_csa   = ggml_get_op_params_i32(dst, 0);
    const int u_max   = dst->ne[0];
    const int nt_s    = dst->ne[1];
    const int n_stream = dst->ne[3];
    const int n_top_k = top_k->ne[0];
    const int64_t nb1_tk = top_k->nb[1] / sizeof(int32_t);
    const int64_t nb3_tk = top_k->nb[3] / sizeof(int32_t);
    const int64_t nb3_u  = uni->nb[3] / sizeof(int32_t);
    const int64_t nb1_m  = dst->nb[1] / sizeof(float);
    const int64_t nb3_m  = dst->nb[3] / sizeof(float);
    const dim3 grid((nt_s + DSV4_MEMB_TPB - 1) / DSV4_MEMB_TPB, n_stream, 1);
    const size_t smem = (size_t) u_max * sizeof(int32_t);
    dsv4_memb_kernel<<<grid, 256, smem, ctx.stream()>>>(
        (float *) dst->data, (const int32_t *) top_k->data, (const int32_t *) uni->data,
        nb1_tk, nb3_tk, nb1_m, nb3_m, nb3_u, n_top_k, nt_s, n_csa, u_max);
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

    // Env-gated one-shot input dump for the fp4 selection-fidelity oracle
    // (experiments/ds4-tile/fp4_oracle.cpp). Dumps the first prefill call
    // whose n_lid reaches the threshold: q/w for the first <=256 tokens
    // (f32) and the full k view (converted to f32). Layouts match the
    // oracle's --pre-rotated dump mode.
    //   LLAMA_DSV4_LID_DUMP=<dir>  LLAMA_DSV4_LID_DUMP_NLID=<min n_lid>
    static const char * dump_dir = getenv("LLAMA_DSV4_LID_DUMP");
    static const int dump_nlid = getenv("LLAMA_DSV4_LID_DUMP_NLID")
        ? atoi(getenv("LLAMA_DSV4_LID_DUMP_NLID")) : 0;
    static bool dumped = false;
    if (dump_dir && !dumped && nt > 1 && n_lid >= dump_nlid) {
        dumped = true;
        const int nt_dump = nt < 256 ? nt : 256;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> hq((size_t) nt_dump * n_head * d_idx);
        std::vector<float> hw((size_t) nt_dump * n_head);
        std::vector<float> hk((size_t) n_lid * d_idx);
        CUDA_CHECK(cudaMemcpy(hq.data(), q->data, hq.size()*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hw.data(), weights->data, hw.size()*sizeof(float), cudaMemcpyDeviceToHost));
        const int64_t nbk2_e = k->nb[2] / ggml_type_size(k->type);
        if (k->type == GGML_TYPE_F16) {
            std::vector<half> hk16((size_t) n_lid * d_idx);
            for (int j = 0; j < n_lid; j++) {
                CUDA_CHECK(cudaMemcpy(hk16.data() + (size_t) j * d_idx,
                        (const half *) k->data + (int64_t) j * nbk2_e,
                        d_idx*sizeof(half), cudaMemcpyDeviceToHost));
            }
            for (size_t i = 0; i < hk.size(); i++) hk[i] = __half2float(hk16[i]);
        } else {
            for (int j = 0; j < n_lid; j++) {
                CUDA_CHECK(cudaMemcpy(hk.data() + (size_t) j * d_idx,
                        (const float *) k->data + (int64_t) j * nbk2_e,
                        d_idx*sizeof(float), cudaMemcpyDeviceToHost));
            }
        }
        auto wr = [&](const char * name, const void * p, size_t bytes) {
            std::string path = std::string(dump_dir) + "/" + name;
            FILE * f = fopen(path.c_str(), "wb");
            GGML_ASSERT(f && fwrite(p, 1, bytes, f) == bytes);
            fclose(f);
        };
        wr("q.bin", hq.data(), hq.size()*sizeof(float));
        wr("w.bin", hw.data(), hw.size()*sizeof(float));
        wr("k.bin", hk.data(), hk.size()*sizeof(float));
        fprintf(stderr, "\ndsv4_lid_topk: dumped nt=%d n_head=%d n_lid=%d d_idx=%d to %s\n",
                nt_dump, n_head, n_lid, d_idx, dump_dir);
    }

    // scores buffer lives in the pool, not the graph compute buffer.
    ggml_cuda_pool_alloc<float> scores_alloc(pool, (size_t) nt * n_lid);
    float * scores = scores_alloc.get();

    int64_t nbk2 = k->nb[2] / ggml_type_size(k->type);
    int64_t nbk3 = k->nb[3] / ggml_type_size(k->type);
    const int64_t nbm1 = mask->nb[1] / sizeof(float);
    const int64_t nbm3 = mask->nb[3] / sizeof(float);

    const float * q_d = (const float *) q->data;
    const float * w_d = (const float *) weights->data;
    const float * m_d = (const float *) mask->data;

    // fp4 QAT path (LLAMA_DSV4_LID_FP4): fake-quant q and k to e2m1 before
    // scoring. Both become contiguous f32; k is copied out of its strided cache
    // view into a dense [d_idx, n_lid, n_stream] buffer, so downstream always
    // takes the float dispatch. Numerics-only (no speedup) — validates the
    // device e2m1 path against the CPU reference and the resident model.
    static const bool dsv4_lid_fp4 = []() {
        const char * e = getenv("LLAMA_DSV4_LID_FP4");
        return e && e[0] == '1';
    }();
    ggml_cuda_pool_alloc<float> q_fp4_alloc(pool);
    ggml_cuda_pool_alloc<float> k_fp4_alloc(pool);
    bool k_force_f32 = false;
    GGML_ASSERT(!dsv4_lid_fp4 || d_idx % 32 == 0);
    if (dsv4_lid_fp4) {
        float * q_fp4 = q_fp4_alloc.alloc((size_t) nt * n_head * d_idx);
        float * k_fp4 = k_fp4_alloc.alloc((size_t) n_stream * n_lid * d_idx);
        // q: contiguous [d_idx, n_head, nt] -> rows_per_group = n_head*nt,
        // one flat group, row_stride0 = d_idx.
        dsv4_fp4_quant_kernel<float><<<n_head * nt, d_idx, 0, stream>>>(
                q_fp4, q_d, d_idx, 0, n_head * nt, d_idx);
        // k: strided view, group = stream (stride nbk3), row = j (stride nbk2).
        if (k->type == GGML_TYPE_F16) {
            dsv4_fp4_quant_kernel<half><<<n_stream * n_lid, d_idx, 0, stream>>>(
                    k_fp4, (const half *) k->data, nbk2, nbk3, n_lid, d_idx);
        } else {
            dsv4_fp4_quant_kernel<float><<<n_stream * n_lid, d_idx, 0, stream>>>(
                    k_fp4, (const float *) k->data, nbk2, nbk3, n_lid, d_idx);
        }
        q_d = q_fp4;
        k_force_f32 = true;
        nbk2 = d_idx;
        nbk3 = (int64_t) d_idx * n_lid;
    }
    const float * k_f32_d = k_fp4_alloc.get();
    const bool k_is_f16 = (k->type == GGML_TYPE_F16) && !k_force_f32;

    // int8 dp4a path (LLAMA_DSV4_LID_INT8): halves K smem width to attack the
    // L1-bandwidth bound of the wmma path. Same per-stream launch geometry.
    static const bool dsv4_lid_int8 = []() {
        const char * e = getenv("LLAMA_DSV4_LID_INT8");
        return e && e[0] == '1';
    }();
    // Dedicated decode kernel (nt_s==1): warp-per-comp, no 16-token tile padding.
    static const bool dsv4_lid_dec = []() {
        const char * e = getenv("LLAMA_DSV4_LID_DEC");
        return e && e[0] == '1';
    }();

    // Tensor-core path: requires head_dim == 128 (the wmma 16x16x16 k-tiling).
    // One launch per CUDA stream-group so a 16-token tile never straddles a
    // stream boundary; falls back to the scalar dot-product kernel otherwise.
    if (d_idx == 128 && dsv4_lid_dec && nt_s == 1) {
        const int block   = 256;
        const int nwarps  = block / 32;
        const size_t smem = (size_t) n_head * 128 + (size_t) n_head * 2 * sizeof(float);
        int gx = (n_lid + nwarps - 1) / nwarps;
        if (gx > 512) gx = 512;
        for (int s = 0; s < n_stream; s++) {
            float       * sc_s = scores + (int64_t) s * nt_s * n_lid;
            const float * q_s  = q_d + (int64_t) s * nt_s * n_head * d_idx;
            const float * w_s  = w_d + (int64_t) s * nt_s * n_head;
            const float * m_s  = m_d + (int64_t) s * nbm3;
            if (k_is_f16) {
                dsv4_score_decode_kernel<half><<<gx, block, smem, stream>>>(
                        sc_s, q_s, w_s, (const half *) k->data + (int64_t) s * nbk3, m_s, nbk2, n_lid, n_head);
            } else {
                dsv4_score_decode_kernel<float><<<gx, block, smem, stream>>>(
                        sc_s, q_s, w_s, (k_force_f32 ? k_f32_d : (const float *) k->data) + (int64_t) s * nbk3, m_s, nbk2, n_lid, n_head);
            }
        }
    } else if (d_idx == 128) {
        const dim3 block(256);
        const dim3 grid((n_lid + 127) / 128, (nt_s + 15) / 16, 1);
        for (int s = 0; s < n_stream; s++) {
            const int64_t t0    = (int64_t) s * nt_s;
            float       * sc_s  = scores + t0 * n_lid;
            const float * q_s   = q_d + t0 * n_head * d_idx;
            const float * w_s   = w_d + t0 * n_head;
            const float * m_s   = m_d + (int64_t) s * nbm3;
            if (k_is_f16) {
                const half * k_s = (const half *) k->data + (int64_t) s * nbk3;
                if (dsv4_lid_int8) {
                    dsv4_score_int8_kernel<half><<<grid, block, 0, stream>>>(
                            sc_s, q_s, w_s, k_s, m_s, nbk2, nbm1, nt_s, n_lid, n_head);
                } else {
                    dsv4_score_wmma128_kernel<half><<<grid, block, 0, stream>>>(
                            sc_s, q_s, w_s, k_s, m_s, nbk2, nbm1, nt_s, n_lid, n_head);
                }
            } else {
                const float * k_s = (k_force_f32 ? k_f32_d : (const float *) k->data) + (int64_t) s * nbk3;
                if (dsv4_lid_int8) {
                    dsv4_score_int8_kernel<float><<<grid, block, 0, stream>>>(
                            sc_s, q_s, w_s, k_s, m_s, nbk2, nbm1, nt_s, n_lid, n_head);
                } else {
                    dsv4_score_wmma128_kernel<float><<<grid, block, 0, stream>>>(
                            sc_s, q_s, w_s, k_s, m_s, nbk2, nbm1, nt_s, n_lid, n_head);
                }
            }
        }
    } else {
        const int j_tile  = 512;
        const int block   = 256;
        const size_t smem = ((size_t) d_idx*n_head + n_head) * sizeof(float);
        dim3 grid_score(nt, (n_lid + j_tile - 1) / j_tile, 1);
        if (k_is_f16) {
            dsv4_score_kernel<half><<<grid_score, block, smem, stream>>>(
                    scores, q_d, w_d, (const half *) k->data, m_d,
                    nbk2, nbk3, nbm1, nbm3, nt, nt_s, n_lid, d_idx, n_head, j_tile);
        } else {
            dsv4_score_kernel<float><<<grid_score, block, smem, stream>>>(
                    scores, q_d, w_d, (k_force_f32 ? k_f32_d : (const float *) k->data), m_d,
                    nbk2, nbk3, nbm1, nbm3, nt, nt_s, n_lid, d_idx, n_head, j_tile);
        }
    }

    // output is contiguous [n_top_k, nt_s, 1, n_stream] == flat [nt * n_top_k]
    dsv4_topk_launch(pool, (uint32_t *) dst->data, scores, n_lid, nt, n_top_k, stream);
}
