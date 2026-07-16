#include "dsv4_lid_topk.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <mma.h>

#include "mma.cuh" // ggml_cuda_mma block-scaled fp4 tensor-core (step 3 probe)

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

// scores-buffer load: the d_idx==128 score kernels store f16 (halves the
// [nt, n_lid] DRAM round-trip); the scalar path keeps f32 to stay bit-exact
// under its strict 0.0-tolerance gate. Sort compares stay f32 either way.
static __device__ __forceinline__ float dsv4_lds(const float * p) { return *p; }
static __device__ __forceinline__ float dsv4_lds(const half  * p) { return __half2float(*p); }

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

// ---------------------------------------------------------------------------
// P3b packed MXFP4 lid container: 17B block-32 (e8m0 scale byte + 16 nibble
// bytes, ggml block_mxfp4 element order: low nibble -> j, high -> j+16).
// Rounding is the DSV4 QAT search above, NOT ggml's stock quantizer. With
// level values stored at TRUE e2m1 magnitudes, e = s + 127 makes the dequant
// scale 2^(e-127) == the QAT scale, so dequant(pack(x)) == dsv4_fp4_rt(x)
// exactly (pure power-of-two products).
// ---------------------------------------------------------------------------

// nibble -> TRUE e2m1 level (index 8 = -0 for sign fidelity with the
// fake-quant path); dequant value = DSV4_LV16[ni] * 2^(e-127)
static __constant__ float DSV4_LV16[16] = {  0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };

// dequant element d (0..ne0) of a packed 17B-block row at byte pointer kb
static __device__ __forceinline__ float dsv4_mxfp4_get(const uint8_t * kb, int d) {
    const uint8_t * blk = kb + (d >> 5) * 17;
    const int p = d & 31;
    const uint8_t byte = blk[1 + (p & 15)];
    const int ni = p < 16 ? (byte & 0x0F) : (byte >> 4);
    return DSV4_LV16[ni] * exp2f((float) ((int) blk[0] - 127));
}

static __device__ __forceinline__ int dsv4_e2m1_index(float x) {
    const int sgn = x < 0.0f ? 8 : 0;
    const float ax = fminf(fabsf(x), 6.0f);
    const float lv[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    int   best = 0;
    float bd   = fabsf(ax - lv[0]);
#pragma unroll
    for (int i = 1; i < 8; i++) {
        const float d = fabsf(ax - lv[i]);
        if (d < bd || (d == bd && (i & 1) == 0 && (best & 1) != 0)) { best = i; bd = d; }
    }
    return best | sgn;
}

// One block per source row (blockDim.x == ne0, multiple of 32); QAT-quantizes
// and packs the row into the MXFP4 container at row idxs[blockIdx.x].
static __global__ void dsv4_qat_set_rows_kernel(
        uint8_t * __restrict__ dst, const float * __restrict__ src,
        const void * __restrict__ idxs, const int idx_i64,
        int64_t nb1_dst, int64_t nb1_src, int ne0) {
    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int64_t idx = idx_i64 ? ((const int64_t *) idxs)[row]
                                : (int64_t) ((const int32_t *) idxs)[row];
    const float v = src[(int64_t) row * nb1_src + tid];
    float a = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    a = fmaxf(a, 7.052966104933725e-38f);
    const int   sc    = (int) ceilf(log2f(a / 6.0f));
    const float scale = exp2f((float) sc);
    const int ni = dsv4_e2m1_index(fminf(6.0f, fmaxf(-6.0f, v / scale)));
    const int hi = __shfl_down_sync(0xffffffffu, ni, 16);
    uint8_t * blk = dst + idx*nb1_dst + (int64_t) (tid >> 5) * 17;
    if (lane == 0)  blk[0] = (uint8_t) (sc + 127);
    if (lane < 16)  blk[1 + lane] = (uint8_t) (ni | (hi << 4));
}

// Staging dequant: packed [d_idx, n_lid, n_stream] cache view -> dense f32
// buffer, one block per row (blockDim.x == d_idx). Strides in BYTES (block
// rows; element strides are meaningless for packed types).
// OT = half for the prefill staging buffer (f16-of-QAT is bit-exact — 2-bit
// mantissa x pow2 scale — and halves staged-K read bandwidth vs f32).
template <typename OT>
static __global__ void dsv4_mxfp4_dequant_rows_kernel(
        OT * __restrict__ out, const uint8_t * __restrict__ in,
        int64_t nb2, int64_t nb3, int rows_per_group, int ne0) {
    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int64_t g = row / rows_per_group;
    const int64_t r = row % rows_per_group;
    const uint8_t * blk = in + g*nb3 + r*nb2 + (int64_t) (tid >> 5) * 17;
    const float d = exp2f((float) ((int) blk[0] - 127));
    const uint8_t byte = blk[1 + (lane & 15)];
    const int ni = lane < 16 ? (byte & 0x0F) : (byte >> 4);
    out[(int64_t) row * ne0 + tid] = (OT) (DSV4_LV16[ni] * d);
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
        half        * __restrict__ scores,   // [nt_s, n_lid] for this stream (f16 store)
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
            scores[(int64_t) token * n_lid + comp] = __float2half(acc[slot] + mv);
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

// Global int8 pre-quant of lid-K: one warp per comp (K-row). Layout and
// arithmetic identical to the quant the score kernel used to run per
// (16-token x 128-comp) block — strided dims d = lane + i*32, warp shfl_xor
// amax, round-to-nearest-even, per-comp scale amax/127 — so consuming the
// buffer is bit-identical to re-quantizing, minus the ceil(nt_s/16)-fold
// redundant work and the wider f16/f32 global reads.
template <typename KT>
static __global__ void dsv4_prequant_k_int8_kernel(
        int8_t      * __restrict__ k_i8,   // [comp*128 + d] int8, dim-contiguous
        float       * __restrict__ k_sc,   // [comp] per-comp scale (amax/127)
        const KT    * __restrict__ k,      // [comp*nbk2 + d]
        int64_t nbk2, int n_lid) {
    const int comp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (comp >= n_lid) return;
    const KT * krow = k + (int64_t) comp * nbk2;
    float v[4];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int d = lane + i * 32;
        v[i] = dsv4_ldk(krow + d);
        amax = fmaxf(amax, fabsf(v[i]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    if (lane == 0) k_sc[comp] = amax > 0.0f ? amax / 127.0f : 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int d = lane + i * 32;
        k_i8[(int64_t) comp * 128 + d] = (int8_t) __float2int_rn(v[i] * inv);
    }
}

static __global__ void dsv4_score_int8_kernel(
        half         * __restrict__ scores,   // [n_tokens, n_lid] for this stream (f16 store)
        const float  * __restrict__ q,        // [(t*n_head + h)*128 + d]
        const float  * __restrict__ weights,  // [t*n_head + h]
        const int8_t * __restrict__ k_i8,     // [j*128 + d] pre-quantized (this stream)
        const float  * __restrict__ k_sc,     // [j] per-comp scale
        const float  * __restrict__ mask,     // [t*nbm1 + j]
        int64_t nbm1,
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

    // --- load pre-quantized K tile into smem (shared across heads) ---
    for (int c = warp; c < 128; c += nwarps) {
        const int comp = tile_c + c;
        const bool ok  = comp < n_lid;
        if (lane == 0) sk[c] = ok ? k_sc[comp] : 0.0f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int d = lane + i * 32;
            ks[c * 128 + d] = ok ? k_i8[(int64_t) comp * 128 + d] : (int8_t) 0;
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
                scores[(int64_t) token * n_lid + comp] = __float2half(acc[cc] + mask[(int64_t) token * nbm1 + comp]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// fp4-mma score kernel (head_dim==128, packed MXFP4 lid container) —
// LLAMA_DSV4_LID_FP4_MMA probe (Blackwell only, step 3).
//
// Register-resident-K block-scaled fp4 tensor-core scoring. The MLA indexer
// shares ONE K across all 64 heads, so K (the mma B operand) is loaded into
// registers ONCE per 16-token x 128-comp block and reused across the head
// loop — attacking the int8 kernel's L1 bound (it re-reads K from smem 64x).
// K is read from the 68B block_mxfp4 container rows straight onto the e2m1
// grid (transform-free: nibbles ARE hardware e2m1, scale byte is raw ue8m0);
// q is packed to e2m1 per head at runtime. Numerics = LID_FP4 class; under
// LID_EXACT the packed-direct rescore backstop makes selection bit-exact
// (oracle: pass-1 displacement p100<=4 vs the m=64 window).
//
// smem tile format mirrors mmq's MXFP4 path (16 qs ints + 2 packed ue8m0
// scale u32 per row, row stride padded to 16B) so load_ldmatrix / load_generic
// and the block-scale lane maps are correct by construction.
// ---------------------------------------------------------------------------

#ifdef GGML_USE_HIP
static __global__ void dsv4_score_fp4mma_kernel(
        half *, const float *, const float *, const uint8_t *, const float *,
        int64_t, int64_t, int, int, int) { NO_DEVICE_CODE; }
#else
using namespace ggml_cuda_mma;

// per-row smem stride in ints: 16 qs (128 e2m1 nibbles) + 2 scale u32,
// padded to a multiple of 4 so every row base and k64-frag base is 16B-aligned
#define DSV4_FP4_ROW 20
#define DSV4_FP4_QSC 16  // scale u32 offset within the row

static __global__ void dsv4_score_fp4mma_kernel(
        half         * __restrict__ scores,   // [n_tokens, n_lid] for this stream
        const float  * __restrict__ q,        // [(t*n_head + h)*128 + d] f32
        const float  * __restrict__ weights,  // [t*n_head + h]
        const uint8_t * __restrict__ k,       // packed block_mxfp4 rows, BYTE strides
        const float  * __restrict__ mask,     // [t*nbm1 + j]
        int64_t nbk2, int64_t nbm1,
        int n_tokens, int n_lid, int n_head) {
#ifdef BLACKWELL_MMA_AVAILABLE
    typedef tile<16, 8, int>   tile_A; // q: 16 tokens x k64
    typedef tile<8,  8, int>   tile_B; // K:  8 comps  x k64
    typedef tile<16, 8, float> tile_C; // out: 16 tokens x 8 comps

    // launched as blockDim (32, nwarps): threadIdx.x = lane (the mma/ldmatrix
    // primitives use it directly as the warp lane), threadIdx.y = warp.
    const int tile_c = blockIdx.x * 128; // comp base for this block
    const int tile_t = blockIdx.y * 16;  // token base
    const int lane   = threadIdx.x;       // 0..31
    const int warp   = threadIdx.y;       // 0..nwarps-1, owns comps [warp*16, +16)
    const int nwarps = blockDim.y;

    // scale-supply lanes (PTX warp block-scaling: 2 threads/quad), from mmq
    const int tidx_A = lane / 4 + (lane % 2) * 8;
    const int tidx_B = lane / 4;

    __shared__ int q_sm[16 * DSV4_FP4_ROW];   // A tile: 16 tokens
    __shared__ int k_sm[128 * DSV4_FP4_ROW];  // B tiles: 128 comps

    // ---- pack this block's 128 K rows to smem ONCE (resident across heads) --
    // one warp per 16 comps; lane owns 8 comps? No: cooperative over 128 rows.
    for (int c = warp; c < 128; c += nwarps) {
        const int comp = tile_c + c;
        int * row_qs = k_sm + c * DSV4_FP4_ROW;
        if (comp < n_lid) {
            const uint8_t * krow = k + (int64_t) comp * nbk2; // 4 x 17B blocks
            // lane l (0..15) copies block l>>2 's 4 qs bytes for k64-frag; use
            // 16 lanes to memcpy the 4x16 qs bytes and set scales
            for (int b = lane; b < 4; b += 32) {
                const uint8_t * blk = krow + b * 17;
                // 16 qs bytes -> 4 ints at row_qs[b*4 .. b*4+4)
                int tmp[4];
                #pragma unroll
                for (int w = 0; w < 4; w++) {
                    tmp[w] = (int) blk[1 + w*4 + 0]       | ((int) blk[1 + w*4 + 1] << 8)
                           | ((int) blk[1 + w*4 + 2] << 16) | ((int) blk[1 + w*4 + 3] << 24);
                }
                #pragma unroll
                for (int w = 0; w < 4; w++) row_qs[b*4 + w] = tmp[w];
            }
            // pack the 4 e8m0 scale bytes into 2 u32 (2 sub-blocks per k64 frag)
            if (lane == 0) {
                const uint32_t e0 = krow[0*17], e1 = krow[1*17], e2 = krow[2*17], e3 = krow[3*17];
                row_qs[DSV4_FP4_QSC + 0] = (int) (e0 | (e1 << 8));
                row_qs[DSV4_FP4_QSC + 1] = (int) (e2 | (e3 << 8));
            }
        } else if (lane == 0) {
            #pragma unroll
            for (int w = 0; w < 16; w++) row_qs[w] = 0;
            row_qs[DSV4_FP4_QSC + 0] = 0;
            row_qs[DSV4_FP4_QSC + 1] = 0;
        }
    }
    __syncthreads();

    // resident B (K) fragments for this warp's 2 comp-tiles x 2 k64 frags
    tile_B    Bk[2][2];
    uint32_t  scB[2][2];
    #pragma unroll
    for (int ct = 0; ct < 2; ct++) {
        const int crow = warp * 16 + ct * 8; // comp-tile base within the block
        #pragma unroll
        for (int f = 0; f < 2; f++) {
            load_generic(Bk[ct][f], k_sm + crow * DSV4_FP4_ROW + f * 8, DSV4_FP4_ROW);
            scB[ct][f] = ((const uint32_t *) (k_sm + (crow + tidx_B) * DSV4_FP4_ROW + DSV4_FP4_QSC))[f];
        }
    }

    float acc[2][4];
    #pragma unroll
    for (int ct = 0; ct < 2; ct++)
        #pragma unroll
        for (int l = 0; l < 4; l++) acc[ct][l] = 0.0f;

    // ---- head loop: pack q per head, ldmatrix A, 2 mma/comp-tile, relu drain -
    for (int h = 0; h < n_head; h++) {
        __syncthreads();
        // pack q[16 tokens][128 dims] for this head into the A smem tile, in the
        // SAME block_mxfp4 layout K uses (byte m of a 32-dim block holds dim m
        // low-nibble, dim m+16 high-nibble; int w of block b = bytes [w*4..+4)).
        // One warp per token; lane owns 4 consecutive dims [lane*4, +4). Block
        // b = lane/8; sub-lane s = lane%8. Sub-lanes s<4 hold the LOW dims of
        // int w=s, sub-lanes s+4 hold the matching HIGH dims — combine via shfl.
        for (int tk = warp; tk < 16; tk += nwarps) {
            const int tokn = tile_t + tk;
            int * row_qs = q_sm + tk * DSV4_FP4_ROW;
            const float * qrow = (tokn < n_tokens)
                    ? q + ((int64_t) tokn * n_head + h) * 128 : nullptr;
            const int b = lane >> 3;   // 0..3 which 32-dim block
            const int s = lane & 7;    // 0..7 sub-lane within block
            float v[4];
            float amax = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                v[i] = qrow ? qrow[lane * 4 + i] : 0.0f;
                amax = fmaxf(amax, fabsf(v[i]));
            }
            // amax over each aligned 8-lane octet (= one 32-dim block)
            #pragma unroll
            for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            amax = fmaxf(amax, 7.052966104933725e-38f);
            const int   sc    = (int) ceilf(log2f(amax / 6.0f));
            const float scale = exp2f((float) sc);
            // this lane's 4 e2m1 nibbles, packed one-per-byte (low positions)
            int mynib = 0;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int ni = dsv4_e2m1_index(fminf(6.0f, fmaxf(-6.0f, v[i] / scale)));
                mynib |= (ni & 0xF) << (i * 8);
            }
            // sub-lane s<4 owns int w=s: its own nibbles are the low halves,
            // the high halves come from sub-lane s+4 of the same block
            const int hinib = __shfl_sync(0xffffffffu, mynib, (b << 3) + s + 4);
            if (s < 4) {
                row_qs[b * 4 + s] = mynib | (hinib << 4);
            }
            // block scale (biased ue8m0). amax is uniform across a block's 8
            // lanes, so lane b*8 holds block b's scale; lane 0 gathers all four
            // and writes the two u32 exactly as K packs them (e0|e1<<8, e2|e3<<8,
            // high bytes zero).
            const int e8 = sc + 127;
            const int e0 = __shfl_sync(0xffffffffu, e8, 0);
            const int e1 = __shfl_sync(0xffffffffu, e8, 8);
            const int e2 = __shfl_sync(0xffffffffu, e8, 16);
            const int e3 = __shfl_sync(0xffffffffu, e8, 24);
            if (lane == 0) {
                row_qs[DSV4_FP4_QSC + 0] = e0 | (e1 << 8);
                row_qs[DSV4_FP4_QSC + 1] = e2 | (e3 << 8);
            }
        }
        __syncthreads();

        tile_A Aq[2];
        uint32_t scA[2];
        #pragma unroll
        for (int f = 0; f < 2; f++) {
            load_ldmatrix(Aq[f], q_sm + f * 8, DSV4_FP4_ROW);
            scA[f] = ((const uint32_t *) (q_sm + tidx_A * DSV4_FP4_ROW + DSV4_FP4_QSC))[f];
        }

        #pragma unroll
        for (int ct = 0; ct < 2; ct++) {
            tile_C C = {};
            mma_block_scaled_fp4<GGML_TYPE_MXFP4>(C, Aq[0], Bk[ct][0], scA[0], scB[ct][0]);
            mma_block_scaled_fp4<GGML_TYPE_MXFP4>(C, Aq[1], Bk[ct][1], scA[1], scB[ct][1]);
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                const int tok = tile_t + ((l / 2) * 8) + (lane / 4);
                const float wl = tok < n_tokens ? weights[(int64_t) tok * n_head + h] : 0.0f;
                acc[ct][l] += fmaxf(C.x[l], 0.0f) * wl;
            }
        }
    }

    // ---- epilogue: + mask, f16 store ---------------------------------------
    #pragma unroll
    for (int ct = 0; ct < 2; ct++) {
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            const int tok  = tile_t + ((l / 2) * 8) + (lane / 4);
            const int comp = tile_c + warp * 16 + ct * 8 + ((lane % 4) * 2) + (l % 2);
            if (tok < n_tokens && comp < n_lid) {
                scores[(int64_t) tok * n_lid + comp] =
                        __float2half(acc[ct][l] + mask[(int64_t) tok * nbm1 + comp]);
            }
        }
    }
#else
    GGML_UNUSED_VARS(scores, q, weights, k, mask, nbk2, nbm1, n_tokens, n_lid, n_head);
    NO_DEVICE_CODE;
#endif // BLACKWELL_MMA_AVAILABLE
}
#endif // GGML_USE_HIP

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

// PACKED: k points at 17B-block MXFP4 rows and nbk2 is a BYTE stride — the
// warp reads 68B/row instead of 256B f16 (the P3b-ii decode bandwidth win).
template <typename KT, bool PACKED = false>
static __global__ void dsv4_score_decode_kernel(
        half        * __restrict__ scores,   // [n_lid] (f16 store)
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
        int8_t kq[4];
        float amax = 0.0f;
        float kv[4];
        if constexpr (PACKED) {
            // 68B row: lane's 4 elements live in one block (b = lane>>3);
            // dequant in place, values are QAT by construction
            const uint8_t * kjb = (const uint8_t *) k + (int64_t) j * nbk2;
            const uint8_t * blk = kjb + (lane >> 3) * 17;
            const int j0 = (lane * 4) & 31;
            const float d = exp2f((float) ((int) blk[0] - 127));
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const uint8_t byte = blk[1 + ((j0 + i) & 15)];
                const int ni = j0 < 16 ? (byte & 0x0F) : (byte >> 4);
                kv[i] = DSV4_LV16[ni] * d;
                amax = fmaxf(amax, fabsf(kv[i]));
            }
        } else {
            const KT * kj = k + (int64_t) j * nbk2;
#pragma unroll
            for (int i = 0; i < 4; i++) { kv[i] = dsv4_ldk(kj + lane * 4 + i); amax = fmaxf(amax, fabsf(kv[i])); }
        }
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
        if (lane == 0) scores[j] = __float2half(acc + mask[j]);
    }
}

// ---------------------------------------------------------------------------
// B2 per-tile union + membership (one block per (tile, stream), smem bitmap
// over n_csa; W tokens per tile, last tile may be partial)
// ---------------------------------------------------------------------------

static __global__ void dsv4_union_kernel(
        int32_t * __restrict__ out, const int32_t * __restrict__ top_k,
        int64_t nb1_tk, int64_t nb3_tk, int64_t nb1_out, int64_t nb3_out,
        int n_top_k, int nt_s, int n_csa, int u_max, int W,
        int32_t * __restrict__ stats /* optional [n_tiles*n_stream] exact union sizes */) {
    const int tile = blockIdx.x;
    const int s    = blockIdx.y;
    const int t0   = tile * W;
    const int t1   = min(t0 + W, nt_s);
    extern __shared__ uint32_t bm[];
    const int n_words = (n_csa + 31) / 32;
    for (int i = threadIdx.x; i < n_words; i += blockDim.x) bm[i] = 0;
    __syncthreads();
    const int total = n_top_k * (t1 - t0);
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int t = t0 + idx / n_top_k, i = idx % n_top_k;
        const int c = top_k[(int64_t) i + t * nb1_tk + (int64_t) s * nb3_tk];
        if (c >= 0 && c < n_csa) atomicOr(&bm[c >> 5], 1u << (c & 31));
    }
    __syncthreads();
    if (stats) {
        __shared__ int total_cnt;
        if (threadIdx.x == 0) total_cnt = 0;
        __syncthreads();
        int c = 0;
        for (int i = threadIdx.x; i < n_words; i += blockDim.x) c += __popc(bm[i]);
        atomicAdd(&total_cnt, c);
        __syncthreads();
        if (threadIdx.x == 0) stats[(int64_t) s * gridDim.x + tile] = total_cnt;
    }
    if (threadIdx.x == 0) {
        int32_t * o = out + (int64_t) tile * nb1_out + (int64_t) s * nb3_out;
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

// One block per (token-group, stream). Uses union_idx of the token's tile
// directly (binary search) so the membership rank is GUARANTEED consistent
// with the gather order. The tile's union is cached in smem and reloaded when
// the token group crosses a tile boundary (never happens for W % TPB == 0).
#define DSV4_MEMB_TPB 8
static __global__ void dsv4_memb_kernel(
        float * __restrict__ memb, const int32_t * __restrict__ top_k,
        const int32_t * __restrict__ union_idx,
        int64_t nb1_tk, int64_t nb3_tk, int64_t nb1_m, int64_t nb3_m, int64_t nb1_u, int64_t nb3_u,
        int n_top_k, int nt_s, int n_csa, int u_max, int W) {
    const int s      = blockIdx.y;
    const int t_base = blockIdx.x * DSV4_MEMB_TPB;
    extern __shared__ int32_t uni_s[]; // [u_max] union_idx for the current tile
    int cur_tile = -1;

    for (int lt = 0; lt < DSV4_MEMB_TPB; lt++) {
        const int t = t_base + lt;
        if (t >= nt_s) break;
        const int tile = t / W;
        if (tile != cur_tile) {
            __syncthreads();
            for (int i = threadIdx.x; i < u_max; i += blockDim.x) {
                uni_s[i] = union_idx[(int64_t) i + (int64_t) tile * nb1_u + (int64_t) s * nb3_u];
            }
            __syncthreads();
            cur_tile = tile;
        }
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

// QAT e2m1 round-trip at lid-cache write time (GGML_OP_DSV4_FP4_RT):
// contiguous f32 rows -> contiguous f32 rows, one block per 128-wide row
// chunk via the existing quant kernel (identity strides).
void ggml_cuda_op_dsv4_fp4_rt(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    GGML_ASSERT(x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(dst));
    const int64_t ne0   = x->ne[0];
    GGML_ASSERT(ne0 % 32 == 0 && ne0 <= 1024);
    const int64_t nrows = ggml_nrows(x);
    dsv4_fp4_quant_kernel<float><<<nrows, ne0, 0, ctx.stream()>>>(
            (float *) dst->data, (const float *) x->data, ne0, 0, nrows, ne0);
}

// QAT-rounded scatter into the packed MXFP4 lid container
// (GGML_OP_DSV4_QAT_SET_ROWS). dst is a view of the container; src0 = f32
// rows, src1 = row indices.
void ggml_cuda_op_dsv4_qat_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * b = dst->src[0];
    const ggml_tensor * c = dst->src[1];
    GGML_ASSERT(dst->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(b->type == GGML_TYPE_F32);
    const int64_t ne0 = b->ne[0];
    GGML_ASSERT(ne0 % 32 == 0 && ne0 <= 1024);
    const int64_t n_rows = b->ne[1];
    if (n_rows == 0) {
        return;
    }
    dsv4_qat_set_rows_kernel<<<n_rows, ne0, 0, ctx.stream()>>>(
            (uint8_t *) dst->data, (const float *) b->data, c->data,
            c->type == GGML_TYPE_I64 ? 1 : 0,
            dst->nb[1], b->nb[1] / sizeof(float), (int) ne0);
}

// merge two flash_attn_ext_with_lse results computed over disjoint KV subsets
// (GGML_OP_DSV4_FA_MERGE): out = (ea*a + eb*b)/(ea+eb) per row,
// ea = exp(lse_a - max(lse_a, lse_b)). srcs [DV, H, Q, S+1] with LSE tail at
// element offset DV*H*Q*S, tail idx == row idx; dst [DV, H, Q, S].
static __global__ void dsv4_fa_merge_kernel(
        const float * __restrict__ a, const float * __restrict__ b, const float * __restrict__ c,
        float * __restrict__ dst, const int DV, const int64_t n_rows) {
    const int64_t r = blockIdx.x;

    const float la = a[DV*n_rows + r];
    const float lb = b[DV*n_rows + r];
    const float lc = c ? c[DV*n_rows + r] : -INFINITY;
    const float m  = fmaxf(fmaxf(la, lb), lc);

    float wa = 0.0f;
    float wb = 0.0f;
    float wc = 0.0f;
    if (m != -INFINITY) { // else all parts fully masked -> zeros
        const float ea = expf(la - m);
        const float eb = expf(lb - m);
        const float ec = c ? expf(lc - m) : 0.0f;
        const float es = ea + eb + ec;
        wa = ea / es;
        wb = eb / es;
        wc = ec / es;
    }

    const float * ar = a   + r*DV;
    const float * br = b   + r*DV;
    const float * cr = c ? c + r*DV : nullptr;
    float       * dr = dst + r*DV;

    // a part with lse == -inf (fully masked, weight exactly 0) may carry
    // NaN/uninitialized values — skip it entirely, never multiply by 0
    for (int d = threadIdx.x; d < DV; d += blockDim.x) {
        float v = 0.0f;
        if (wa != 0.0f) { v += wa*ar[d]; }
        if (wb != 0.0f) { v += wb*br[d]; }
        if (wc != 0.0f) { v += wc*cr[d]; }
        dr[d] = v;
    }
}

void ggml_cuda_op_dsv4_fa_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];
    const ggml_tensor * c = dst->src[2]; // optional third disjoint subset (merge3)

    GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(dst));
    GGML_ASSERT(c == nullptr || (c->type == GGML_TYPE_F32 && ggml_is_contiguous(c)));

    const int64_t DV     = dst->ne[0];
    const int64_t n_rows = dst->ne[1]*dst->ne[2]*dst->ne[3];
    if (n_rows == 0) {
        return;
    }

    const int n_threads = DV >= 256 ? 256 : 128;
    dsv4_fa_merge_kernel<<<n_rows, n_threads, 0, ctx.stream()>>>(
            (const float *) a->data, (const float *) b->data,
            c ? (const float *) c->data : nullptr, (float *) dst->data,
            (int) DV, n_rows);
}

void ggml_cuda_op_dsv4_lid_union(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * top_k = dst->src[0];
    GGML_ASSERT(top_k->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32);
    const int n_csa   = ggml_get_op_params_i32(dst, 0);
    const int u_max   = dst->ne[0];
    const int n_tiles = dst->ne[1];
    const int n_top_k = top_k->ne[0];
    const int nt_s    = top_k->ne[1];
    const int n_stream = top_k->ne[3];
    const int W_p = ggml_get_op_params_i32(dst, 2);
    const int W   = (n_tiles > 1 && W_p > 0) ? W_p : nt_s;
    const int64_t nb1_tk  = top_k->nb[1] / sizeof(int32_t);
    const int64_t nb3_tk  = top_k->nb[3] / sizeof(int32_t);
    const int64_t nb1_out = dst->nb[1] / sizeof(int32_t);
    const int64_t nb3_out = dst->nb[3] / sizeof(int32_t);
    const size_t smem = (size_t) ((n_csa + 31) / 32) * sizeof(uint32_t);
    const dim3 grid(n_tiles, n_stream, 1);

    // LLAMA_DSV4_UNION_STATS=1: report exact per-tile union sizes (popcount of
    // the full bitmap, i.e. including cells dropped by the u_max cap)
    static const bool union_stats = getenv("LLAMA_DSV4_UNION_STATS") != nullptr;
    if (union_stats && n_tiles > 1) {
        ggml_cuda_pool_alloc<int32_t> stats_buf(ctx.pool(), (size_t) n_tiles * n_stream);
        dsv4_union_kernel<<<grid, 256, smem, ctx.stream()>>>(
            (int32_t *) dst->data, (const int32_t *) top_k->data,
            nb1_tk, nb3_tk, nb1_out, nb3_out, n_top_k, nt_s, n_csa, u_max, W, stats_buf.get());
        std::vector<int32_t> h((size_t) n_tiles * n_stream);
        CUDA_CHECK(cudaMemcpyAsync(h.data(), stats_buf.get(), h.size()*sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, ctx.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        int cmin = INT32_MAX, cmax = 0, nover = 0;
        int64_t csum = 0;
        for (int32_t v : h) {
            cmin = v < cmin ? v : cmin;
            cmax = v > cmax ? v : cmax;
            csum += v;
            nover += v > u_max;
        }
        fprintf(stderr, "US n_csa=%d W=%d T=%d u_max=%d cnt min=%d mean=%d max=%d over=%d/%zu\n",
                n_csa, W, n_tiles, u_max, cmin, (int)(csum/(int64_t)h.size()), cmax, nover, h.size());
        return;
    }

    dsv4_union_kernel<<<grid, 256, smem, ctx.stream()>>>(
        (int32_t *) dst->data, (const int32_t *) top_k->data,
        nb1_tk, nb3_tk, nb1_out, nb3_out, n_top_k, nt_s, n_csa, u_max, W, nullptr);
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
    const int n_tiles = uni->ne[1];
    const int W_p = ggml_get_op_params_i32(uni, 2);
    const int W   = (n_tiles > 1 && W_p > 0) ? W_p : nt_s;
    const int64_t nb1_tk = top_k->nb[1] / sizeof(int32_t);
    const int64_t nb3_tk = top_k->nb[3] / sizeof(int32_t);
    const int64_t nb1_u  = uni->nb[1] / sizeof(int32_t);
    const int64_t nb3_u  = uni->nb[3] / sizeof(int32_t);
    const int64_t nb1_m  = dst->nb[1] / sizeof(float);
    const int64_t nb3_m  = dst->nb[3] / sizeof(float);
    const dim3 grid((nt_s + DSV4_MEMB_TPB - 1) / DSV4_MEMB_TPB, n_stream, 1);
    const size_t smem = (size_t) u_max * sizeof(int32_t);
    dsv4_memb_kernel<<<grid, 256, smem, ctx.stream()>>>(
        (float *) dst->data, (const int32_t *) top_k->data, (const int32_t *) uni->data,
        nb1_tk, nb3_tk, nb1_m, nb3_m, nb1_u, nb3_u, n_top_k, nt_s, n_csa, u_max, W);
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

static __device__ __forceinline__ uint32_t dsv4_next_pow2(uint32_t x) {
    return x <= 1u ? 1u : (1u << (32 - __clz((int) (x - 1u))));
}

// Partial bitonic top-K selection: leaves the top K elements of
// vals/idxs[0..SORT_N) (dsv4_better order) in [0..K), sorted descending.
// K must be a power of two <= SORT_N. Phase 1 sorts every K-block
// descending (the k==K stage forces one direction for all blocks; their
// input is bitonic from the alternating k<K stages, so the merge is valid
// for either direction). Phase 2 halves the candidate blocks per round:
// fold pairs of descending blocks (compare-exchange A[i] vs B[K-1-i] —
// the winners are exactly the top K of the union and form a bitonic
// sequence), then one descending bitonic merge re-sorts the winners.
// dsv4_better is a strict total order (val desc, idx asc), so the top-K
// SET is unique — output is identical to a full sort's first K.
// Work ~ N/2*log^2 K + fold rounds, vs the full sort's N/2*log^2 N.
template <uint32_t SORT_N>
static __device__ void dsv4_bitonic_topk(float * vals, uint32_t * idxs, uint32_t K) {
    if (K >= SORT_N) {
        dsv4_bitonic_sort<SORT_N>(vals, idxs);
        return;
    }
    // phase 1: sort each K-block descending
    for (uint32_t k = 2u; k <= K; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = threadIdx.x; i < SORT_N; i += blockDim.x) {
                const uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float    av = vals[i],    bv = vals[other];
                    const uint32_t ai = idxs[i],    bi = idxs[other];
                    const bool desc_half = (k == K) ? true : ((i & k) == 0u);
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
    // phase 2: fold + re-sort rounds; active blocks start at multiples of
    // (step<<1), winners stay in the low block of each pair
    uint32_t n_active = SORT_N / K;
    uint32_t step     = K;
    while (n_active > 1u) {
        const uint32_t n_pairs = n_active >> 1u;
        for (uint32_t t = threadIdx.x; t < n_pairs * K; t += blockDim.x) {
            const uint32_t p = t / K, i = t % K;
            const uint32_t a = p * (step << 1u) + i;
            const uint32_t b = p * (step << 1u) + step + (K - 1u - i);
            const float    av = vals[a], bv = vals[b];
            const uint32_t ai = idxs[a], bi = idxs[b];
            if (dsv4_better(bv, bi, av, ai)) {
                vals[a] = bv; idxs[a] = bi;
                vals[b] = av; idxs[b] = ai;
            }
        }
        __syncthreads();
        for (uint32_t j = K >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t t = threadIdx.x; t < n_pairs * K; t += blockDim.x) {
                const uint32_t p = t / K, i = t % K;
                const uint32_t x = p * (step << 1u) + i;
                const uint32_t other = x ^ j; // stays in-block: j < K, block K-aligned
                if (other > x) {
                    const float    av = vals[x],    bv = vals[other];
                    const uint32_t ai = idxs[x],    bi = idxs[other];
                    if (dsv4_better(bv, bi, av, ai)) {
                        vals[x] = bv; idxs[x] = bi;
                        vals[other] = av; idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
        step <<= 1u;
        n_active = n_pairs;
    }
}

// single-block-per-token, n_lid <= SORT_N.
// The three topk kernels carve one dynamic extern buffer (host raises the
// limit via CUDA_SET_SHARED_MEMORY_LIMIT) so SORT_N is not capped by the
// 48KB static smem limit. NOTE: SORT_N=8192 was measured a net NEGATIVE
// (+7-13% op time at n_lid 8704-17000): bitonic work grows N*log^2 N, so
// fewer-but-bigger chunks add ~30% sort work — more than the halved merge
// launches save. Keep 4096.
template <uint32_t SORT_N, typename score_t>
static __global__ void dsv4_topk_single_kernel(
        uint32_t * selected, const score_t * scores,
        int n_lid, int n_tokens, int top_k) {
    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    extern __shared__ float dsv4_topk_smem[];
    float    * vals = dsv4_topk_smem;
    uint32_t * idxs = reinterpret_cast<uint32_t *>(dsv4_topk_smem + SORT_N);
    const score_t * row = scores + (int64_t)t * n_lid;
    for (uint32_t i = threadIdx.x; i < SORT_N; i += blockDim.x) {
        if ((int)i < n_lid) { vals[i] = dsv4_lds(row + i); idxs[i] = i; }
        else                { vals[i] = -INFINITY; idxs[i] = UINT32_MAX; }
    }
    __syncthreads();
    dsv4_bitonic_topk<SORT_N>(vals, idxs, dsv4_next_pow2((uint32_t) top_k));
    for (int i = threadIdx.x; i < top_k; i += blockDim.x) {
        selected[(int64_t)t*top_k + i] = idxs[i];
    }
}

// per (token, chunk): emit top_k candidate indices for this chunk
template <uint32_t SORT_N, typename score_t>
static __global__ void dsv4_topk_chunk_kernel(
        uint32_t * candidates, const score_t * scores,
        int n_lid, int n_tokens, int top_k, int candidate_stride) {
    const int t     = blockIdx.x;
    const int chunk = blockIdx.y;
    if (t >= n_tokens) return;
    const uint32_t chunk_start = (uint32_t)chunk * SORT_N;
    if ((int)chunk_start >= n_lid) return;
    const uint32_t chunk_n = ((int)(n_lid - chunk_start) < (int)SORT_N) ? (uint32_t)(n_lid - chunk_start) : SORT_N;
    extern __shared__ float dsv4_topk_smem[];
    float    * vals = dsv4_topk_smem;
    uint32_t * idxs = reinterpret_cast<uint32_t *>(dsv4_topk_smem + SORT_N);
    const score_t * row = scores + (int64_t)t * n_lid;
    for (uint32_t i = threadIdx.x; i < SORT_N; i += blockDim.x) {
        if (i < chunk_n) { vals[i] = dsv4_lds(row + chunk_start + i); idxs[i] = chunk_start + i; }
        else             { vals[i] = -INFINITY; idxs[i] = UINT32_MAX; }
    }
    __syncthreads();
    dsv4_bitonic_topk<SORT_N>(vals, idxs, dsv4_next_pow2((uint32_t) top_k));
    uint32_t * out = candidates + (int64_t)t*candidate_stride + (int64_t)chunk*top_k;
    for (int i = threadIdx.x; i < top_k; i += blockDim.x) out[i] = idxs[i];
}

// merge `set_count` candidate sets (per group) -> top_k, re-reading scores
template <uint32_t SORT_N, typename score_t>
static __global__ void dsv4_topk_merge_kernel(
        uint32_t * out, const uint32_t * candidates, const score_t * scores,
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

    extern __shared__ float dsv4_topk_smem[];
    float    * vals = dsv4_topk_smem;
    uint32_t * idxs = reinterpret_cast<uint32_t *>(dsv4_topk_smem + SORT_N);
    const score_t  * row  = scores + (int64_t)t * n_lid;
    const uint32_t * cand = candidates + (int64_t)t*candidate_stride + (int64_t)set0*top_k;
    for (uint32_t i = threadIdx.x; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float    v   = -INFINITY;
        if ((int)i < candidate_count) {
            idx = cand[i];
            if (idx < (uint32_t) n_lid) v = dsv4_lds(row + idx);
        }
        vals[i] = v; idxs[i] = idx;
    }
    __syncthreads();
    dsv4_bitonic_topk<SORT_N>(vals, idxs, dsv4_next_pow2((uint32_t) top_k));
    uint32_t * dst = final ? (out + (int64_t)t*top_k)
                           : (out + (int64_t)t*out_stride + (int64_t)group*top_k);
    for (int i = threadIdx.x; i < top_k; i += blockDim.x) dst[i] = idxs[i];
}

// ---------------------------------------------------------------------------
// radix top-k (LLAMA_DSV4_LID_RADIX, default ON) — d128/half prefill path.
// One block per token replaces the whole chunk+merge bitonic tree: two 8-bit
// MSB-first histogram passes over order-preserving 16-bit keys find the
// threshold key T, an ordered prefix-scan compaction emits every key > T plus
// the LOWEST-INDEX (top_k - count_gt) members of key == T, and one small
// bitonic sorts the top_k selected pairs. Selection is bit-identical to the
// bitonic path: key equality == f32 value equality of the stored halves
// (after canonicalizing -0, which compares equal to +0 in f32 but has a
// distinct bit pattern and IS reachable via __float2half of a tiny negative
// score), so "all > T, then lowest index among == T" is exactly dsv4_better.
// The compaction MUST be an ordered scan (tiles ascending, in-tile prefix
// preserves index order) — an atomic append would be nondeterministic in
// WHICH ==T members win. No NaN can appear (finite acc + {0, -INF} mask).
// ---------------------------------------------------------------------------

static __device__ __forceinline__ uint16_t dsv4_score_key16(const half h) {
    uint16_t hb = __half_as_ushort(h);
    if (hb == 0x8000u) hb = 0u; // -0 == +0 under f32 compare
    return (hb & 0x8000u) ? (uint16_t) ~hb : (uint16_t) (hb | 0x8000u);
}

// runtime-width variant of dsv4_bitonic_sort (n = pow2 <= 4096); sorts the
// selected candidates descending for the output contract
static __device__ void dsv4_bitonic_sort_rt(float * vals, uint32_t * idxs, uint32_t n) {
    for (uint32_t k = 2u; k <= n; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
                const uint32_t other = i ^ j;
                if (other > i && other < n) {
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

static __global__ void dsv4_topk_radix_kernel(
        uint32_t * __restrict__ selected, const half * __restrict__ scores,
        int n_lid, int top_k, uint32_t K2) {
    extern __shared__ uint8_t dsv4_radix_smem[];
    float    * sel_v = (float *) dsv4_radix_smem;              // [K2]
    uint32_t * sel_i = (uint32_t *) (sel_v + K2);              // [K2]
    uint32_t * hist  = sel_i + K2;                             // [256]
    uint32_t * wtot  = hist + 256;                             // [64] warp gt/eq totals -> offsets
    __shared__ uint32_t sh_bhi, sh_cnt_gt_hi, sh_cnt_gt, sh_T, sh_tile_gt, sh_tile_eq;

    const int t = blockIdx.x;
    const half * row = scores + (int64_t) t * n_lid;
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;

    // pass 1: high-byte histogram, suffix-scan for the threshold high byte
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) hist[i] = 0u;
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < (uint32_t) n_lid; i += blockDim.x) {
        atomicAdd(&hist[dsv4_score_key16(row[i]) >> 8], 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t suffix = 0u;
        for (int b = 255; b >= 0; b--) {
            if (suffix + hist[b] >= (uint32_t) top_k) { sh_bhi = (uint32_t) b; sh_cnt_gt_hi = suffix; break; }
            suffix += hist[b];
        }
    }
    __syncthreads();
    const uint32_t bhi = sh_bhi;

    // pass 2: low-byte histogram within the threshold high-byte group
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) hist[i] = 0u;
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < (uint32_t) n_lid; i += blockDim.x) {
        const uint16_t key = dsv4_score_key16(row[i]);
        if ((uint32_t) (key >> 8) == bhi) atomicAdd(&hist[key & 0xFFu], 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t cgh = sh_cnt_gt_hi;
        uint32_t suffix = 0u;
        for (int b = 255; b >= 0; b--) {
            if (cgh + suffix + hist[b] >= (uint32_t) top_k) {
                sh_T      = (bhi << 8) | (uint32_t) b;
                sh_cnt_gt = cgh + suffix;
                break;
            }
            suffix += hist[b];
        }
    }
    __syncthreads();
    const uint32_t T      = sh_T;
    const uint32_t cnt_gt = sh_cnt_gt;
    const uint32_t need   = (uint32_t) top_k - cnt_gt;

    // pass 3: ordered compaction — tiles ascending + in-tile prefix scan
    // preserve index order, so the ==T fill takes the LOWEST indices
    uint32_t running_gt = 0u, running_eq = 0u;
    for (uint32_t start = 0u; start < (uint32_t) n_lid; start += blockDim.x) {
        const uint32_t i = start + threadIdx.x;
        half hv = __float2half(0.0f);
        bool gt = false, eq = false;
        if (i < (uint32_t) n_lid) {
            hv = row[i];
            const uint32_t key = dsv4_score_key16(hv);
            gt = key > T;
            eq = key == T;
        }
        const uint32_t bal_gt = __ballot_sync(0xffffffffu, gt);
        const uint32_t bal_eq = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) { wtot[wid] = __popc(bal_gt); wtot[32 + wid] = __popc(bal_eq); }
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t sg = 0u, se = 0u;
            for (int w = 0; w < 32; w++) {
                const uint32_t g = wtot[w], e = wtot[32 + w];
                wtot[w] = sg; wtot[32 + w] = se; // exclusive warp offsets
                sg += g; se += e;
            }
            sh_tile_gt = sg; sh_tile_eq = se;
        }
        __syncthreads();
        if (gt) {
            const uint32_t slot = running_gt + wtot[wid] + __popc(bal_gt & ((1u << lane) - 1u));
            sel_v[slot] = __half2float(hv); sel_i[slot] = i;
        }
        if (eq) {
            const uint32_t e = running_eq + wtot[32 + wid] + __popc(bal_eq & ((1u << lane) - 1u));
            if (e < need) { sel_v[cnt_gt + e] = __half2float(hv); sel_i[cnt_gt + e] = i; }
        }
        running_gt += sh_tile_gt;
        running_eq += sh_tile_eq;
        __syncthreads(); // wtot reused next tile
    }

    // pad to K2 and sort the selected top_k descending
    for (uint32_t i = threadIdx.x; i < K2; i += blockDim.x) {
        if (i >= (uint32_t) top_k) { sel_v[i] = -INFINITY; sel_i[i] = UINT32_MAX; }
    }
    __syncthreads();
    dsv4_bitonic_sort_rt(sel_v, sel_i, K2);
    for (uint32_t i = threadIdx.x; i < (uint32_t) top_k; i += blockDim.x) {
        selected[(int64_t) t * top_k + i] = sel_i[i];
    }
}

static void dsv4_topk_radix_launch(
        uint32_t * selected, const half * scores,
        int n_lid, int n_tokens, int top_k, cudaStream_t stream) {
    uint32_t K2 = 1u;
    while (K2 < (uint32_t) top_k) K2 <<= 1u;
    const size_t smem = (size_t) K2 * 8 + 256 * 4 + 64 * 4;
    dsv4_topk_radix_kernel<<<n_tokens, 1024, smem, stream>>>(
            selected, scores, n_lid, top_k, K2);
}

// radix needs enough token-blocks to fill the GPU and a row wider than the
// selection; everything else (f32 scores, decode/small-nt) keeps bitonic
static bool dsv4_topk_try_radix(uint32_t *, const float *, int, int, int, cudaStream_t) {
    return false;
}
static bool dsv4_topk_try_radix(uint32_t * selected, const half * scores,
        int n_lid, int n_tokens, int top_k, cudaStream_t stream) {
    static const bool radix_on = []() {
        const char * e = getenv("LLAMA_DSV4_LID_RADIX");
        return !e || e[0] != '0';
    }();
    if (!radix_on || n_tokens < 16 || n_lid <= top_k) {
        return false;
    }
    dsv4_topk_radix_launch(selected, scores, n_lid, n_tokens, top_k, stream);
    return true;
}

// ---------------------------------------------------------------------------
// exact-selection pass 2 (LLAMA_DSV4_LID_EXACT): serial-order fp32 rescore of
// the pass-1 candidates. One block per token; each thread owns candidates
// block-strided and accumulates the score in the SAME serial order as the CPU
// reference (d ascending within head, h ascending), so candidate scores are
// bitwise equal to the reference and the final top-k selection (desc score,
// asc index tie-break) is exact — pass-1 numerics only need to land the true
// top-k inside the candidate window.
// ---------------------------------------------------------------------------

// Phase A: rescore a chunk of candidates for one token. Grid (nt, n_chunks);
// each block stages its chunk's K rows into smem (coalesced, read once) and
// one thread per candidate accumulates in the reference serial order (each
// head's dot is its own ascending-d chain; 8 heads interleaved for ILP; heads
// combined in ascending order) — bitwise the CPU/ds4.c result.
// PACKED: k points at 17B-block MXFP4 rows, nbk2/nbk3 are BYTE strides; the
// smem stage dequants inline (KT must be float) — the compute loop below is
// untouched, so candidate scores stay bitwise the CPU/ds4.c reference.
template <typename KT, bool PACKED = false>
static __global__ void dsv4_lid_rescore_score_kernel(
        float * __restrict__ cand_vals, const uint32_t * __restrict__ cand,
        const float * __restrict__ q,     // QAT f32 [d_idx, n_head, nt] contiguous
        const float * __restrict__ w,     // [n_head, nt] contiguous
        const KT    * __restrict__ k,     // QAT values, strided (f32 pool copy or f16 cache)
        const float * __restrict__ mask,  // strided f32
        int64_t nbk2, int64_t nbk3, int64_t nbm1, int64_t nbm3,
        int n_cand, int n_lid, int n_head, int d_idx, int nt_s) {
    constexpr int WAVE = 128; // candidates per block; smem = WAVE*d_idx*sizeof(KT)
    const int t       = blockIdx.x;
    const int s       = t / nt_s;
    const int t_local = t % nt_s;
    const int base    = blockIdx.y * WAVE;
    if (base >= n_cand) return;
    const int wn = (n_cand - base) < WAVE ? (n_cand - base) : WAVE;
    __shared__ KT rows[WAVE * 128];
    const float    * q_t = q + (int64_t) t * n_head * d_idx;
    const float    * w_t = w + (int64_t) t * n_head;
    const uint32_t * c_t = cand + (int64_t) t * n_cand;
    for (int i = threadIdx.x; i < wn * d_idx; i += blockDim.x) {
        const uint32_t j = c_t[base + i / d_idx];
        if constexpr (PACKED) {
            rows[i] = (j < (uint32_t) n_lid)
                    ? (KT) dsv4_mxfp4_get((const uint8_t *) k + (int64_t) s * nbk3 + (int64_t) j * nbk2, i % d_idx)
                    : (KT) 0.0f;
        } else {
            rows[i] = (j < (uint32_t) n_lid)
                    ? k[(int64_t) s * nbk3 + (int64_t) j * nbk2 + (i % d_idx)]
                    : (KT) 0.0f;
        }
    }
    __syncthreads();
    for (int ci = threadIdx.x; ci < wn; ci += blockDim.x) {
        const uint32_t j = c_t[base + ci];
        float sc = -INFINITY;
        if (j < (uint32_t) n_lid) {
            const KT * kj = rows + (int64_t) ci * d_idx;
            float acc = 0.0f;
            for (int h = 0; h < n_head; h += 8) {
                const int hu = (n_head - h) < 8 ? (n_head - h) : 8;
                float dt[8] = {0,0,0,0,0,0,0,0};
                const float * qh = q_t + (int64_t) h * d_idx;
                if (hu == 8) {
                    for (int d = 0; d < d_idx; d++) {
                        const float kv = dsv4_ldk(kj + d);
#pragma unroll
                        for (int u = 0; u < 8; u++) dt[u] += qh[u*d_idx + d] * kv;
                    }
                } else {
                    for (int d = 0; d < d_idx; d++) {
                        const float kv = dsv4_ldk(kj + d);
                        for (int u = 0; u < hu; u++) dt[u] += qh[u*d_idx + d] * kv;
                    }
                }
                for (int u = 0; u < hu; u++) acc += fmaxf(dt[u], 0.0f) * w_t[h + u];
            }
            sc = acc + mask[(int64_t) s * nbm3 + (int64_t) t_local * nbm1 + (int64_t) j];
        }
        cand_vals[(int64_t) t * n_cand + base + ci] = sc;
    }
}

// Phase B: exact top-k over the rescored candidates (desc score, asc index
// tie-break). One block per token, pure smem bitonic.
static __global__ void dsv4_lid_rescore_select_kernel(
        int32_t * __restrict__ out, const uint32_t * __restrict__ cand,
        const float * __restrict__ cand_vals,
        int n_cand, int top_k) {
    const int t = blockIdx.x;
    __shared__ float    vals[1024];
    __shared__ uint32_t idxs[1024];
    const uint32_t * c_t = cand      + (int64_t) t * n_cand;
    const float    * v_t = cand_vals + (int64_t) t * n_cand;
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        vals[i] = (i < n_cand) ? v_t[i] : -INFINITY;
        idxs[i] = (i < n_cand) ? c_t[i] : UINT32_MAX;
    }
    __syncthreads();
    dsv4_bitonic_sort<1024u>(vals, idxs);
    for (int i = threadIdx.x; i < top_k; i += blockDim.x) {
        out[(int64_t) t * top_k + i] = (int32_t) idxs[i];
    }
}

// ---------------------------------------------------------------------------
// host launcher
// ---------------------------------------------------------------------------

template <typename score_t>
static void dsv4_topk_launch(
        ggml_cuda_pool & pool, uint32_t * selected, const score_t * scores,
        int n_lid, int n_tokens, int top_k, cudaStream_t stream) {
    constexpr uint32_t SORT_N = DSV4_TOPK_SORT_N;
    GGML_ASSERT((uint32_t) top_k <= SORT_N);

    if (dsv4_topk_try_radix(selected, scores, n_lid, n_tokens, top_k, stream)) {
        return;
    }

    const int block = 1024;
    const size_t smem = (size_t) SORT_N * (sizeof(float) + sizeof(uint32_t));

    if ((uint32_t) n_lid <= SORT_N) {
        CUDA_SET_SHARED_MEMORY_LIMIT((dsv4_topk_single_kernel<SORT_N, score_t>), smem);
        dsv4_topk_single_kernel<SORT_N, score_t><<<n_tokens, block, smem, stream>>>(
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
    CUDA_SET_SHARED_MEMORY_LIMIT((dsv4_topk_chunk_kernel<SORT_N, score_t>), smem);
    dsv4_topk_chunk_kernel<SORT_N, score_t><<<grid_chunks, block, smem, stream>>>(
            cur, scores, n_lid, n_tokens, top_k, candidate_stride);

    // tree merges until n_sets <= merge_group
    while (n_sets > merge_group) {
        const int next_sets   = (n_sets + merge_group - 1) / merge_group;
        const int next_stride = next_sets * top_k;
        uint32_t * next = cur + (int64_t) n_tokens * cur_stride;
        dim3 grid_merge(n_tokens, next_sets, 1);
        CUDA_SET_SHARED_MEMORY_LIMIT((dsv4_topk_merge_kernel<SORT_N, score_t>), smem);
        dsv4_topk_merge_kernel<SORT_N, score_t><<<grid_merge, block, smem, stream>>>(
                next, cur, scores, n_lid, n_tokens, top_k,
                n_sets, merge_group, cur_stride, next_stride, /*final=*/0);
        cur = next;
        n_sets = next_sets;
        cur_stride = next_stride;
    }

    // final merge -> selected
    dim3 grid_final(n_tokens, 1, 1);
    CUDA_SET_SHARED_MEMORY_LIMIT((dsv4_topk_merge_kernel<SORT_N, score_t>), smem);
    dsv4_topk_merge_kernel<SORT_N, score_t><<<grid_final, block, smem, stream>>>(
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
    GGML_ASSERT(k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(k->nb[0] == ggml_type_size(k->type)); // head dim contiguous
    GGML_ASSERT(k->type != GGML_TYPE_MXFP4 || q->ne[0] % 32 == 0);

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
    if (dump_dir && !dumped && k->type == GGML_TYPE_MXFP4) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "\ndsv4_lid_topk: LID_DUMP not supported with packed MXFP4 cache — skipping\n");
        }
    } else if (dump_dir && !dumped && nt > 1 && n_lid >= dump_nlid) {
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
    // d_idx==128 kernels store f16 scores to halve the [nt, n_lid] DRAM
    // round-trip; the scalar (d_idx != 128) path keeps f32 to stay bit-exact
    // under its strict 0.0-tolerance test gate.
    const bool scores_half = (d_idx == 128);
    ggml_cuda_pool_alloc<float> scores_f32_alloc(pool);
    ggml_cuda_pool_alloc<half>  scores_h_alloc(pool);
    float * scores   = scores_half ? nullptr : scores_f32_alloc.alloc((size_t) nt * n_lid);
    half  * scores_h = scores_half ? scores_h_alloc.alloc((size_t) nt * n_lid) : nullptr;

    // element strides are meaningless for the packed MXFP4 container; its
    // staging path below overwrites these with dense-buffer strides
    const bool k_is_mxfp4 = k->type == GGML_TYPE_MXFP4;
    int64_t nbk2 = k_is_mxfp4 ? 0 : k->nb[2] / ggml_type_size(k->type);
    int64_t nbk3 = k_is_mxfp4 ? 0 : k->nb[3] / ggml_type_size(k->type);
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
    // Exact selection (LLAMA_DSV4_LID_EXACT): two-pass — implies the fp4/QAT
    // path so pass-2 rescores the official numerics. m = candidate margin.
    static const bool dsv4_lid_exact = []() {
        const char * e = getenv("LLAMA_DSV4_LID_EXACT");
        return e && e[0] == '1';
    }();
    static const int dsv4_rescore_m = []() {
        const char * e = getenv("LLAMA_DSV4_LID_RESCORE_M");
        return e ? atoi(e) : 64;
    }();
    // QAT-at-write: the lid cache already holds official QAT values (written
    // via GGML_OP_DSV4_FP4_RT), so the per-call k-side re-quant is skipped and
    // the f16 fast paths stay in use. q-side QAT still applies per call.
    static const bool dsv4_lid_qat_write = []() {
        const char * e = getenv("LLAMA_DSV4_LID_QAT_WRITE");
        return e && e[0] == '1';
    }();
    static const bool dsv4_lid_fp4 = []() {
        const char * e = getenv("LLAMA_DSV4_LID_FP4");
        const char * x = getenv("LLAMA_DSV4_LID_EXACT");
        return (e && e[0] == '1') || (x && x[0] == '1');
    }();
    ggml_cuda_pool_alloc<float> q_fp4_alloc(pool);
    ggml_cuda_pool_alloc<float> k_fp4_alloc(pool);
    ggml_cuda_pool_alloc<half>  k_f16_alloc(pool); // packed-prefill staging (f16-of-QAT, bit-exact)
    bool k_force_f32 = false;
    GGML_ASSERT(!dsv4_lid_fp4 || d_idx % 32 == 0);
    if (dsv4_lid_fp4) {
        float * q_fp4 = q_fp4_alloc.alloc((size_t) nt * n_head * d_idx);
        // q: contiguous rows, always quantized per call (cheap)
        dsv4_fp4_quant_kernel<float><<<n_head * nt, d_idx, 0, stream>>>(
                q_fp4, q_d, d_idx, 0, n_head * nt, d_idx);
        q_d = q_fp4;
    }
    // int8 dp4a path (default ON; LLAMA_DSV4_LID_INT8=0 disables): halves K
    // smem width to attack the L1-bandwidth bound of the wmma path. Same
    // per-stream launch geometry.
    static const bool dsv4_lid_int8 = []() {
        const char * e = getenv("LLAMA_DSV4_LID_INT8");
        return !e || e[0] != '0';
    }();
    // Dedicated decode kernel (nt_s==1): warp-per-comp, no 16-token tile
    // padding. Default ON; LLAMA_DSV4_LID_DEC=0 disables.
    static const bool dsv4_lid_dec = []() {
        const char * e = getenv("LLAMA_DSV4_LID_DEC");
        return !e || e[0] != '0';
    }();
    // fp4-mma (step 3): register-resident-K block-scaled fp4 tensor-core
    // scoring. Prefill only, packed MXFP4 container, Blackwell only.
    // Default ON since 2026-07-16 (=0 disables): flip gates passed — PPL
    // 4.2350 vs 4.2352 int8 (identical), passkey 42k 3/3 + deterministic,
    // op -47.8%, serving +4.7% pp@d65k / +8.6% @d131k; official e2m1 grid
    // (more canonical than int8). Only bites when the container is active.
    static const bool dsv4_lid_fp4_mma = []() {
        const char * e = getenv("LLAMA_DSV4_LID_FP4_MMA");
        return !e || e[0] != '0';
    }();
    bool fp4_mma_active = dsv4_lid_fp4_mma && k_is_mxfp4 && d_idx == 128 && nt_s > 1;
    if (fp4_mma_active) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        if (!(GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_BLACKWELL)) {
            static bool warned = false;
            if (!warned) { warned = true;
                fprintf(stderr, "\ndsv4_lid_topk: LID_FP4_MMA requested but device not Blackwell — using int8\n"); }
            fp4_mma_active = false;
        }
    }

    // P3b-ii: at decode (nt_s==1) the packed rows are read directly by the
    // 68B/row kernel variants — whole-cache staging there would cost more
    // traffic than the f16 baseline it replaces. Prefill keeps staging
    // (amortized over the ubatch; all wide kernels stay float-dispatch).
    const bool k_packed_direct = k_is_mxfp4 && d_idx == 128 && dsv4_lid_dec && nt_s == 1;
    if (fp4_mma_active) {
        // read the 68B block_mxfp4 rows directly — no staging, no pre-quant
        nbk2 = k->nb[2];
        nbk3 = k->nb[3];
    } else if (k_is_mxfp4 && !k_packed_direct) {
        // packed container: stage-dequant the whole strided cache view into a
        // dense F16 buffer. Rows hold QAT values by construction (written via
        // DSV4_QAT_SET_ROWS) and f16-of-QAT is bit-exact, so every downstream
        // kernel takes its (faster, half-width) f16 dispatch.
        half * k_deq = k_f16_alloc.alloc((size_t) n_stream * n_lid * d_idx);
        dsv4_mxfp4_dequant_rows_kernel<half><<<n_stream * n_lid, d_idx, 0, stream>>>(
                k_deq, (const uint8_t *) k->data, k->nb[2], k->nb[3], n_lid, d_idx);
        nbk2 = d_idx;
        nbk3 = (int64_t) d_idx * n_lid;
    } else if (k_packed_direct) {
        // byte strides for the packed-direct kernels
        nbk2 = k->nb[2];
        nbk3 = k->nb[3];
    } else if (dsv4_lid_fp4 && !dsv4_lid_qat_write) {
        float * k_fp4 = k_fp4_alloc.alloc((size_t) n_stream * n_lid * d_idx);
        // k: strided view, group = stream (stride nbk3), row = j (stride nbk2).
        if (k->type == GGML_TYPE_F16) {
            dsv4_fp4_quant_kernel<half><<<n_stream * n_lid, d_idx, 0, stream>>>(
                    k_fp4, (const half *) k->data, nbk2, nbk3, n_lid, d_idx);
        } else {
            dsv4_fp4_quant_kernel<float><<<n_stream * n_lid, d_idx, 0, stream>>>(
                    k_fp4, (const float *) k->data, nbk2, nbk3, n_lid, d_idx);
        }
        k_force_f32 = true;
        nbk2 = d_idx;
        nbk3 = (int64_t) d_idx * n_lid;
    }
    const float * k_f32_d = k_fp4_alloc.get();
    const half  * k_f16_d = k_f16_alloc.get(); // non-null only for packed-prefill staging
    const bool k_is_f16 = ((k->type == GGML_TYPE_F16) && !k_force_f32) || k_f16_d != nullptr;
    // half-arm K pointer: staged buffer wins over the raw cache
    const half * k_h_d = k_f16_d ? k_f16_d : (const half *) k->data;

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
            half        * sc_s = scores_h + (int64_t) s * nt_s * n_lid;
            const float * q_s  = q_d + (int64_t) s * nt_s * n_head * d_idx;
            const float * w_s  = w_d + (int64_t) s * nt_s * n_head;
            const float * m_s  = m_d + (int64_t) s * nbm3;
            if (k_packed_direct) {
                dsv4_score_decode_kernel<float, true><<<gx, block, smem, stream>>>(
                        sc_s, q_s, w_s, (const float *) ((const uint8_t *) k->data + (int64_t) s * nbk3), m_s, nbk2, n_lid, n_head);
            } else if (k_is_f16) {
                dsv4_score_decode_kernel<half><<<gx, block, smem, stream>>>(
                        sc_s, q_s, w_s, k_h_d + (int64_t) s * nbk3, m_s, nbk2, n_lid, n_head);
            } else {
                dsv4_score_decode_kernel<float><<<gx, block, smem, stream>>>(
                        sc_s, q_s, w_s, (k_force_f32 ? k_f32_d : (const float *) k->data) + (int64_t) s * nbk3, m_s, nbk2, n_lid, n_head);
            }
        }
    } else if (d_idx == 128) {
        const dim3 block(256);
        const dim3 grid((n_lid + 127) / 128, (nt_s + 15) / 16, 1);
        // int8 path: pre-quantize K to a global int8 buffer + per-comp scales
        // once per stream (bit-identical layout/arithmetic to the quant the
        // score kernel used to redo per 16-token tile).
        ggml_cuda_pool_alloc<int8_t> k_i8_alloc(pool);
        ggml_cuda_pool_alloc<float>  k_sc_alloc(pool);
        int8_t * k_i8 = nullptr;
        float  * k_sc = nullptr;
        if (dsv4_lid_int8 && !fp4_mma_active) {
            k_i8 = k_i8_alloc.alloc((size_t) n_stream * n_lid * 128);
            k_sc = k_sc_alloc.alloc((size_t) n_stream * n_lid);
            const int pq_block = 256; // 8 warps -> 8 comps/block
            const int pq_grid  = (n_lid + 7) / 8;
            for (int s = 0; s < n_stream; s++) {
                int8_t * ki_s = k_i8 + (int64_t) s * n_lid * 128;
                float  * ks_s = k_sc + (int64_t) s * n_lid;
                if (k_is_f16) {
                    dsv4_prequant_k_int8_kernel<half><<<pq_grid, pq_block, 0, stream>>>(
                            ki_s, ks_s, k_h_d + (int64_t) s * nbk3, nbk2, n_lid);
                } else {
                    dsv4_prequant_k_int8_kernel<float><<<pq_grid, pq_block, 0, stream>>>(
                            ki_s, ks_s, (k_force_f32 ? k_f32_d : (const float *) k->data) + (int64_t) s * nbk3, nbk2, n_lid);
                }
            }
        }
        for (int s = 0; s < n_stream; s++) {
            const int64_t t0    = (int64_t) s * nt_s;
            half        * sc_s  = scores_h + t0 * n_lid;
            const float * q_s   = q_d + t0 * n_head * d_idx;
            const float * w_s   = w_d + t0 * n_head;
            const float * m_s   = m_d + (int64_t) s * nbm3;
            if (fp4_mma_active) {
                const dim3 block_mma(32, 8); // threadIdx.x=lane, threadIdx.y=warp
                dsv4_score_fp4mma_kernel<<<grid, block_mma, 0, stream>>>(
                        sc_s, q_s, w_s,
                        (const uint8_t *) k->data + (int64_t) s * nbk3, m_s,
                        nbk2, nbm1, nt_s, n_lid, n_head);
            } else if (dsv4_lid_int8) {
                dsv4_score_int8_kernel<<<grid, block, 0, stream>>>(
                        sc_s, q_s, w_s,
                        k_i8 + (int64_t) s * n_lid * 128, k_sc + (int64_t) s * n_lid,
                        m_s, nbm1, nt_s, n_lid, n_head);
            } else if (k_is_f16) {
                dsv4_score_wmma128_kernel<half><<<grid, block, 0, stream>>>(
                        sc_s, q_s, w_s, k_h_d + (int64_t) s * nbk3, m_s, nbk2, nbm1, nt_s, n_lid, n_head);
            } else {
                dsv4_score_wmma128_kernel<float><<<grid, block, 0, stream>>>(
                        sc_s, q_s, w_s, (k_force_f32 ? k_f32_d : (const float *) k->data) + (int64_t) s * nbk3,
                        m_s, nbk2, nbm1, nt_s, n_lid, n_head);
            }
        }
    } else {
        const int j_tile  = 512;
        const int block   = 256;
        const size_t smem = ((size_t) d_idx*n_head + n_head) * sizeof(float);
        dim3 grid_score(nt, (n_lid + j_tile - 1) / j_tile, 1);
        if (k_is_f16) {
            dsv4_score_kernel<half><<<grid_score, block, smem, stream>>>(
                    scores, q_d, w_d, k_h_d, m_d,
                    nbk2, nbk3, nbm1, nbm3, nt, nt_s, n_lid, d_idx, n_head, j_tile);
        } else {
            dsv4_score_kernel<float><<<grid_score, block, smem, stream>>>(
                    scores, q_d, w_d, (k_force_f32 ? k_f32_d : (const float *) k->data), m_d,
                    nbk2, nbk3, nbm1, nbm3, nt, nt_s, n_lid, d_idx, n_head, j_tile);
        }
    }

    // output is contiguous [n_top_k, nt_s, 1, n_stream] == flat [nt * n_top_k]
    //
    // Exact mode (LLAMA_DSV4_LID_EXACT): two-pass selection. Pass 1 (whatever
    // score kernel ran above, int8/dec/wmma — all consuming the QAT-simulated
    // q/k, since exact implies the fp4 path) ranks to top-(n_top_k + m);
    // pass 2 rescores ONLY those candidates in serial-order fp32 (bitwise the
    // CPU reference / ds4.c order) and emits the exact top-n_top_k. Selection
    // becomes bit-exact vs the official QAT graph as long as m covers the
    // pass-1 rank displacement (oracle: p100=36 synthetic @n_lid 33k; default
    // m=64, override LLAMA_DSV4_LID_RESCORE_M).
    if (dsv4_lid_exact) {
        const int m      = dsv4_rescore_m;
        const int n_cand = (n_top_k + m) < n_lid ? (n_top_k + m) : n_lid;
        GGML_ASSERT(n_cand <= 1024); // rescore kernel smem/bitonic width
        ggml_cuda_pool_alloc<uint32_t> cand_alloc(pool, (size_t) nt * n_cand);
        ggml_cuda_pool_alloc<float>    cand_val_alloc(pool, (size_t) nt * n_cand);
        if (scores_half) {
            dsv4_topk_launch<half>(pool, cand_alloc.get(), scores_h, n_lid, nt, n_cand, stream);
        } else {
            dsv4_topk_launch<float>(pool, cand_alloc.get(), scores, n_lid, nt, n_cand, stream);
        }
        const dim3 grid_rs(nt, (n_cand + 127) / 128, 1);
        if (k_packed_direct || fp4_mma_active) {
            // packed cache, no staged buffer (decode packed-direct OR fp4-mma
            // prefill): dequant the candidates' 68B rows inline (byte strides).
            // Bitwise the CPU/ds4.c reference — the fp4-mma pass-1 only needs to
            // land the true top-k inside the m window (oracle p100<=4 << m=64).
            dsv4_lid_rescore_score_kernel<float, true><<<grid_rs, 128, 0, stream>>>(
                    cand_val_alloc.get(), cand_alloc.get(),
                    q_d, w_d, (const float *) k->data, m_d, nbk2, nbk3, nbm1, nbm3,
                    n_cand, n_lid, n_head, d_idx, nt_s);
        } else if ((dsv4_lid_qat_write || k_f16_d != nullptr) && k_is_f16) {
            // half arm: f16 cache under QAT_WRITE, or the packed-prefill
            // staged-f16 buffer (QAT by construction) — both bit-exact
            dsv4_lid_rescore_score_kernel<half><<<grid_rs, 128, 0, stream>>>(
                    cand_val_alloc.get(), cand_alloc.get(),
                    q_d, w_d, k_h_d, m_d, nbk2, nbk3, nbm1, nbm3,
                    n_cand, n_lid, n_head, d_idx, nt_s);
        } else if (dsv4_lid_qat_write && !k_force_f32) {
            // qat_write with an F32 cache: rows already QAT, read in place.
            // MUST NOT trigger when a staged buffer exists (packed MXFP4
            // cache) — k->data would be reinterpreted block bytes.
            dsv4_lid_rescore_score_kernel<float><<<grid_rs, 128, 0, stream>>>(
                    cand_val_alloc.get(), cand_alloc.get(),
                    q_d, w_d, (const float *) k->data, m_d, nbk2, nbk3, nbm1, nbm3,
                    n_cand, n_lid, n_head, d_idx, nt_s);
        } else {
            dsv4_lid_rescore_score_kernel<float><<<grid_rs, 128, 0, stream>>>(
                    cand_val_alloc.get(), cand_alloc.get(),
                    q_d, w_d, k_f32_d, m_d, nbk2, nbk3, nbm1, nbm3,
                    n_cand, n_lid, n_head, d_idx, nt_s);
        }
        dsv4_lid_rescore_select_kernel<<<nt, 256, 0, stream>>>(
                (int32_t *) dst->data, cand_alloc.get(), cand_val_alloc.get(), n_cand, n_top_k);
        return;
    }

    if (scores_half) {
        dsv4_topk_launch<half>(pool, (uint32_t *) dst->data, scores_h, n_lid, nt, n_top_k, stream);
    } else {
        dsv4_topk_launch<float>(pool, (uint32_t *) dst->data, scores, n_lid, nt, n_top_k, stream);
    }
}
