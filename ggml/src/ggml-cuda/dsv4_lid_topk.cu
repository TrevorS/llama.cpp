#include "dsv4_lid_topk.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "mma.cuh" // ggml_cuda_mma block-scaled fp4 tensor-core scoring

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

// Mask-probe early-out (variant i, LLAMA_DSV4_LID_MASK_PROBE=0 disables).
// A key row whose mask entry is -INF contributes
// -INF to the score no matter what the dot products are, so the score COMPUTE
// (K load + quant + the n_head-deep dot loop) is dead work. The probe reads the
// mask FIRST and, for masked rows, skips the compute but ALWAYS STORES -INF:
// the top-k scans the full [0, n_lid) row range, so a skipped store would leave
// stale scratch in the scores buffer and corrupt the selection.
//
// Bit-exactness: acc is a finite sum (relu(dot) * weight) and the mask carries
// only {0, -INF} — the same invariant the radix top-k already relies on — so
// acc + (-INF) == -INF, and writing -INF directly reproduces the exact encoding
// (f32 0xFF800000 / f16 0xFC00) the full-compute path stores. Comparison is
// against -INFINITY exactly, so a merely very-negative finite mask (alibi) never
// trips the early-out.
static __device__ __forceinline__ float dsv4_ldk(const float * p) { return *p; }
static __device__ __forceinline__ float dsv4_ldk(const half  * p) { return __half2float(*p); }

// One-shot loud report for a non-finite score. The whole cost of this class of
// bug is that it is SILENT: a NaN row reads downstream as 0% draft acceptance,
// not as a crash, so it looks like a quality regression rather than a fault.
// The atomic is only ever reached on the NaN path, so the guard is free when
// the scores are clean.
static __device__ int dsv4_nan_reported = 0;

static __device__ __forceinline__ void dsv4_report_nan() {
    if (atomicExch(&dsv4_nan_reported, 1) == 0) {
        printf("\nggml-cuda dsv4_lid_topk: NON-FINITE indexer score detected and "
               "clamped to -INF (this token's top-k is degraded; upstream q/k/weights/mask "
               "are suspect). Reported once per process.\n");
    }
}

// One-shot report for a tile union that did not fit u_cap.
//
// The B2 tile path is exact only while the union of the tile's W token top-k
// sets fits u_cap; past that dsv4_union_kernel drops the highest-index cells,
// so a query silently loses keys the indexer chose for it. It is structurally
// impossible only when u_cap >= W*top_k -- the shipped default (W=16, top_k=512,
// u_cap=4096) is a quarter of its exact width of 8192, and the measured union at
// d65k-d131k runs to a max around 5200 against a mean near 2400. So it does bite,
// on the tail of the tile distribution, and until now it did so without a trace.
//
// Reported rather than fixed by widening: exactness costs about 2x on the union
// half either way (W=8 doubles the tile count at the same u_cap; u_cap=8192
// doubles the width at the same tile count), so the quality/throughput call needs
// to be made against a real measurement of how often this fires in serving.
static __device__ int dsv4_union_overflow_reported = 0;

static __device__ __forceinline__ void dsv4_report_union_overflow(int u_max) {
    if (atomicExch(&dsv4_union_overflow_reported, 1) == 0) {
        printf("\nggml-cuda dsv4_lid_topk: tile union exceeded u_cap=%d; the highest-index "
               "selected cells were DROPPED for this tile (attention is missing keys the "
               "indexer chose). Raise LLAMA_DSV4_CSA_TILE_UCAP or lower LLAMA_DSV4_CSA_TILE. "
               "Reported once per process.\n", u_max);
    }
}

// scores-buffer load: the d_idx==128 score kernels store f16 (halves the
// [nt, n_lid] DRAM round-trip); the scalar path keeps f32 to stay bit-exact
// under its strict 0.0-tolerance gate. Sort compares stay f32 either way.
//
// NaN clamp: the score kernels' own invariant (finite acc + {0,-INF} mask) holds
// only while q/k/weights/mask are finite and acc does not overflow to +INF --
// acc==+INF with a -INF mask is NaN. An unclamped NaN never compares greater OR
// less, so which element wins a bitonic compare-exchange depends on evaluation
// order: selection becomes nondeterministic rather than merely wrong. Clamping
// to -INF makes a poisoned score behave exactly like a masked one.
static __device__ __forceinline__ float dsv4_lds(const float * p) {
    const float v = *p;
    if (v == v) return v;
    dsv4_report_nan();
    return -INFINITY;
}
static __device__ __forceinline__ float dsv4_lds(const half  * p) {
    const float v = __half2float(*p);
    if (v == v) return v;
    dsv4_report_nan();
    return -INFINITY;
}

// ---------------------------------------------------------------------------
// P3b packed MXFP4 lid container: 17B block-32 (e8m0 scale byte + 16 nibble
// bytes, ggml block_mxfp4 element order: low nibble -> j, high -> j+16).
// Rounding is the DSV4 QAT nearest-level search (dsv4_e2m1_index below), NOT
// ggml's stock quantizer. With level values stored at TRUE e2m1 magnitudes,
// e = s + 127 makes the dequant scale 2^(e-127) == the QAT scale, so the
// packed container holds the official post-QAT numerics exactly (pure
// power-of-two products).
// ---------------------------------------------------------------------------

// nibble -> TRUE e2m1 level (index 8 = -0 for sign fidelity with the
// fake-quant path); dequant value = DSV4_LV16[ni] * 2^(e-127)
static __constant__ float DSV4_LV16[16] = {  0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };

// amax floor for the QAT e8m0 scale: 6.0f*FLT_MIN, so amax/6 stays normal and
// ceilf(log2f(amax/6)) is finite (mirrors dsv4_qat_pack_row_cpu in ops.cpp)
#define DSV4_QAT_AMAX_MIN 7.052966104933725e-38f

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
        uint8_t * dst, const float * src,
        const void * idxs, const int idx_i64,
        int64_t nb1_dst, int64_t nb1_src, int ne0) {
    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    ggml_cuda_pdl_sync(); // wait on producers (src, idxs) before the first global read
    const int64_t idx = idx_i64 ? ((const int64_t *) idxs)[row]
                                : (int64_t) ((const int32_t *) idxs)[row];
    const float v = src[(int64_t) row * nb1_src + tid];
    float a = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    a = fmaxf(a, DSV4_QAT_AMAX_MIN);
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
        int nt, int nt_s, int n_lid, int d_idx, int n_head, int j_tile,
        bool mask_probe) {
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
        // mask probe: WARP-uniform (one warp owns one j, and s/t_local are
        // block-uniform), so every lane loads the same address — one broadcast
        // sector, the load the epilogue used to do — and the branch never
        // diverges within the warp.
        const float mv = mask[(int64_t)s*nbm3 + (int64_t)t_local*nbm1 + (int64_t)j];
        if (mask_probe && mv == -INFINITY) {
            if (lane == 0) {
                scores[(int64_t)t*n_lid + j] = mv; // store, never skip
            }
            continue;
        }
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
            scores[(int64_t)t*n_lid + j] = acc + mv;
        }
    }
}

// ---------------------------------------------------------------------------
// int8 dp4a score kernel (head_dim == 128)
//
// Supersedes the earlier fp16 wmma score kernel (removed; see git history),
// which was L1/shared-memory-bandwidth bound (ncu: 81% mem, 80% L1, 14%
// compute): the tensor cores sat idle while K was re-read from smem once per
// head (K is shared across heads, MLA-style). Storing K int8 in smem
// (16 KB vs 32 KB) halves that L1 traffic and lifts occupancy. dp4a runs on the
// idle int ALU. Per-row symmetric int8 quant of q and k; scales applied after
// the int32 accumulation. One CUDA-stream-group per launch (pointers pre-offset
// per stream group). 256 threads, one (16-token x 128-comp) tile per block:
// thread owns 1 token x 8 comps.
// ---------------------------------------------------------------------------

// Global int8 pre-quant of lid-K: one warp per comp (K-row), run once per
// stream before the score kernel. Layout and arithmetic match the per-row
// symmetric int8 quant the score kernel applies to K inline — strided dims
// d = lane + i*32, warp shfl_xor amax, round-to-nearest-even, per-comp scale
// amax/127 — so pre-quantizing here is bit-identical to quantizing in the
// score kernel, minus the ceil(nt_s/16)-fold redundant re-quant per token
// tile and the wider f16/f32 global reads.
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
        int n_tokens, int n_lid, int n_head, bool mask_probe) {
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

    const int my_t     = tid & 15;            // 0..15 token within tile
    const int my_c0    = (tid >> 4) * 8;      // 0,8,..,120 comp base
    const int token    = tile_t + my_t;
    const bool tok_ok  = token < n_tokens;

    // --- mask probe (early-out) ---------------------------------------------
    // The thread's 8 mask values are hoisted out of the epilogue (SAME global
    // traffic — the epilogue no longer reloads them; at depth this row is
    // nt_s*n_lid*4 B, so re-reading it would not be free) and kept in
    // registers across the head loop. Two granularities:
    //   block-uniform — if all 16x128 mask entries are -INF the tile is dead:
    //       return before the 16 KB K-tile smem load and the whole head loop.
    //       __syncthreads_and gives a block-uniform predicate, so every thread
    //       takes the same branch (no divergence, no orphaned barrier).
    //   per-(thread,comp) — partial tiles (the causal diagonal band) skip just
    //       the masked comps' dp4a chains. The mask is per-(query,key) and one
    //       thread owns 8 DIFFERENT keys of one query, so there is no coarser
    //       uniform unit here; the skip rides the already-serial cc loop, so a
    //       diverged warp costs at most the union of its lanes' live comps.
    // acc[cc] stays 0.0f for a skipped comp and the epilogue stores
    // acc[cc] + mv[cc] == 0 + (-INF) == -INF: the exact bit pattern the
    // full-compute path would have stored.
    float mv[8];
    bool  all_masked = true;
#pragma unroll
    for (int cc = 0; cc < 8; cc++) {
        const int comp = tile_c + my_c0 + cc;
        mv[cc] = (tok_ok && comp < n_lid) ? mask[(int64_t) token * nbm1 + comp] : -INFINITY;
        all_masked = all_masked && (mv[cc] == -INFINITY);
    }
    if (__syncthreads_and(all_masked) && mask_probe) {
        if (tok_ok) {
#pragma unroll
            for (int cc = 0; cc < 8; cc++) {
                const int comp = tile_c + my_c0 + cc;
                if (comp < n_lid) {
                    scores[(int64_t) token * n_lid + comp] = __float2half(mv[cc]); // store, never skip
                }
            }
        }
        return;
    }

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
            if (mask_probe && mv[cc] == -INFINITY) {
                continue; // masked comp: acc[cc] stays 0 -> epilogue stores -INF
            }
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
                scores[(int64_t) token * n_lid + comp] = __float2half(acc[cc] + mv[cc]);
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
        int64_t, int64_t, int, int, int, bool) { NO_DEVICE_CODE; }
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
        int n_tokens, int n_lid, int n_head, bool mask_probe) {
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

    // ---- mask probe (early-out) --------------------------------------------
    // The mma produces a whole 16-token x 8-comp tile at once, so the finest
    // skippable unit is one ct tile — which is also exactly 128 mask entries =
    // 4 per lane, i.e. one __all_sync makes the skip WARP-uniform (no
    // divergence, unlike a per-element test that the mma could not honour
    // anyway). A skipped ct drops its 2 mma ops + relu drain for EVERY head and
    // leaves acc[ct][] at 0, which the epilogue turns into 0 + (-INF) == the
    // same -INF the full path stores. If every ct in the block is masked the
    // tile is dead: return before the K pack, the resident B fragment loads and
    // the head loop (the per-head q pack is the dominant cost). Probe layout is
    // independent of the epilogue's mma lane map — it only has to COVER the
    // same 16x8 rectangle, which lane*4+i does.
    // Unlike the int8 kernel this does NOT hoist the mask values into registers
    // for the epilogue: K lives in registers here (Bk[2][2] is the whole point
    // of this kernel), so 8 more live floats across the head loop is the wrong
    // trade. Only the 2 skip bits survive, and the epilogue re-reads the mask —
    // but only for the ct's that were NOT skipped, since a skipped ct already
    // knows its tile is -INF. So a fully-masked block reads the mask once, and
    // only a mixed block pays the second read.
    bool skip_ct[2];
    #pragma unroll
    for (int ct = 0; ct < 2; ct++) {
        const int cbase = tile_c + warp * 16 + ct * 8;
        bool am = true;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int idx  = lane * 4 + i;        // 0..127 over the ct tile
            const int tok  = tile_t + (idx >> 3); // 16 tokens
            const int comp = cbase + (idx & 7);   //  8 comps
            if (tok < n_tokens && comp < n_lid &&
                mask[(int64_t) tok * nbm1 + comp] != -INFINITY) {
                am = false;
            }
        }
        skip_ct[ct] = __all_sync(0xffffffffu, am) && mask_probe;
    }
    if (__syncthreads_and(skip_ct[0] && skip_ct[1])) {
        #pragma unroll
        for (int ct = 0; ct < 2; ct++) {
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                const int tok  = tile_t + ((l / 2) * 8) + (lane / 4);
                const int comp = tile_c + warp * 16 + ct * 8 + ((lane % 4) * 2) + (l % 2);
                if (tok < n_tokens && comp < n_lid) {
                    scores[(int64_t) tok * n_lid + comp] = __float2half(-INFINITY); // store, never skip
                }
            }
        }
        return;
    }

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
            amax = fmaxf(amax, DSV4_QAT_AMAX_MIN);
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
            if (skip_ct[ct]) {
                continue; // masked comp-tile: acc stays 0 -> epilogue stores -INF
            }
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
                // a skipped ct already proved every entry of its tile is -INF,
                // so it stores without re-reading the mask (acc is still 0)
                const float mv = skip_ct[ct] ? -INFINITY : mask[(int64_t) tok * nbm1 + comp];
                scores[(int64_t) tok * n_lid + comp] = __float2half(acc[ct][l] + mv);
            }
        }
    }
#else
    GGML_UNUSED_VARS(scores, q, weights, k, mask, nbk2, nbm1, n_tokens, n_lid, n_head, mask_probe);
    NO_DEVICE_CODE;
#endif // BLACKWELL_MMA_AVAILABLE
}
#endif // GGML_USE_HIP

// ---------------------------------------------------------------------------
// dedicated decode score kernel (nt==1, head_dim==128) — LLAMA_DSV4_LID_DEC
//
// At decode the int8 tile path pads 1 token to a 16-token tile (15/16 wasted)
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
        half        * scores,   // [n_lid] (f16 store)
        const float * q,        // [h*128 + d]  (single token)
        const float * weights,  // [h]
        const KT    * k,        // [j*nbk2 + d]
        const float * mask,     // [j]
        int64_t nbk2, int n_lid, int n_head, bool mask_probe) {
    extern __shared__ char smem_dec[];
    int8_t * qs   = (int8_t *) smem_dec;                 // [n_head*128]
    float  * sq   = (float *) (qs + (size_t) n_head * 128); // [n_head]
    float  * sw   = sq + n_head;                          // [n_head]

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int nwarps = blockDim.x >> 5;

    ggml_cuda_pdl_sync(); // wait on producers (q, weights, k, mask) before the first global read
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
        // mask probe: WARP-uniform (one warp owns one comp j), every lane loads
        // the same address. Skips the whole 68B/256B K row read + quant + the
        // n_head dp4a chain; the store still happens.
        const float mv = mask[j];
        if (mask_probe && mv == -INFINITY) {
            if (lane == 0) {
                scores[j] = __float2half(mv); // store, never skip
            }
            continue;
        }
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
        if (lane == 0) scores[j] = __float2half(acc + mv);
    }
}

// ---------------------------------------------------------------------------
// B2 per-tile union + membership (one block per (tile, stream), smem bitmap
// over n_csa; W tokens per tile, last tile may be partial)
// ---------------------------------------------------------------------------

static __global__ void dsv4_union_kernel(
        int32_t * out, const int32_t * top_k,
        int64_t nb1_tk, int64_t nb3_tk, int64_t nb1_out, int64_t nb3_out,
        int n_top_k, int nt_s, int n_csa, int u_max, int W) {
    const int tile = blockIdx.x;
    const int s    = blockIdx.y;
    const int t0   = tile * W;
    const int t1   = min(t0 + W, nt_s);
    extern __shared__ uint32_t bm[];
    const int n_words = (n_csa + 31) / 32;
    for (int i = threadIdx.x; i < n_words; i += blockDim.x) bm[i] = 0;
    __syncthreads();
    ggml_cuda_pdl_sync(); // wait on producer (top_k) before the first global read
    const int total = n_top_k * (t1 - t0);
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int t = t0 + idx / n_top_k, i = idx % n_top_k;
        const int c = top_k[(int64_t) i + t * nb1_tk + (int64_t) s * nb3_tk];
        if (c >= 0 && c < n_csa) atomicOr(&bm[c >> 5], 1u << (c & 31));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int32_t * o = out + (int64_t) tile * nb1_out + (int64_t) s * nb3_out;
        int pos = 0;
        int w   = 0;
        for (; w < n_words && pos < u_max; w++) {
            uint32_t word = bm[w];
            while (word && pos < u_max) {
                const int b = __ffs(word) - 1;
                o[pos++] = w * 32 + b;
                word &= word - 1;
            }
            // ran out of slots with bits still set in this word
            if (word) {
                dsv4_report_union_overflow(u_max);
                break;
            }
        }
        // ...or with whole words still to scan
        if (pos == u_max) {
            for (; w < n_words; w++) {
                if (bm[w]) {
                    dsv4_report_union_overflow(u_max);
                    break;
                }
            }
        }
        // pad with n_csa-1: the MAXIMUM representable index, so ascending order
        // is preserved and dsv4_memb_kernel's lower_bound (first match) resolves
        // each index to exactly one slot — if n_csa-1 is genuinely in the union
        // it sits at the last real slot and wins the search; if not, no query
        // selected it and the pad slots are never unmasked. Padding with any
        // OTHER value (or unsorted) would double-count via duplicate matches.
        for (; pos < u_max; pos++) o[pos] = n_csa - 1;
    }
}

// One block per (token-group, stream). Uses union_idx of the token's tile
// directly (binary search) so the membership rank is GUARANTEED consistent
// with the gather order. The tile's union is cached in smem and reloaded when
// the token group crosses a tile boundary (never happens for W % TPB == 0).
#define DSV4_MEMB_TPB 8
static __global__ void dsv4_memb_kernel(
        float * memb, const int32_t * top_k,
        const int32_t * union_idx,
        int64_t nb1_tk, int64_t nb3_tk, int64_t nb1_m, int64_t nb3_m, int64_t nb1_u, int64_t nb3_u,
        int n_top_k, int nt_s, int n_csa, int u_max, int W) {
    const int s      = blockIdx.y;
    const int t_base = blockIdx.x * DSV4_MEMB_TPB;
    extern __shared__ int32_t uni_s[]; // [u_max] union_idx for the current tile
    int cur_tile = -1;

    ggml_cuda_pdl_sync(); // wait on producers (top_k, union_idx) before the first global read
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
    const ggml_cuda_kernel_launch_params lp = { dim3((unsigned) n_rows), dim3((unsigned) ne0), 0, ctx.stream() };
    ggml_cuda_kernel_launch(dsv4_qat_set_rows_kernel, lp,
            (uint8_t *) dst->data, (const float *) b->data, c->data,
            c->type == GGML_TYPE_I64 ? 1 : 0,
            dst->nb[1], b->nb[1] / sizeof(float), (int) ne0);
}

// merge two flash_attn_ext_with_lse results computed over disjoint KV subsets
// (GGML_OP_DSV4_FA_MERGE): out = (ea*a + eb*b)/(ea+eb) per row,
// ea = exp(lse_a - max(lse_a, lse_b)). srcs [DV, H, Q, S+1] with LSE tail at
// element offset DV*H*Q*S, tail idx == row idx; dst [DV, H, Q, S].
static __global__ void dsv4_fa_merge_kernel(
        const float * a, const float * b,
        float * dst, const int DV, const int64_t n_rows) {
    const int64_t r = blockIdx.x;

    ggml_cuda_pdl_sync(); // wait on producers (a, b FA sub-results) before the first global read
    const float la = a[DV*n_rows + r];
    const float lb = b[DV*n_rows + r];
    const float m  = fmaxf(la, lb);

    float wa = 0.0f;
    float wb = 0.0f;
    if (m != -INFINITY) { // else both parts fully masked -> zeros
        const float ea = expf(la - m);
        const float eb = expf(lb - m);
        const float es = ea + eb;
        wa = ea / es;
        wb = eb / es;
    }

    const float * ar = a   + r*DV;
    const float * br = b   + r*DV;
    float       * dr = dst + r*DV;

    // a part with lse == -inf (fully masked, weight exactly 0) may carry
    // NaN/uninitialized values — skip it entirely, never multiply by 0
    for (int d = threadIdx.x; d < DV; d += blockDim.x) {
        float v = 0.0f;
        if (wa != 0.0f) { v += wa*ar[d]; }
        if (wb != 0.0f) { v += wb*br[d]; }
        dr[d] = v;
    }
}

void ggml_cuda_op_dsv4_fa_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];

    GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(dst));

    const int64_t DV     = dst->ne[0];
    const int64_t n_rows = dst->ne[1]*dst->ne[2]*dst->ne[3];
    if (n_rows == 0) {
        return;
    }

    const int n_threads = DV >= 256 ? 256 : 128;
    const ggml_cuda_kernel_launch_params lp = { dim3((unsigned) n_rows), dim3((unsigned) n_threads), 0, ctx.stream() };
    ggml_cuda_kernel_launch(dsv4_fa_merge_kernel, lp,
            (const float *) a->data, (const float *) b->data, (float *) dst->data,
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

    const ggml_cuda_kernel_launch_params lp = { grid, dim3(256), smem, ctx.stream() };
    ggml_cuda_kernel_launch(dsv4_union_kernel, lp,
        (int32_t *) dst->data, (const int32_t *) top_k->data,
        nb1_tk, nb3_tk, nb1_out, nb3_out, n_top_k, nt_s, n_csa, u_max, W);
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
    const ggml_cuda_kernel_launch_params lp = { grid, dim3(256), smem, ctx.stream() };
    ggml_cuda_kernel_launch(dsv4_memb_kernel, lp,
        (float *) dst->data, (const int32_t *) top_k->data, (const int32_t *) uni->data,
        nb1_tk, nb3_tk, nb1_m, nb3_m, nb1_u, nb3_u, n_top_k, nt_s, n_csa, u_max, W);
}

// ---------------------------------------------------------------------------
// top-k (bitonic, descending, lower-index tie-break) — ported from ds4_cuda.cu
// ---------------------------------------------------------------------------

static __device__ __forceinline__ bool dsv4_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

// one bitonic compare-exchange step: lane i acts against partner i^j (only the
// lower index of each pair), keeping the dsv4_better winner in the slot dictated
// by desc_half. __forceinline__ so each caller's loops unroll exactly as before.
static __device__ __forceinline__ void dsv4_ce_step(
        float * vals, uint32_t * idxs, uint32_t i, uint32_t j, uint32_t n, bool desc_half) {
    const uint32_t other = i ^ j;
    if (other > i && other < n) {
        const float    av = vals[i], bv = vals[other];
        const uint32_t ai = idxs[i], bi = idxs[other];
        const bool swap = desc_half ? dsv4_better(bv, bi, av, ai)
                                    : dsv4_better(av, ai, bv, bi);
        if (swap) {
            vals[i] = bv; idxs[i] = bi;
            vals[other] = av; idxs[other] = ai;
        }
    }
}

template <uint32_t SORT_N>
static __device__ void dsv4_bitonic_sort(float * vals, uint32_t * idxs) {
    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = threadIdx.x; i < SORT_N; i += blockDim.x) {
                dsv4_ce_step(vals, idxs, i, j, SORT_N, (i & k) == 0u);
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
                dsv4_ce_step(vals, idxs, i, j, SORT_N, (k == K) ? true : ((i & k) == 0u));
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
    ggml_cuda_pdl_sync(); // wait on producer (scores) before the first global read
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
// WHICH ==T members win. A NaN should not be reachable (finite acc + {0, -INF}
// mask), but that invariant is conditional on upstream finiteness and on acc
// not overflowing to +INF, so dsv4_score_key16 clamps rather than trusting it —
// see the note there for why an unclamped NaN is worse here than elsewhere.
// ---------------------------------------------------------------------------

// Order-preserving f16 -> u16 key. Keys run 0x0000 (most negative) to 0xFFFF.
//
// NaN clamp: a NaN half is exp==0x1F with a nonzero mantissa, so a POSITIVE NaN
// (0x7C01-0x7FFF) keys to 0xFC01-0xFFFF — strictly ABOVE +INF's 0xFC00. Left
// alone it would be the single highest key in the row and would be selected
// into the top-k on every token, silently evicting a real candidate.
//
// Canonicalize to the -INF bit pattern rather than to key 0: that lands on the
// same key -INF gets (0x03FF) and so matches BOTH the bitonic path (dsv4_lds
// returns -INFINITY) and the CPU reference (same clamp). Sorting NaN strictly
// below -INF instead would make the radix and bitonic paths disagree with each
// other whenever selection has to reach into the masked tail.
static __device__ __forceinline__ uint16_t dsv4_score_key16(const half h) {
    uint16_t hb = __half_as_ushort(h);
    if ((hb & 0x7C00u) == 0x7C00u && (hb & 0x03FFu) != 0u) {
        dsv4_report_nan();
        hb = 0xFC00u; // -INF
    }
    if (hb == 0x8000u) hb = 0u; // -0 == +0 under f32 compare
    return (hb & 0x8000u) ? (uint16_t) ~hb : (uint16_t) (hb | 0x8000u);
}

// runtime-width variant of dsv4_bitonic_sort (n = pow2 <= 4096); sorts the
// selected candidates descending for the output contract
static __device__ void dsv4_bitonic_sort_rt(float * vals, uint32_t * idxs, uint32_t n) {
    for (uint32_t k = 2u; k <= n; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
                dsv4_ce_step(vals, idxs, i, j, n, (i & k) == 0u);
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
    // Decode (n_tokens<16) historically kept bitonic to fill the GPU with many
    // token-blocks. But radix is O(n_lid) while the bitonic chunk+merge tree is
    // O(n_lid log^2 SORT_N), so on a wide row even a single-token radix block
    // wins big. Allow decode radix above the single-chunk bitonic capacity.
    // LLAMA_DSV4_LID_RADIX=0 reverts everything to bitonic (the A/B baseline).
    static const int radix_dec_min = []() {
        const char * e = getenv("LLAMA_DSV4_LID_RADIX_DEC_MIN");
        return e ? atoi(e) : (int) DSV4_TOPK_SORT_N;
    }();
    const bool allow = (n_tokens >= 16) || (n_lid > radix_dec_min); // prefill, or wide decode row
    if (!radix_on || n_lid <= top_k || !allow) {
        return false;
    }
    dsv4_topk_radix_launch(selected, scores, n_lid, n_tokens, top_k, stream);
    return true;
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
        const ggml_cuda_kernel_launch_params lp = { dim3(n_tokens), dim3(block), smem, stream };
        ggml_cuda_kernel_launch(dsv4_topk_single_kernel<SORT_N, score_t>, lp,
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
    // CUDA-graph capture invariant: this RAII scratch is released back to the
    // pool on return while a captured graph keeps the raw pointer. Safe only
    // because (a) the size derives from tensor ne (covered by the property
    // scan) and (b) every scratch buffer is WRITTEN by a kernel inside the
    // same captured graph before any kernel reads it. A scratch whose producer
    // runs outside the captured graph would silently break replay.
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

    ggml_cuda_pool_alloc<half>  k_f16_alloc(pool); // packed-prefill staging (f16-of-QAT, bit-exact)
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
    // DEPTH GATE (2026-07-16, GB10 wedges #9-#11): sustained fp4-mma at the
    // d131k shape (n_lid 33280) hard-locks the box even under a perfectly
    // functioning 75% duty cycle (burst draw latches the firmware power cap;
    // telemetry in experiments/profiles/wedge-hunt/). 1-for-4 at d131k vs
    // dozens of clean runs at <= d65k shapes -> fall back to int8 above the
    // cap. Default 24576 clears every 65k-class shape and blocks 131k+.
    static const int64_t dsv4_lid_fp4_mma_max_nlid = []() {
        const char * e = getenv("LLAMA_DSV4_LID_FP4_MMA_MAX_NLID");
        return e ? atoll(e) : (long long) 24576;
    }();
    // mask-probe early-out (see the note above dsv4_ldk): masked key rows skip
    // the score compute but still store -INF. Bit-exact, so default ON;
    // LLAMA_DSV4_LID_MASK_PROBE=0 disables. This used to also answer to
    // DS4_CUDA_NO_MASK_PROBE, a vestige of the ds4_cuda.cu port lineage —
    // two names for one switch, referenced by nothing outside this file.
    static const bool dsv4_lid_mask_probe = []() {
        const char * e = getenv("LLAMA_DSV4_LID_MASK_PROBE");
        return !e || e[0] != '0';
    }();
    bool fp4_mma_active = dsv4_lid_fp4_mma && k_is_mxfp4 && d_idx == 128 && nt_s > 1 &&
                          n_lid <= dsv4_lid_fp4_mma_max_nlid;
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
    }
    const half  * k_f16_d = k_f16_alloc.get(); // non-null only for packed-prefill staging
    const bool k_is_f16 = (k->type == GGML_TYPE_F16) || k_f16_d != nullptr;
    // half-arm K pointer: staged buffer wins over the raw cache
    const half * k_h_d = k_f16_d ? k_f16_d : (const half *) k->data;

    // Tiled paths require head_dim == 128 (16-token x 128-comp tiling).
    // One launch per CUDA stream-group so a 16-token tile never straddles a
    // stream boundary; falls back to the scalar dot-product kernel otherwise.
    if (d_idx == 128 && dsv4_lid_dec && nt_s == 1) {
        const int block   = 256;
        const int nwarps  = block / 32;
        const size_t smem = (size_t) n_head * 128 + (size_t) n_head * 2 * sizeof(float);
        int gx = (n_lid + nwarps - 1) / nwarps;
        if (gx > 512) gx = 512;
        const ggml_cuda_kernel_launch_params lp = { dim3(gx), dim3(block), smem, stream };
        for (int s = 0; s < n_stream; s++) {
            half        * sc_s = scores_h + (int64_t) s * nt_s * n_lid;
            const float * q_s  = q_d + (int64_t) s * nt_s * n_head * d_idx;
            const float * w_s  = w_d + (int64_t) s * nt_s * n_head;
            const float * m_s  = m_d + (int64_t) s * nbm3;
            if (k_packed_direct) {
                ggml_cuda_kernel_launch(dsv4_score_decode_kernel<float, true>, lp,
                        sc_s, q_s, w_s, (const float *) ((const uint8_t *) k->data + (int64_t) s * nbk3), m_s, nbk2, n_lid, n_head, dsv4_lid_mask_probe);
            } else if (k_is_f16) {
                ggml_cuda_kernel_launch(dsv4_score_decode_kernel<half>, lp,
                        sc_s, q_s, w_s, k_h_d + (int64_t) s * nbk3, m_s, nbk2, n_lid, n_head, dsv4_lid_mask_probe);
            } else {
                ggml_cuda_kernel_launch(dsv4_score_decode_kernel<float>, lp,
                        sc_s, q_s, w_s, (const float *) k->data + (int64_t) s * nbk3, m_s, nbk2, n_lid, n_head, dsv4_lid_mask_probe);
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
        if (!fp4_mma_active) {
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
                            ki_s, ks_s, (const float *) k->data + (int64_t) s * nbk3, nbk2, n_lid);
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
                        nbk2, nbm1, nt_s, n_lid, n_head, dsv4_lid_mask_probe);
            } else {
                dsv4_score_int8_kernel<<<grid, block, 0, stream>>>(
                        sc_s, q_s, w_s,
                        k_i8 + (int64_t) s * n_lid * 128, k_sc + (int64_t) s * n_lid,
                        m_s, nbm1, nt_s, n_lid, n_head, dsv4_lid_mask_probe);
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
                    nbk2, nbk3, nbm1, nbm3, nt, nt_s, n_lid, d_idx, n_head, j_tile, dsv4_lid_mask_probe);
        } else {
            dsv4_score_kernel<float><<<grid_score, block, smem, stream>>>(
                    scores, q_d, w_d, (const float *) k->data, m_d,
                    nbk2, nbk3, nbm1, nbm3, nt, nt_s, n_lid, d_idx, n_head, j_tile, dsv4_lid_mask_probe);
        }
    }

    // output is contiguous [n_top_k, nt_s, 1, n_stream] == flat [nt * n_top_k]
    if (scores_half) {
        dsv4_topk_launch<half>(pool, (uint32_t *) dst->data, scores_h, n_lid, nt, n_top_k, stream);
    } else {
        dsv4_topk_launch<float>(pool, (uint32_t *) dst->data, scores, n_lid, nt, n_top_k, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}
