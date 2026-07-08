#include "dsv4_moe_gate_up.cuh"

#include <cstdint>
#include <cstring>

// ============================================================================
// DeepSeek-V4 MoE prefill gate+up+activation fused tile op (IQ2_XXS).
//
// Ports the ds4.c expert-tile pipeline (ds4_cuda.cu) into a ggml CUDA op:
//   1. q8_K-quantize the activations x[n_tokens x n_embd]
//   2. count / prefix / scatter routed (token,slot) pairs by expert
//   3. build expert tiles (block_m = 8)
//   4. gate_up_mid IQ2_XXS tile8 rowspan kernel producing
//        mid[pair][r] = silu(clamp(gate)) * clamp(up)
//      (the routing weight is intentionally NOT applied here; ggml applies it
//       after the down matmul, exactly as in the unfused path).
//
// All device helpers / kernels are lifted verbatim from ds4_cuda.cu (via the
// experiments/ds4-tile extraction) except the gate_up kernel, which drops the
// weight multiply and the write_aux gate/up outputs.
// ============================================================================

#define DSV4_QK_K 256
#define DSV4_MG_ROW_SPAN 512u

// ---- block struct definitions (bit-identical to ggml quantized blocks) ----
typedef struct {
    uint16_t d;
    uint16_t qs[DSV4_QK_K / 8];
} dsv4_block_iq2_xxs;

typedef struct {
    float   d;
    int8_t  qs[DSV4_QK_K];
    int16_t bsums[DSV4_QK_K / 16];
} dsv4_block_q8_K;

static_assert(sizeof(dsv4_block_iq2_xxs) == sizeof(block_iq2_xxs), "iq2_xxs block size mismatch");
static_assert(sizeof(dsv4_block_q8_K)    == sizeof(block_q8_K),    "q8_K block size mismatch");

#include "dsv4_moe_iq2_tables.cuh"

// ---- device helpers (ds4_cuda.cu:10354-10511) ----
__device__ static float dsv4_f16_to_f32(uint16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}

__device__ __forceinline__ static uint32_t dsv4_unpack_iq2_signs(uint32_t v) {
    const uint32_t p = __popc(v) & 1u;
    const uint32_t s = v ^ (p << 7u);
    return s * 0x01010101u;
}

__device__ __forceinline__ static void dsv4_iq2_i8x8_lut(
        const uint64_t *grid, const uint8_t *signs,
        uint8_t grid_idx, uint32_t sign_idx, int32_t *w0, int32_t *w1) {
    const uint32_t s = dsv4_unpack_iq2_signs(signs[sign_idx]);
    const int32_t sm0 = __vcmpne4(s & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(s & 0x80402010u, 0);
    const uint64_t g = grid[grid_idx];
    *w0 = __vsub4((int32_t)(uint32_t)g ^ sm0, sm0);
    *w1 = __vsub4((int32_t)(uint32_t)(g >> 32) ^ sm1, sm1);
}

// 8-way batched IQ2_XXS . q8_K dot with a shared-memory codebook (block8_deq_lut).
__device__ static void dsv4_dot_iq2_xxs_q8_K_block8(
        const dsv4_block_iq2_xxs *x,
        const dsv4_block_q8_K *y0, const dsv4_block_q8_K *y1,
        const dsv4_block_q8_K *y2, const dsv4_block_q8_K *y3,
        const dsv4_block_q8_K *y4, const dsv4_block_q8_K *y5,
        const dsv4_block_q8_K *y6, const dsv4_block_q8_K *y7,
        uint32_t n, float acc[8],
        const uint64_t *grid, const uint8_t *signs) {
    const float xd = dsv4_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t *q8[8] = {
        y0 ? y0->qs : nullptr, y1 ? y1->qs : nullptr, y2 ? y2->qs : nullptr, y3 ? y3->qs : nullptr,
        y4 ? y4->qs : nullptr, y5 ? y5->qs : nullptr, y6 ? y6->qs : nullptr, y7 ? y7->qs : nullptr,
    };
    for (int ib32 = 0; ib32 < DSV4_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dsv4_iq2_i8x8_lut(grid, signs, (uint8_t)(aux0 & 0xffu),         (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dsv4_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu), (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dsv4_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu), (aux1 >> 14) & 127u, &w[4], &w[5]);
        dsv4_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu), (aux1 >> 21) & 127u, &w[6], &w[7]);
        for (uint32_t p = 0; p < n; p++) {
            const int8_t *q = q8[p] + ib32 * 32;
            int32_t sumi = 0;
            sumi = __dp4a(w[0], *(const int32_t *)(q + 0),  sumi);
            sumi = __dp4a(w[1], *(const int32_t *)(q + 4),  sumi);
            sumi = __dp4a(w[2], *(const int32_t *)(q + 8),  sumi);
            sumi = __dp4a(w[3], *(const int32_t *)(q + 12), sumi);
            sumi = __dp4a(w[4], *(const int32_t *)(q + 16), sumi);
            sumi = __dp4a(w[5], *(const int32_t *)(q + 20), sumi);
            sumi = __dp4a(w[6], *(const int32_t *)(q + 24), sumi);
            sumi = __dp4a(w[7], *(const int32_t *)(q + 28), sumi);
            bsum[p] += sumi * ls;
        }
    }
    const dsv4_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static float dsv4_quarter_warp_sum_f32(float v) {
    uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (int offset = 4; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 8);
    }
    return v;
}

// ---- q8_K quantize (ds4_cuda.cu:10908-10955) ----
__global__ static void dsv4_q8_K_quantize_kernel(dsv4_block_q8_K *out, const float *x, uint32_t in_dim, uint32_t n_rows) {
    uint32_t b = blockIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / DSV4_QK_K) return;
    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * DSV4_QK_K;
    dsv4_block_q8_K *yb = out + (uint64_t)row * (in_dim / DSV4_QK_K) + b;
    __shared__ float abs_part[256];
    __shared__ float val_part[256];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    uint32_t tid = threadIdx.x;
    float v = tid < DSV4_QK_K ? xr[tid] : 0.0f;
    abs_part[tid] = tid < DSV4_QK_K ? fabsf(v) : 0.0f;
    val_part[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    float amax = abs_part[0];
    if (amax == 0.0f) {
        if (tid == 0) yb->d = 0.0f;
        if (tid < DSV4_QK_K) yb->qs[tid] = 0;
        if (tid < DSV4_QK_K / 16) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0) {
        maxv_s = val_part[0];
        iscale_s = -127.0f / maxv_s;
    }
    __syncthreads();
    if (tid < DSV4_QK_K) {
        int qv = (int)lrintf(iscale_s * xr[tid]);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        yb->qs[tid] = (int8_t)qv;
    }
    __syncthreads();
    if (tid < DSV4_QK_K / 16) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0) yb->d = 1.0f / iscale_s;
}

// ---- routing bridge: build ds4 sorted-pair tiles from the ggml ids tensor ----
// (ds4_cuda.cu:11228-11300; "selected" == ggml ids [n_expert_used, n_tokens] I32)
__global__ static void dsv4_count_sorted_pairs_kernel(uint32_t *counts, const int32_t *selected, uint32_t pair_count) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t e = selected[pair];
    if (e < 0) e = 0;
    atomicAdd(counts + (uint32_t)e, 1u);
}

__global__ static void dsv4_prefix_sorted_pairs_kernel(uint32_t *offsets, uint32_t *cursors, const uint32_t *counts, uint32_t expert_count) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < expert_count; e++) {
            offsets[e] = sum;
            cursors[e] = sum;
            sum += counts[e];
        }
        offsets[expert_count] = sum;
    }
}

__global__ static void dsv4_scatter_sorted_pairs_kernel(uint32_t *sorted_pairs, uint32_t *cursors, const int32_t *selected, uint32_t pair_count) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t e = selected[pair];
    if (e < 0) e = 0;
    uint32_t pos = atomicAdd(cursors + (uint32_t)e, 1u);
    sorted_pairs[pos] = pair;
}

__global__ static void dsv4_build_tile_offsets_kernel(uint32_t *tile_offsets, uint32_t *tile_total, const uint32_t *counts, uint32_t expert_count, uint32_t block_m) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < expert_count; e++) {
            tile_offsets[e] = sum;
            sum += (counts[e] + block_m - 1u) / block_m;
        }
        tile_offsets[expert_count] = sum;
        *tile_total = sum;
    }
}

__global__ static void dsv4_build_tiles_kernel(uint32_t *tile_experts, uint32_t *tile_starts, const uint32_t *tile_offsets, const uint32_t *counts, uint32_t expert_count, uint32_t block_m) {
    uint32_t e = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= expert_count) return;
    uint32_t ntiles = (counts[e] + block_m - 1u) / block_m;
    uint32_t off = tile_offsets[e];
    for (uint32_t t = 0; t < ntiles; t++) {
        tile_experts[off + t] = e;
        tile_starts[off + t] = t * block_m;
    }
}

// ---- gate+up+activation tile kernel (ds4_cuda.cu:11677-11768, IQ2_XXS, no weight) ----
template <uint32_t ROW_SPAN>
__global__ static void dsv4_gate_up_mid_iq2_rowspan_kernel(
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const dsv4_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert_used,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ dsv4_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const dsv4_block_q8_K *xqb[8] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        uint32_t tok = pair[np] / n_expert_used;
        xqb[np] = xq + (uint64_t)tok * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    } else {
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const dsv4_block_iq2_xxs *gr = (const dsv4_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const dsv4_block_iq2_xxs *ur = (const dsv4_block_iq2_xxs *)(up_base   + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8]   = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dsv4_dot_iq2_xxs_q8_K_block8(gr + b, xqb[0] ? xqb[0] + b : nullptr, xqb[1] ? xqb[1] + b : nullptr,
                                         xqb[2] ? xqb[2] + b : nullptr, xqb[3] ? xqb[3] + b : nullptr,
                                         xqb[4] ? xqb[4] + b : nullptr, xqb[5] ? xqb[5] + b : nullptr,
                                         xqb[6] ? xqb[6] + b : nullptr, xqb[7] ? xqb[7] + b : nullptr, np, gate,
                                         s_iq2_grid, s_iq2_signs);
            dsv4_dot_iq2_xxs_q8_K_block8(ur + b, xqb[0] ? xqb[0] + b : nullptr, xqb[1] ? xqb[1] + b : nullptr,
                                         xqb[2] ? xqb[2] + b : nullptr, xqb[3] ? xqb[3] + b : nullptr,
                                         xqb[4] ? xqb[4] + b : nullptr, xqb[5] ? xqb[5] + b : nullptr,
                                         xqb[6] ? xqb[6] + b : nullptr, xqb[7] ? xqb[7] + b : nullptr, np, up,
                                         s_iq2_grid, s_iq2_signs);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = dsv4_quarter_warp_sum_f32(gate[p]);
            up[p]   = dsv4_quarter_warp_sum_f32(up[p]);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p];
            }
        }
    }
}

// ============================================================================
// host op
// ============================================================================
void ggml_cuda_op_dsv4_moe_gate_up(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * gate = dst->src[0];
    const ggml_tensor * up   = dst->src[1];
    const ggml_tensor * cur  = dst->src[2];   // F32 activations [n_embd, .., n_tokens]
    const ggml_tensor * ids  = dst->src[3];   // I32 [n_expert_used, n_tokens]

    GGML_ASSERT(gate->type == GGML_TYPE_IQ2_XXS && up->type == GGML_TYPE_IQ2_XXS);
    GGML_ASSERT(cur->type  == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(cur));
    GGML_ASSERT(ggml_is_contiguous(ids));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const uint32_t n_embd        = (uint32_t) gate->ne[0];
    const uint32_t n_ff          = (uint32_t) gate->ne[1];
    const uint32_t n_expert      = (uint32_t) gate->ne[2];
    const uint32_t n_expert_used = (uint32_t) ids->ne[0];
    const uint32_t n_tokens      = (uint32_t) ids->ne[1];

    GGML_ASSERT(n_embd % DSV4_QK_K == 0);
    GGML_ASSERT((uint32_t) cur->ne[0] == n_embd);
    GGML_ASSERT((uint32_t) ggml_nelements(cur) == n_embd * n_tokens);
    GGML_ASSERT((uint32_t) up->ne[0] == n_embd && (uint32_t) up->ne[1] == n_ff && (uint32_t) up->ne[2] == n_expert);
    GGML_ASSERT((uint32_t) dst->ne[0] == n_ff && (uint32_t) dst->ne[1] == n_expert_used && (uint32_t) dst->ne[2] == n_tokens);

    float clamp;
    memcpy(&clamp, dst->op_params, sizeof(float));

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();

    const uint32_t xq_blocks     = n_embd / DSV4_QK_K;
    const uint32_t pair_count    = n_tokens * n_expert_used;
    const uint32_t tile_capacity = (pair_count + 7u) / 8u + n_expert;

    // pool workspace (not the graph compute buffer, so depth stays allocatable)
    ggml_cuda_pool_alloc<dsv4_block_q8_K> xq_alloc(pool, (size_t) n_tokens * xq_blocks);
    ggml_cuda_pool_alloc<uint32_t> counts_alloc      (pool, n_expert);
    ggml_cuda_pool_alloc<uint32_t> offsets_alloc     (pool, n_expert + 1);
    ggml_cuda_pool_alloc<uint32_t> cursors_alloc     (pool, n_expert);
    ggml_cuda_pool_alloc<uint32_t> sorted_alloc      (pool, pair_count);
    ggml_cuda_pool_alloc<uint32_t> tile_offsets_alloc(pool, n_expert + 1);
    ggml_cuda_pool_alloc<uint32_t> tile_total_alloc  (pool, 1);
    ggml_cuda_pool_alloc<uint32_t> tile_experts_alloc(pool, tile_capacity);
    ggml_cuda_pool_alloc<uint32_t> tile_starts_alloc (pool, tile_capacity);

    dsv4_block_q8_K * xq        = xq_alloc.get();
    uint32_t * counts           = counts_alloc.get();
    uint32_t * offsets          = offsets_alloc.get();
    uint32_t * cursors          = cursors_alloc.get();
    uint32_t * sorted_pairs     = sorted_alloc.get();
    uint32_t * tile_offsets     = tile_offsets_alloc.get();
    uint32_t * tile_total       = tile_total_alloc.get();
    uint32_t * tile_experts     = tile_experts_alloc.get();
    uint32_t * tile_starts      = tile_starts_alloc.get();

    const int32_t * selected = (const int32_t *) ids->data;

    // 1. quantize activations to q8_K
    dim3 xq_grid(xq_blocks, n_tokens, 1);
    dsv4_q8_K_quantize_kernel<<<xq_grid, 256, 0, stream>>>(xq, (const float *) cur->data, n_embd, n_tokens);

    // 2. sorted (token,slot) pairs by expert
    CUDA_CHECK(cudaMemsetAsync(counts, 0, n_expert * sizeof(uint32_t), stream));
    dsv4_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256, 0, stream>>>(counts, selected, pair_count);
    dsv4_prefix_sorted_pairs_kernel<<<1, 1, 0, stream>>>(offsets, cursors, counts, n_expert);
    dsv4_scatter_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256, 0, stream>>>(sorted_pairs, cursors, selected, pair_count);

    // 3. expert tiles (block_m = 8)
    dsv4_build_tile_offsets_kernel<<<1, 1, 0, stream>>>(tile_offsets, tile_total, counts, n_expert, 8u);
    dsv4_build_tiles_kernel<<<(n_expert + 255u) / 256u, 256, 0, stream>>>(tile_experts, tile_starts, tile_offsets, counts, n_expert, 8u);

    // 4. fused gate+up+activation -> mid
    const uint64_t gate_row_bytes    = gate->nb[1];
    const uint64_t gate_expert_bytes = gate->nb[2];
    dim3 gu_grid((n_ff + DSV4_MG_ROW_SPAN - 1u) / DSV4_MG_ROW_SPAN, tile_capacity, 1);
    dsv4_gate_up_mid_iq2_rowspan_kernel<DSV4_MG_ROW_SPAN><<<gu_grid, 256, 0, stream>>>(
        (float *) dst->data,
        (const char *) gate->data, (const char *) up->data,
        xq, sorted_pairs, offsets, counts, tile_total, tile_experts, tile_starts,
        gate_expert_bytes, gate_row_bytes, xq_blocks, n_ff, n_expert_used, clamp);
    CUDA_CHECK(cudaGetLastError());
}
