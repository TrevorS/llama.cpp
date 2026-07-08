// ds4_tile_kernels.cuh
// -----------------------------------------------------------------------------
// MoE prefill tile pipeline lifted from the standalone DeepSeek-V4-Flash CUDA
// engine at /home/trevor/Projects/ds4/ds4_cuda.cu (626 KB). The extracted code
// below is VERBATIM from that file unless noted; source line ranges (as of the
// Jun 16 2026 revision, 14100 lines) are cited per block. The pipeline is the
// Q4_K x q8_K expert-tiled MoE FFN that is ~2x faster than llama.cpp's on the
// DGX Spark (GB10, sm_121a).
//
// Extraction map (ds4_cuda.cu -> here):
//   struct cuda_block_q2_K            : lines 48-53   (needed only for parity/asserts)
//   struct cuda_block_q4_K            : lines 55-60
//   struct cuda_block_q8_K            : lines 62-66
//   struct cuda_block_iq2_xxs         : lines 68-71
//   #include ds4_iq2_tables_cuda.inc  : line 73  (IQ2 grid + ksigns tables)
//   dev_f16_to_f32                    : lines 10354-10356
//   dev_unpack_iq2_signs             : lines 10358-10362
//   dev_iq2_dp4a_8                    : lines 10364-10373
//   dev_dot_q2_16                     : lines 10375-10383  (used by dev_dot_q2_K_q8_K_block)
//   dev_dot_iq2_pair_16              : lines 10385-10390
//   dev_iq2_i8x8_lut                 : lines 10392-10405
//   dev_dot_iq2_xxs_q8_K_block_lut   : lines 10407-10438
//   dev_q4_K_get_scale_min           : lines 10589-10601
//   dev_dot_q4_32                     : lines 10603-10611
//   dev_dot_q4_K_q8_K_block          : lines 10613-10628
//   dev_dot_q4_32_q8_K_block8        : lines 10630-10645
//   dev_dot_q4_32_q8_K_block8_full   : lines 10647-10672
//   dev_dot_q4_K_q8_K_block8         : lines 10674-10715
//   dev_dot_q4_K_q8_K_block8_full    : lines 10717-10779
//   quarter_warp_sum_f32             : lines 10899-10906
//   q8_K_quantize_kernel             : lines 10908-10955
//   moe_count_sorted_pairs_kernel    : lines 11228-11237
//   moe_prefix_sorted_pairs_kernel   : lines 11239-11253
//   moe_scatter_sorted_pairs_kernel  : lines 11255-11266
//   moe_build_expert_tile_offsets_kernel : lines 11268-11283
//   moe_build_expert_tiles_kernel    : lines 11285-11300
//   moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel : lines 11986-12083
//   moe_down_q4K_expert_tile8_rowspan_kernel        : lines 12165-12237
//   moe_down_q4K_expert_tile16_rowspan_kernel       : lines 12239-12325
//
// Host launch geometry (grid/block/scratch sizing) is cited in harness.cu,
// mirroring the ds4 host driver at ds4_cuda.cu lines 13104-13574.
// -----------------------------------------------------------------------------
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// ---- ds4_cuda.cu:27 ----
#define CUDA_QK_K 256

// ---- ds4_cuda.cu:48-71 (block struct definitions, verbatim) ----
typedef struct {
    uint8_t scales[CUDA_QK_K / 16];
    uint8_t qs[CUDA_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} cuda_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[CUDA_QK_K / 2];
} cuda_block_q4_K;

typedef struct {
    float d;
    int8_t qs[CUDA_QK_K];
    int16_t bsums[CUDA_QK_K / 16];
} cuda_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[CUDA_QK_K / 8];
} cuda_block_iq2_xxs;

// ---- ds4_cuda.cu:73 (IQ2 constant tables) ----
#include "ds4_iq2_tables_cuda.inc"

// =============================================================================
// ggml layout parity checks
// Include ggml-common.h (CPP decl flavor) so we can static_assert the ds4 cuda
// block structs are bit-identical to ggml's. The relative path assumes this
// header is compiled from the llama.cpp tree; the Makefile adds -I for ggml/src.
// =============================================================================
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

static_assert(sizeof(cuda_block_q4_K) == sizeof(block_q4_K), "cuda_block_q4_K size != ggml block_q4_K");
static_assert(sizeof(cuda_block_q8_K) == sizeof(block_q8_K), "cuda_block_q8_K size != ggml block_q8_K");
static_assert(sizeof(cuda_block_iq2_xxs) == sizeof(block_iq2_xxs), "cuda_block_iq2_xxs size != ggml block_iq2_xxs");
static_assert(sizeof(cuda_block_q2_K) == sizeof(block_q2_K), "cuda_block_q2_K size != ggml block_q2_K");
// q4_K member offsets: ggml keeps d/dmin in an anonymous union at offset 0.
static_assert(offsetof(cuda_block_q4_K, scales) == offsetof(block_q4_K, scales), "q4_K scales offset mismatch");
static_assert(offsetof(cuda_block_q4_K, qs)     == offsetof(block_q4_K, qs),     "q4_K qs offset mismatch");
// q8_K member offsets.
static_assert(offsetof(cuda_block_q8_K, d)      == offsetof(block_q8_K, d),      "q8_K d offset mismatch");
static_assert(offsetof(cuda_block_q8_K, qs)     == offsetof(block_q8_K, qs),     "q8_K qs offset mismatch");
static_assert(offsetof(cuda_block_q8_K, bsums)  == offsetof(block_q8_K, bsums),  "q8_K bsums offset mismatch");
// iq2_xxs member offsets.
static_assert(offsetof(cuda_block_iq2_xxs, qs)  == offsetof(block_iq2_xxs, qs),  "iq2_xxs qs offset mismatch");

// =============================================================================
// Device helpers (verbatim from ds4_cuda.cu)
// =============================================================================

// ---- ds4_cuda.cu:10354-10356 ----
__device__ static float dev_f16_to_f32(uint16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}

// ---- ds4_cuda.cu:10358-10362 ----
__device__ __forceinline__ static uint32_t dev_unpack_iq2_signs(uint32_t v) {
    const uint32_t p = __popc(v) & 1u;
    const uint32_t s = v ^ (p << 7u);
    return s * 0x01010101u;
}

// ---- ds4_cuda.cu:10364-10373 ----
__device__ __forceinline__ static int32_t dev_iq2_dp4a_8(uint64_t grid, uint32_t sign, const int8_t *q8, int32_t acc) {
    const uint32_t signs = dev_unpack_iq2_signs(sign);
    const int32_t sm0 = __vcmpne4(signs & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(signs & 0x80402010u, 0);
    const int32_t g0 = __vsub4((int32_t)(uint32_t)grid ^ sm0, sm0);
    const int32_t g1 = __vsub4((int32_t)(uint32_t)(grid >> 32) ^ sm1, sm1);
    acc = __dp4a(g0, *(const int32_t *)(q8 + 0), acc);
    acc = __dp4a(g1, *(const int32_t *)(q8 + 4), acc);
    return acc;
}

// ---- ds4_cuda.cu:10375-10383 ----
__device__ static int32_t dev_dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 16; i += 4) {
        const int32_t v = (*(const int32_t *)(q2 + i) >> shift) & 0x03030303;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

// ---- ds4_cuda.cu:10385-10390 ----
__device__ static int32_t dev_dot_iq2_pair_16(uint8_t grid0, uint32_t sign0, uint8_t grid1, uint32_t sign1, const int8_t *q8) {
    int32_t sum = 0;
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid0], cuda_ksigns_iq2xs[sign0], q8, sum);
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid1], cuda_ksigns_iq2xs[sign1], q8 + 8, sum);
    return sum;
}

// ---- ds4_cuda.cu:10392-10405 ----
__device__ __forceinline__ static void dev_iq2_i8x8_lut(
        const uint64_t *grid,
        const uint8_t *signs,
        uint8_t grid_idx,
        uint32_t sign_idx,
        int32_t *w0,
        int32_t *w1) {
    const uint32_t s = dev_unpack_iq2_signs(signs[sign_idx]);
    const int32_t sm0 = __vcmpne4(s & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(s & 0x80402010u, 0);
    const uint64_t g = grid[grid_idx];
    *w0 = __vsub4((int32_t)(uint32_t)g ^ sm0, sm0);
    *w1 = __vsub4((int32_t)(uint32_t)(g >> 32) ^ sm1, sm1);
}

// ---- ds4_cuda.cu:10407-10438 ----
__device__ static float dev_dot_iq2_xxs_q8_K_block_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y,
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)(aux0 & 0xffu),           (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu),   (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu),   (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu),   (aux1 >> 21) & 127u, &w[6], &w[7]);
        int32_t sumi = 0;
        sumi = __dp4a(w[0], *(const int32_t *)(q8 + ib32 * 32u + 0),  sumi);
        sumi = __dp4a(w[1], *(const int32_t *)(q8 + ib32 * 32u + 4),  sumi);
        sumi = __dp4a(w[2], *(const int32_t *)(q8 + ib32 * 32u + 8),  sumi);
        sumi = __dp4a(w[3], *(const int32_t *)(q8 + ib32 * 32u + 12), sumi);
        sumi = __dp4a(w[4], *(const int32_t *)(q8 + ib32 * 32u + 16), sumi);
        sumi = __dp4a(w[5], *(const int32_t *)(q8 + ib32 * 32u + 20), sumi);
        sumi = __dp4a(w[6], *(const int32_t *)(q8 + ib32 * 32u + 24), sumi);
        sumi = __dp4a(w[7], *(const int32_t *)(q8 + ib32 * 32u + 28), sumi);
        bsum += sumi * ls;
    }
    return 0.125f * xd * y->d * (float)bsum;
}

// ---- ds4_cuda.cu:10589-10601 ----
__device__ static void dev_q4_K_get_scale_min(
        uint32_t j,
        const uint8_t *scales,
        uint8_t *d_out,
        uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

// ---- ds4_cuda.cu:10603-10611 ----
__device__ __forceinline__ static int32_t dev_dot_q4_32(const uint8_t *qs, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

// ---- ds4_cuda.cu:10613-10628 ----
__device__ static float dev_dot_q4_K_q8_K_block(const cuda_block_q4_K *x, const cuda_block_q8_K *y) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum = 0;
    int summs = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        summs += (int)m * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        isum += (int)sc * dev_dot_q4_32(x->qs + byte_off, y->qs + j * 32u, shift);
    }
    return y->d * xd * (float)isum - y->d * xmin * (float)summs;
}

// ---- ds4_cuda.cu:10630-10645 ----
__device__ __forceinline__ static void dev_dot_q4_32_q8_K_block8(
        const uint8_t *qs,
        const cuda_block_q8_K *const ys[8],
        uint32_t n,
        uint32_t y_off,
        int shift,
        int32_t sums[8]) {
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        #pragma unroll
        for (uint32_t p = 0; p < 8u; p++) {
            if (p < n) sums[p] = __dp4a(v, *(const int32_t *)(ys[p]->qs + y_off + i), sums[p]);
        }
    }
}

// ---- ds4_cuda.cu:10647-10672 ----
__device__ __forceinline__ static void dev_dot_q4_32_q8_K_block8_full(
        const uint8_t *qs,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t y_off,
        int shift,
        int32_t sums[8]) {
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        sums[0] = __dp4a(v, *(const int32_t *)(y0->qs + y_off + i), sums[0]);
        sums[1] = __dp4a(v, *(const int32_t *)(y1->qs + y_off + i), sums[1]);
        sums[2] = __dp4a(v, *(const int32_t *)(y2->qs + y_off + i), sums[2]);
        sums[3] = __dp4a(v, *(const int32_t *)(y3->qs + y_off + i), sums[3]);
        sums[4] = __dp4a(v, *(const int32_t *)(y4->qs + y_off + i), sums[4]);
        sums[5] = __dp4a(v, *(const int32_t *)(y5->qs + y_off + i), sums[5]);
        sums[6] = __dp4a(v, *(const int32_t *)(y6->qs + y_off + i), sums[6]);
        sums[7] = __dp4a(v, *(const int32_t *)(y7->qs + y_off + i), sums[7]);
    }
}

// ---- ds4_cuda.cu:10674-10715 ----
__device__ static void dev_dot_q4_K_q8_K_block8(
        const cuda_block_q4_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        const uint32_t y_off = j * 32u;
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        int32_t dots[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        dev_dot_q4_32_q8_K_block8(x->qs + byte_off, ys, n, y_off, shift, dots);
        #pragma unroll
        for (uint32_t p = 0; p < 8u; p++) {
            if (p < n) {
                summs[p] += (int)m * (int)(ys[p]->bsums[2u * j] + ys[p]->bsums[2u * j + 1u]);
                isum[p] += (int)sc * dots[p];
            }
        }
    }
    #pragma unroll
    for (uint32_t p = 0; p < 8u; p++) {
        if (p < n) {
            const float yd = ys[p]->d;
            acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
        }
    }
}

// ---- ds4_cuda.cu:10717-10779 ----
__device__ static void dev_dot_q4_K_q8_K_block8_full(
        const cuda_block_q4_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        float acc[8]) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        const uint32_t y_off = j * 32u;
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        int32_t dots[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        dev_dot_q4_32_q8_K_block8_full(
            x->qs + byte_off,
            y0, y1, y2, y3, y4, y5, y6, y7,
            y_off,
            shift,
            dots);
        const int ms0 = (int)m * (int)(y0->bsums[2u * j] + y0->bsums[2u * j + 1u]);
        const int ms1 = (int)m * (int)(y1->bsums[2u * j] + y1->bsums[2u * j + 1u]);
        const int ms2 = (int)m * (int)(y2->bsums[2u * j] + y2->bsums[2u * j + 1u]);
        const int ms3 = (int)m * (int)(y3->bsums[2u * j] + y3->bsums[2u * j + 1u]);
        const int ms4 = (int)m * (int)(y4->bsums[2u * j] + y4->bsums[2u * j + 1u]);
        const int ms5 = (int)m * (int)(y5->bsums[2u * j] + y5->bsums[2u * j + 1u]);
        const int ms6 = (int)m * (int)(y6->bsums[2u * j] + y6->bsums[2u * j + 1u]);
        const int ms7 = (int)m * (int)(y7->bsums[2u * j] + y7->bsums[2u * j + 1u]);
        summs[0] += ms0;
        summs[1] += ms1;
        summs[2] += ms2;
        summs[3] += ms3;
        summs[4] += ms4;
        summs[5] += ms5;
        summs[6] += ms6;
        summs[7] += ms7;
        isum[0] += (int)sc * dots[0];
        isum[1] += (int)sc * dots[1];
        isum[2] += (int)sc * dots[2];
        isum[3] += (int)sc * dots[3];
        isum[4] += (int)sc * dots[4];
        isum[5] += (int)sc * dots[5];
        isum[6] += (int)sc * dots[6];
        isum[7] += (int)sc * dots[7];
    }
    acc[0] += y0->d * xd * (float)isum[0] - y0->d * xmin * (float)summs[0];
    acc[1] += y1->d * xd * (float)isum[1] - y1->d * xmin * (float)summs[1];
    acc[2] += y2->d * xd * (float)isum[2] - y2->d * xmin * (float)summs[2];
    acc[3] += y3->d * xd * (float)isum[3] - y3->d * xmin * (float)summs[3];
    acc[4] += y4->d * xd * (float)isum[4] - y4->d * xmin * (float)summs[4];
    acc[5] += y5->d * xd * (float)isum[5] - y5->d * xmin * (float)summs[5];
    acc[6] += y6->d * xd * (float)isum[6] - y6->d * xmin * (float)summs[6];
    acc[7] += y7->d * xd * (float)isum[7] - y7->d * xmin * (float)summs[7];
}

// ---- ds4_cuda.cu:10899-10906 ----
__device__ static float quarter_warp_sum_f32(float v, uint32_t lane8) {
    uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (int offset = 4; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 8);
    }
    (void)lane8;
    return v;
}

// =============================================================================
// Kernels (verbatim from ds4_cuda.cu)
// =============================================================================

// ---- ds4_cuda.cu:10908-10955 ----
__global__ static void q8_K_quantize_kernel(cuda_block_q8_K *out, const float *x, uint32_t in_dim, uint32_t n_rows) {
    uint32_t b = blockIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / CUDA_QK_K) return;
    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * CUDA_QK_K;
    cuda_block_q8_K *yb = out + (uint64_t)row * (in_dim / CUDA_QK_K) + b;
    __shared__ float abs_part[256];
    __shared__ float val_part[256];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    uint32_t tid = threadIdx.x;
    float v = tid < CUDA_QK_K ? xr[tid] : 0.0f;
    abs_part[tid] = tid < CUDA_QK_K ? fabsf(v) : 0.0f;
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
        if (tid < CUDA_QK_K) yb->qs[tid] = 0;
        if (tid < CUDA_QK_K / 16) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0) {
        maxv_s = val_part[0];
        iscale_s = -127.0f / maxv_s;
    }
    __syncthreads();
    if (tid < CUDA_QK_K) {
        int qv = (int)lrintf(iscale_s * xr[tid]);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        yb->qs[tid] = (int8_t)qv;
    }
    __syncthreads();
    if (tid < CUDA_QK_K / 16) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0) yb->d = 1.0f / iscale_s;
}

// ---- ds4_cuda.cu:11228-11237 ----
__global__ static void moe_count_sorted_pairs_kernel(
        uint32_t *counts,
        const int32_t *selected,
        uint32_t pair_count) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    atomicAdd(counts + (uint32_t)expert_i, 1u);
}

// ---- ds4_cuda.cu:11239-11253 ----
__global__ static void moe_prefix_sorted_pairs_kernel(
        uint32_t *offsets,
        uint32_t *cursors,
        const uint32_t *counts,
        uint32_t expert_count) {
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

// ---- ds4_cuda.cu:11255-11266 ----
__global__ static void moe_scatter_sorted_pairs_kernel(
        uint32_t *sorted_pairs,
        uint32_t *cursors,
        const int32_t *selected,
        uint32_t pair_count) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    uint32_t pos = atomicAdd(cursors + (uint32_t)expert_i, 1u);
    sorted_pairs[pos] = pair;
}

// ---- ds4_cuda.cu:11268-11283 ----
__global__ static void moe_build_expert_tile_offsets_kernel(
        uint32_t *tile_offsets,
        uint32_t *tile_total,
        const uint32_t *counts,
        uint32_t expert_count,
        uint32_t block_m) {
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

// ---- ds4_cuda.cu:11285-11300 ----
__global__ static void moe_build_expert_tiles_kernel(
        uint32_t *tile_experts,
        uint32_t *tile_starts,
        const uint32_t *tile_offsets,
        const uint32_t *counts,
        uint32_t expert_count,
        uint32_t block_m) {
    uint32_t e = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= expert_count) return;
    uint32_t ntiles = (counts[e] + block_m - 1u) / block_m;
    uint32_t off = tile_offsets[e];
    for (uint32_t t = 0; t < ntiles; t++) {
        tile_experts[off + t] = e;
        tile_starts[off + t] = t * block_m;
    }
}

// ---- ds4_cuda.cu:11986-12083 ----
template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            if (np == 8u) {
                dev_dot_q4_K_q8_K_block8_full(gr + b, xqb[0] + b, xqb[1] + b,
                                              xqb[2] + b, xqb[3] + b,
                                              xqb[4] + b, xqb[5] + b,
                                              xqb[6] + b, xqb[7] + b, gate);
                dev_dot_q4_K_q8_K_block8_full(ur + b, xqb[0] + b, xqb[1] + b,
                                              xqb[2] + b, xqb[3] + b,
                                              xqb[4] + b, xqb[5] + b,
                                              xqb[6] + b, xqb[7] + b, up);
            } else {
                dev_dot_q4_K_q8_K_block8(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                         xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                         xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                         xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate);
                dev_dot_q4_K_q8_K_block8(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                         xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                         xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                         xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}

// ---- ds4_cuda.cu:12165-12237 ----
template <uint32_t ROW_SPAN>
__global__ static void moe_down_q4K_expert_tile8_rowspan_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][8];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            if (np == 8u) {
                dev_dot_q4_K_q8_K_block8_full(wr + b, xqb[0] + b, xqb[1] + b,
                                              xqb[2] + b, xqb[3] + b,
                                              xqb[4] + b, xqb[5] + b,
                                              xqb[6] + b, xqb[7] + b, acc);
            } else {
                dev_dot_q4_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                         xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                         xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                         xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

// ---- ds4_cuda.cu:12239-12325 ----
template <uint32_t ROW_SPAN>
__global__ static void moe_down_q4K_expert_tile16_rowspan_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            if (np >= 8u) {
                dev_dot_q4_K_q8_K_block8_full(wr + b, xqb[0] + b, xqb[1] + b,
                                              xqb[2] + b, xqb[3] + b,
                                              xqb[4] + b, xqb[5] + b,
                                              xqb[6] + b, xqb[7] + b, acc);
            } else {
                dev_dot_q4_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                         xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                         xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                         xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
            }
            if (np > 8u) {
                if (np == 16u) {
                    dev_dot_q4_K_q8_K_block8_full(wr + b, xqb[8] + b, xqb[9] + b,
                                                  xqb[10] + b, xqb[11] + b,
                                                  xqb[12] + b, xqb[13] + b,
                                                  xqb[14] + b, xqb[15] + b, acc + 8);
                } else {
                    dev_dot_q4_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                             xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                             xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                             xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
                }
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}
