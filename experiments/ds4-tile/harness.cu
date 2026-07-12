// harness.cu
// -----------------------------------------------------------------------------
// Standalone A/B correctness + timing harness for the ds4 MoE prefill tile
// pipeline extracted in ds4_tile_kernels.cuh.
//
// Pipeline under test (mirrors ds4_cuda.cu host driver, lines 13104-13574):
//   1. q8_K-quantize activations x[n_tokens x n_embd]           -> xq
//   2. count / prefix / scatter routed (token,slot) pairs by expert
//   3. build expert tiles (block_m = 8)
//   4. gate_up_mid q4K tile8 rowspan:
//        mid[pair][r] = silu(gate)*up * routing_weight   (weight baked in here)
//   5. q8_K-quantize mid[pair x n_ff_exp]                       -> midq
//   6. down q4K tile rowspan with atomic_out=1:
//        out[token] += down(midq[pair])   (accumulates experts per token)
//
// CPU reference recomputes the same math independently in fp32/scalar, using
// the SAME GPU-produced q8_K activations (dequantized) and the SAME random
// Q4_K weight blocks (dequantized). See notes in run_reference().
//
// Build:  make          (see Makefile, -arch=sm_121a)
// Run:    ./ds4-tile-harness [n_embd n_ff_exp n_expert n_expert_used n_tokens seed]
//         ./ds4-tile-harness --perf
//         ./ds4-tile-harness --tiny
//         flags: --row-span N (32|512|1024|2048), --down-tile16, --perf, --no-check
// -----------------------------------------------------------------------------
#include "ds4_tile_kernels.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#define CUDA_CHECK(x) do { \
    cudaError_t err__ = (x); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(1); \
    } \
} while (0)

// -----------------------------------------------------------------------------
// half helpers (host side)
// -----------------------------------------------------------------------------
static inline uint16_t f32_to_h(float f) {
    __half h = __float2half(f);
    uint16_t u; memcpy(&u, &h, sizeof(u));
    return u;
}
static inline float h_to_f32(uint16_t u) {
    __half h; memcpy(&h, &u, sizeof(h));
    return __half2float(h);
}

// -----------------------------------------------------------------------------
// CPU dequant / quant replicas (bit-faithful to the device helpers)
// -----------------------------------------------------------------------------

// ds4 q4_K get_scale_min (mirror of dev_q4_K_get_scale_min).
static inline void cpu_q4_K_get_scale_min(uint32_t j, const uint8_t *scales, uint8_t *d_out, uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

// Dequantize one Q4_K block (256 values) exactly as the device dot interprets it:
//   x[j*32 + i] = d*sc_j*q4 - dmin*m_j
static void cpu_dequant_q4_K(const cuda_block_q4_K *b, float *out) {
    const float d = h_to_f32(b->d);
    const float dmin = h_to_f32(b->dmin);
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        cpu_q4_K_get_scale_min(j, b->scales, &sc, &m);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        for (uint32_t i = 0; i < 32u; i++) {
            const int q4 = (b->qs[byte_off + i] >> shift) & 0x0f;
            out[j * 32u + i] = d * (float)sc * (float)q4 - dmin * (float)m;
        }
    }
}

// Dequantize one q8_K block: y[i] = d * qs[i].
static void cpu_dequant_q8_K(const cuda_block_q8_K *b, float *out) {
    for (uint32_t i = 0; i < CUDA_QK_K; i++) out[i] = b->d * (float)b->qs[i];
}

// Quantize a row of `in_dim` floats into q8_K blocks, replicating
// q8_K_quantize_kernel exactly (per 256-element super-block).
static void cpu_quantize_q8_K(const float *x, uint32_t in_dim, cuda_block_q8_K *out) {
    const uint32_t nb = in_dim / CUDA_QK_K;
    for (uint32_t b = 0; b < nb; b++) {
        const float *xr = x + (uint64_t)b * CUDA_QK_K;
        cuda_block_q8_K *yb = out + b;
        float amax = 0.0f, maxv = 0.0f;
        for (uint32_t i = 0; i < CUDA_QK_K; i++) {
            float a = fabsf(xr[i]);
            if (a > amax) { amax = a; maxv = xr[i]; }
        }
        if (amax == 0.0f) {
            yb->d = 0.0f;
            for (uint32_t i = 0; i < CUDA_QK_K; i++) yb->qs[i] = 0;
            for (uint32_t i = 0; i < CUDA_QK_K / 16; i++) yb->bsums[i] = 0;
            continue;
        }
        const float iscale = -127.0f / maxv;
        for (uint32_t i = 0; i < CUDA_QK_K; i++) {
            int qv = (int)lrintf(iscale * xr[i]);
            if (qv > 127) qv = 127;
            if (qv < -128) qv = -128;
            yb->qs[i] = (int8_t)qv;
        }
        for (uint32_t j = 0; j < CUDA_QK_K / 16; j++) {
            int sum = 0;
            for (int i = 0; i < 16; i++) sum += yb->qs[j * 16 + i];
            yb->bsums[j] = (int16_t)sum;
        }
        yb->d = 1.0f / iscale;
    }
}

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------
struct Config {
    uint32_t n_embd = 2048;      // = expert_in_dim and out_dim (down output)
    uint32_t n_ff_exp = 1536;    // = expert_mid_dim (rounded to mult of 256)
    uint32_t n_expert = 64;      // total experts
    uint32_t n_expert_used = 8;  // routed slots per token (== kernel `n_expert`)
    uint32_t n_tokens = 512;
    uint32_t row_span = 1024;    // gate row span; down uses down_row_span
    uint32_t down_row_span = 2048;
    bool down_tile16 = false;
    bool perf = false;
    bool check = true;
    uint64_t seed = 1234;
    int perf_warmup = 20;
    int perf_iters = 100;
};

static uint32_t round_up_qk(uint32_t v) {
    if (v == 0) return CUDA_QK_K;
    return ((v + CUDA_QK_K - 1u) / CUDA_QK_K) * CUDA_QK_K;
}

// -----------------------------------------------------------------------------
// Template dispatch for the rowspan kernels
// -----------------------------------------------------------------------------
static void launch_gate_up(uint32_t row_span, dim3 grid, uint32_t mid_dim_grid,
        float *gate, float *up, float *mid, const char *gate_w, const char *up_w,
        const cuda_block_q8_K *xq, const uint32_t *sorted_pairs, const uint32_t *offsets,
        const uint32_t *counts, const uint32_t *tile_total, const uint32_t *tile_experts,
        const uint32_t *tile_starts, const float *weights, uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes, uint32_t xq_blocks, uint32_t expert_mid_dim,
        uint32_t n_expert_used, uint32_t write_aux, float clamp) {
    (void)mid_dim_grid;
    #define GU(RS) moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel<RS><<<grid, 256>>>( \
        gate, up, mid, gate_w, up_w, xq, sorted_pairs, offsets, counts, tile_total, \
        tile_experts, tile_starts, weights, gate_expert_bytes, gate_row_bytes, \
        xq_blocks, expert_mid_dim, n_expert_used, write_aux, clamp)
    switch (row_span) {
        case 32:   GU(32);   break;
        case 512:  GU(512);  break;
        case 1024: GU(1024); break;
        case 2048: GU(2048); break;
        default:   GU(1024); break;
    }
    #undef GU
}

static void launch_down8(uint32_t row_span, dim3 grid,
        float *out, const char *down_w, const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs, const uint32_t *offsets, const uint32_t *counts,
        const uint32_t *tile_total, const uint32_t *tile_experts, const uint32_t *tile_starts,
        uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t midq_blocks,
        uint32_t out_dim, uint32_t n_expert_used, uint32_t atomic_out) {
    #define DN(RS) moe_down_q4K_expert_tile8_rowspan_kernel<RS><<<grid, 256>>>( \
        out, down_w, midq, sorted_pairs, offsets, counts, tile_total, tile_experts, \
        tile_starts, down_expert_bytes, down_row_bytes, midq_blocks, out_dim, \
        n_expert_used, atomic_out)
    switch (row_span) {
        case 32:   DN(32);   break;
        case 512:  DN(512);  break;
        case 1024: DN(1024); break;
        case 2048: DN(2048); break;
        default:   DN(2048); break;
    }
    #undef DN
}

static void launch_down16(uint32_t row_span, dim3 grid,
        float *out, const char *down_w, const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs, const uint32_t *offsets, const uint32_t *counts,
        const uint32_t *tile_total, const uint32_t *tile_experts, const uint32_t *tile_starts,
        uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t midq_blocks,
        uint32_t out_dim, uint32_t n_expert_used, uint32_t atomic_out) {
    #define DN16(RS) moe_down_q4K_expert_tile16_rowspan_kernel<RS><<<grid, 256>>>( \
        out, down_w, midq, sorted_pairs, offsets, counts, tile_total, tile_experts, \
        tile_starts, down_expert_bytes, down_row_bytes, midq_blocks, out_dim, \
        n_expert_used, atomic_out)
    switch (row_span) {
        case 32:   DN16(32);   break;
        case 512:  DN16(512);  break;
        case 1024: DN16(1024); break;
        case 2048: DN16(2048); break;
        default:   DN16(2048); break;
    }
    #undef DN16
}

// zero kernel (mirror of ds4 zero_kernel)
__global__ static void zero_kernel(float *p, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.0f;
}

// -----------------------------------------------------------------------------
// Device resource bundle
// -----------------------------------------------------------------------------
struct Dev {
    // dims
    uint32_t n_embd, n_ff_exp, n_total, n_used, n_tokens;
    uint32_t xq_blocks, midq_blocks, out_dim, expert_mid_dim, expert_in_dim;
    uint32_t pair_count, tile_capacity, tile16_capacity;
    uint64_t gate_expert_bytes, gate_row_bytes, down_expert_bytes, down_row_bytes;

    // host copies of weights (for reference dequant)
    std::vector<cuda_block_q4_K> h_gate, h_up, h_down;
    std::vector<float> h_x;          // activations [n_tokens x n_embd]
    std::vector<int32_t> h_selected; // [pair_count]
    std::vector<float> h_weights;    // [pair_count]

    // device buffers
    float   *d_x = nullptr;
    char    *d_gate = nullptr, *d_up = nullptr, *d_down = nullptr;
    int32_t *d_selected = nullptr;
    float   *d_weights = nullptr;
    cuda_block_q8_K *d_xq = nullptr, *d_midq = nullptr;
    float   *d_mid = nullptr, *d_out = nullptr;

    // scratch
    uint32_t *d_counts=nullptr, *d_offsets=nullptr, *d_cursors=nullptr, *d_sorted=nullptr;
    uint32_t *d_tile_offsets=nullptr, *d_tile_total=nullptr, *d_tile_experts=nullptr, *d_tile_starts=nullptr;
    uint32_t *d_t16_offsets=nullptr, *d_t16_total=nullptr, *d_t16_experts=nullptr, *d_t16_starts=nullptr;
};

static void gen_weights_block(cuda_block_q4_K *b, std::mt19937_64 &rng,
                              float dlo, float dhi, float dminhi) {
    std::uniform_real_distribution<float> du(dlo, dhi);
    std::uniform_real_distribution<float> dmu(0.0f, dminhi);
    b->d = f32_to_h(du(rng));
    b->dmin = f32_to_h(dmu(rng));
    std::uniform_int_distribution<int> byte(0, 255);
    for (int i = 0; i < 12; i++) b->scales[i] = (uint8_t)byte(rng);
    for (int i = 0; i < 128; i++) b->qs[i] = (uint8_t)byte(rng);
}

static void build(Dev &D, const Config &cfg) {
    D.n_embd = cfg.n_embd; D.n_ff_exp = cfg.n_ff_exp;
    D.n_total = cfg.n_expert; D.n_used = cfg.n_expert_used; D.n_tokens = cfg.n_tokens;
    D.expert_in_dim = cfg.n_embd;
    D.expert_mid_dim = cfg.n_ff_exp;
    D.out_dim = cfg.n_embd;
    D.xq_blocks = D.expert_in_dim / CUDA_QK_K;
    D.midq_blocks = D.expert_mid_dim / CUDA_QK_K;
    D.pair_count = D.n_tokens * D.n_used;
    D.tile_capacity = (D.pair_count + 7u) / 8u + D.n_total;
    D.tile16_capacity = (D.pair_count + 15u) / 16u + D.n_total;
    D.gate_row_bytes = (uint64_t)D.xq_blocks * sizeof(cuda_block_q4_K);
    D.gate_expert_bytes = (uint64_t)D.expert_mid_dim * D.gate_row_bytes;
    D.down_row_bytes = (uint64_t)D.midq_blocks * sizeof(cuda_block_q4_K);
    D.down_expert_bytes = (uint64_t)D.out_dim * D.down_row_bytes;

    std::mt19937_64 rng(cfg.seed);

    // --- weights: random valid Q4_K blocks (their dequant IS the ground truth) ---
    // scale chosen so gate/up pre-activations land in O(1) with unit activations.
    const uint64_t gate_blocks = (uint64_t)D.n_total * D.expert_mid_dim * D.xq_blocks;
    const uint64_t down_blocks = (uint64_t)D.n_total * D.out_dim * D.midq_blocks;
    D.h_gate.resize(gate_blocks); D.h_up.resize(gate_blocks); D.h_down.resize(down_blocks);
    const float wdlo = 3.0e-4f, wdhi = 9.0e-4f, wdminhi = 3.0e-4f;
    for (uint64_t i = 0; i < gate_blocks; i++) gen_weights_block(&D.h_gate[i], rng, wdlo, wdhi, wdminhi);
    for (uint64_t i = 0; i < gate_blocks; i++) gen_weights_block(&D.h_up[i],   rng, wdlo, wdhi, wdminhi);
    for (uint64_t i = 0; i < down_blocks; i++) gen_weights_block(&D.h_down[i], rng, wdlo, wdhi, wdminhi);

    // --- activations ~ N(0,1) ---
    D.h_x.resize((uint64_t)D.n_tokens * D.n_embd);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto &v : D.h_x) v = nd(rng);

    // --- routing: distinct experts per token, normalized positive weights ---
    D.h_selected.resize(D.pair_count);
    D.h_weights.resize(D.pair_count);
    std::vector<uint32_t> ids(D.n_total);
    for (uint32_t t = 0; t < D.n_tokens; t++) {
        for (uint32_t e = 0; e < D.n_total; e++) ids[e] = e;
        std::shuffle(ids.begin(), ids.end(), rng);
        float wsum = 0.0f;
        std::vector<float> w(D.n_used);
        std::uniform_real_distribution<float> wu(0.1f, 1.0f);
        for (uint32_t s = 0; s < D.n_used; s++) { w[s] = wu(rng); wsum += w[s]; }
        for (uint32_t s = 0; s < D.n_used; s++) {
            D.h_selected[(uint64_t)t * D.n_used + s] = (int32_t)ids[s];
            D.h_weights[(uint64_t)t * D.n_used + s]  = w[s] / wsum;
        }
    }

    // --- device allocations ---
    CUDA_CHECK(cudaMalloc(&D.d_x, D.h_x.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D.d_gate, gate_blocks * sizeof(cuda_block_q4_K)));
    CUDA_CHECK(cudaMalloc(&D.d_up,   gate_blocks * sizeof(cuda_block_q4_K)));
    CUDA_CHECK(cudaMalloc(&D.d_down, down_blocks * sizeof(cuda_block_q4_K)));
    CUDA_CHECK(cudaMalloc(&D.d_selected, D.pair_count * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&D.d_weights,  D.pair_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D.d_xq,   (uint64_t)D.n_tokens * D.xq_blocks * sizeof(cuda_block_q8_K)));
    CUDA_CHECK(cudaMalloc(&D.d_midq, (uint64_t)D.pair_count * D.midq_blocks * sizeof(cuda_block_q8_K)));
    CUDA_CHECK(cudaMalloc(&D.d_mid,  (uint64_t)D.pair_count * D.expert_mid_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D.d_out,  (uint64_t)D.n_tokens * D.out_dim * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&D.d_counts,  D.n_total * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&D.d_offsets, (D.n_total + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&D.d_cursors, D.n_total * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&D.d_sorted,  D.pair_count * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&D.d_tile_offsets, (D.n_total + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&D.d_tile_total,   sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&D.d_tile_experts, D.tile_capacity * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&D.d_tile_starts,  D.tile_capacity * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&D.d_t16_offsets, (D.n_total + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&D.d_t16_total,   sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&D.d_t16_experts, D.tile16_capacity * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&D.d_t16_starts,  D.tile16_capacity * sizeof(uint32_t)));

    // --- H2D (weights excluded from timed region) ---
    CUDA_CHECK(cudaMemcpy(D.d_x, D.h_x.data(), D.h_x.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(D.d_gate, D.h_gate.data(), gate_blocks*sizeof(cuda_block_q4_K), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(D.d_up,   D.h_up.data(),   gate_blocks*sizeof(cuda_block_q4_K), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(D.d_down, D.h_down.data(), down_blocks*sizeof(cuda_block_q4_K), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(D.d_selected, D.h_selected.data(), D.pair_count*sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(D.d_weights,  D.h_weights.data(),  D.pair_count*sizeof(float), cudaMemcpyHostToDevice));
}

// Run the full device pipeline once. tile-building is included (it is part of
// the per-forward MoE cost in ds4).
static void run_pipeline(Dev &D, const Config &cfg) {
    // 1. quantize activations
    dim3 xq_grid(D.xq_blocks, D.n_tokens, 1);
    q8_K_quantize_kernel<<<xq_grid, 256>>>(D.d_xq, D.d_x, D.expert_in_dim, D.n_tokens);

    // 2. sorted pairs
    CUDA_CHECK(cudaMemsetAsync(D.d_counts, 0, D.n_total * sizeof(uint32_t), 0));
    moe_count_sorted_pairs_kernel<<<(D.pair_count + 255u)/256u, 256>>>(D.d_counts, D.d_selected, D.pair_count);
    moe_prefix_sorted_pairs_kernel<<<1,1>>>(D.d_offsets, D.d_cursors, D.d_counts, D.n_total);
    moe_scatter_sorted_pairs_kernel<<<(D.pair_count + 255u)/256u, 256>>>(D.d_sorted, D.d_cursors, D.d_selected, D.pair_count);

    // 3. tiles (block_m = 8)
    moe_build_expert_tile_offsets_kernel<<<1,1>>>(D.d_tile_offsets, D.d_tile_total, D.d_counts, D.n_total, 8u);
    moe_build_expert_tiles_kernel<<<(D.n_total + 255u)/256u, 256>>>(D.d_tile_experts, D.d_tile_starts, D.d_tile_offsets, D.d_counts, D.n_total, 8u);
    if (cfg.down_tile16) {
        moe_build_expert_tile_offsets_kernel<<<1,1>>>(D.d_t16_offsets, D.d_t16_total, D.d_counts, D.n_total, 16u);
        moe_build_expert_tiles_kernel<<<(D.n_total + 255u)/256u, 256>>>(D.d_t16_experts, D.d_t16_starts, D.d_t16_offsets, D.d_counts, D.n_total, 16u);
    }

    // 4. gate_up_mid (write_aux=0, clamp=0)  grid = (ceil(mid/row_span), tile_capacity)
    dim3 gu_grid((D.expert_mid_dim + cfg.row_span - 1u)/cfg.row_span, D.tile_capacity, 1);
    launch_gate_up(cfg.row_span, gu_grid, 0,
        nullptr, nullptr, D.d_mid, D.d_gate, D.d_up, D.d_xq, D.d_sorted, D.d_offsets, D.d_counts,
        D.d_tile_total, D.d_tile_experts, D.d_tile_starts, D.d_weights,
        D.gate_expert_bytes, D.gate_row_bytes, D.xq_blocks, D.expert_mid_dim, D.n_used, 0u, 0.0f);

    // 5. quantize mid (rows = pair_count)
    dim3 mq_grid(D.midq_blocks, D.pair_count, 1);
    q8_K_quantize_kernel<<<mq_grid, 256>>>(D.d_midq, D.d_mid, D.expert_mid_dim, D.pair_count);

    // 6. down with atomic accumulation into out[n_tokens x out_dim]
    zero_kernel<<<((uint64_t)D.n_tokens*D.out_dim + 255u)/256u, 256>>>(D.d_out, (uint64_t)D.n_tokens*D.out_dim);
    if (cfg.down_tile16) {
        dim3 dn_grid((D.out_dim + cfg.down_row_span - 1u)/cfg.down_row_span, D.tile16_capacity, 1);
        launch_down16(cfg.down_row_span, dn_grid, D.d_out, D.d_down, D.d_midq,
            D.d_sorted, D.d_offsets, D.d_counts, D.d_t16_total, D.d_t16_experts, D.d_t16_starts,
            D.down_expert_bytes, D.down_row_bytes, D.midq_blocks, D.out_dim, D.n_used, 1u);
    } else {
        dim3 dn_grid((D.out_dim + cfg.down_row_span - 1u)/cfg.down_row_span, D.tile_capacity, 1);
        launch_down8(cfg.down_row_span, dn_grid, D.d_out, D.d_down, D.d_midq,
            D.d_sorted, D.d_offsets, D.d_counts, D.d_tile_total, D.d_tile_experts, D.d_tile_starts,
            D.down_expert_bytes, D.down_row_bytes, D.midq_blocks, D.out_dim, D.n_used, 1u);
    }
}

// -----------------------------------------------------------------------------
// CPU reference
//
// Uses the SAME GPU-produced q8_K activations (copied back and dequantized) and
// the SAME random Q4_K weight blocks (dequantized). mid is quantized to q8_K on
// the CPU with the same algorithm as the kernel, mirroring the GPU's mid->midq
// step, so the only residual differences are fp accumulation order and the
// (expected) 8-bit mid quantization.
// -----------------------------------------------------------------------------
static void run_reference(Dev &D, std::vector<float> &out_ref) {
    const uint32_t E = D.expert_mid_dim, IN = D.expert_in_dim, OUT = D.out_dim;
    const uint32_t xqb = D.xq_blocks, mqb = D.midq_blocks;

    // pull GPU-quantized activations back and dequantize per token
    std::vector<cuda_block_q8_K> h_xq((uint64_t)D.n_tokens * xqb);
    CUDA_CHECK(cudaMemcpy(h_xq.data(), D.d_xq, h_xq.size()*sizeof(cuda_block_q8_K), cudaMemcpyDeviceToHost));
    std::vector<float> x_deq((uint64_t)D.n_tokens * IN);
    for (uint32_t t = 0; t < D.n_tokens; t++)
        for (uint32_t b = 0; b < xqb; b++)
            cpu_dequant_q8_K(&h_xq[(uint64_t)t*xqb + b], &x_deq[(uint64_t)t*IN + b*CUDA_QK_K]);

    out_ref.assign((uint64_t)D.n_tokens * OUT, 0.0f);

    #pragma omp parallel
    {
        std::vector<float> gate_w(IN), up_w(IN), wrow(E > OUT ? E : OUT), mid(E), mid_deq(E);
        std::vector<cuda_block_q8_K> midq(mqb);
        #pragma omp for schedule(dynamic, 8)
        for (long pair = 0; pair < (long)D.pair_count; pair++) {
            const uint32_t tok = (uint32_t)pair / D.n_used;
            int32_t exp = D.h_selected[pair];
            if (exp < 0) exp = 0;
            const float rw = D.h_weights[pair];
            const float *xrow = &x_deq[(uint64_t)tok * IN];

            // gate/up GEMV over E rows
            for (uint32_t r = 0; r < E; r++) {
                const uint64_t base = ((uint64_t)exp * E + r) * xqb;
                // dequant gate/up row (xqb blocks of 256)
                float g = 0.0f, u = 0.0f;
                float gblk[CUDA_QK_K], ublk[CUDA_QK_K];
                for (uint32_t b = 0; b < xqb; b++) {
                    cpu_dequant_q4_K(&D.h_gate[base + b], gblk);
                    cpu_dequant_q4_K(&D.h_up[base + b], ublk);
                    const float *xx = xrow + b*CUDA_QK_K;
                    for (uint32_t i = 0; i < CUDA_QK_K; i++) { g += gblk[i]*xx[i]; u += ublk[i]*xx[i]; }
                }
                const float silu = g / (1.0f + expf(-g));
                mid[r] = silu * u * rw;
            }

            // quantize mid to q8_K then dequant (mirror GPU mid->midq)
            cpu_quantize_q8_K(mid.data(), E, midq.data());
            for (uint32_t b = 0; b < mqb; b++)
                cpu_dequant_q8_K(&midq[b], &mid_deq[b*CUDA_QK_K]);

            // down GEMV over OUT rows, accumulate into token output
            float *orow = &out_ref[(uint64_t)tok * OUT];
            for (uint32_t r = 0; r < OUT; r++) {
                const uint64_t base = ((uint64_t)exp * OUT + r) * mqb;
                float acc = 0.0f;
                float wblk[CUDA_QK_K];
                for (uint32_t b = 0; b < mqb; b++) {
                    cpu_dequant_q4_K(&D.h_down[base + b], wblk);
                    const float *mm = &mid_deq[b*CUDA_QK_K];
                    for (uint32_t i = 0; i < CUDA_QK_K; i++) acc += wblk[i]*mm[i];
                }
                #pragma omp atomic
                orow[r] += acc;
            }
        }
    }
}

// -----------------------------------------------------------------------------
static void print_dev() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("Device: %s  (sm_%d%d, %d SMs, %.1f GB)\n", p.name, p.major, p.minor,
           p.multiProcessorCount, p.totalGlobalMem / 1e9);
}

int main(int argc, char **argv) {
    Config cfg;
    // positional: n_embd n_ff_exp n_expert n_expert_used n_tokens seed
    std::vector<char*> pos;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--perf")) { cfg.perf = true; }
        else if (!strcmp(argv[i], "--no-check")) { cfg.check = false; }
        else if (!strcmp(argv[i], "--down-tile16")) { cfg.down_tile16 = true; }
        else if (!strcmp(argv[i], "--tiny")) { cfg.n_embd=256; cfg.n_ff_exp=256; cfg.n_expert=8; cfg.n_expert_used=4; cfg.n_tokens=33; }
        else if (!strcmp(argv[i], "--row-span") && i+1 < argc) { cfg.row_span = (uint32_t)atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--down-row-span") && i+1 < argc) { cfg.down_row_span = (uint32_t)atoi(argv[++i]); }
        else pos.push_back(argv[i]);
    }
    if (pos.size() > 0) cfg.n_embd = (uint32_t)atoi(pos[0]);
    if (pos.size() > 1) cfg.n_ff_exp = (uint32_t)atoi(pos[1]);
    if (pos.size() > 2) cfg.n_expert = (uint32_t)atoi(pos[2]);
    if (pos.size() > 3) cfg.n_expert_used = (uint32_t)atoi(pos[3]);
    if (pos.size() > 4) cfg.n_tokens = (uint32_t)atoi(pos[4]);
    if (pos.size() > 5) cfg.seed = (uint64_t)strtoull(pos[5], nullptr, 10);
    if (cfg.perf) { cfg.check = false; }  // timing mode; pass n_tokens explicitly (e.g. 2048)

    // enforce QK_K multiples for the quantized dims
    uint32_t orig_embd = cfg.n_embd, orig_ff = cfg.n_ff_exp;
    cfg.n_embd = round_up_qk(cfg.n_embd);
    cfg.n_ff_exp = round_up_qk(cfg.n_ff_exp);
    if (cfg.n_expert_used > cfg.n_expert) cfg.n_expert_used = cfg.n_expert;

    print_dev();
    if (orig_embd != cfg.n_embd || orig_ff != cfg.n_ff_exp)
        printf("[note] dims rounded up to QK_K(256) multiples: n_embd %u->%u, n_ff_exp %u->%u\n",
               orig_embd, cfg.n_embd, orig_ff, cfg.n_ff_exp);
    printf("Config: n_embd=%u n_ff_exp=%u n_expert=%u n_expert_used=%u n_tokens=%u seed=%llu row_span=%u down_row_span=%u down_tile16=%d\n",
           cfg.n_embd, cfg.n_ff_exp, cfg.n_expert, cfg.n_expert_used, cfg.n_tokens,
           (unsigned long long)cfg.seed, cfg.row_span, cfg.down_row_span, (int)cfg.down_tile16);

    Dev D;
    build(D, cfg);
    printf("Derived: xq_blocks=%u midq_blocks=%u pair_count=%u tile_capacity=%u tile16_capacity=%u\n",
           D.xq_blocks, D.midq_blocks, D.pair_count, D.tile_capacity, D.tile16_capacity);

    // one pipeline run
    run_pipeline(D, cfg);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> h_out((uint64_t)D.n_tokens * D.out_dim);
    CUDA_CHECK(cudaMemcpy(h_out.data(), D.d_out, h_out.size()*sizeof(float), cudaMemcpyDeviceToHost));

    int rc = 0;
    if (cfg.check) {
        std::vector<float> ref;
        run_reference(D, ref);
        double max_rel = 0.0, sum_rel = 0.0; uint64_t n_bad = 0, n = h_out.size();
        double max_abs_gpu = 0.0;
        for (uint64_t i = 0; i < n; i++) max_abs_gpu = std::max(max_abs_gpu, (double)fabsf(ref[i]));
        const double denom_floor = 1e-3 * (max_abs_gpu > 0 ? max_abs_gpu : 1.0);
        for (uint64_t i = 0; i < n; i++) {
            double a = h_out[i], b = ref[i];
            double rel = fabs(a - b) / (fabs(b) + denom_floor);
            max_rel = std::max(max_rel, rel);
            sum_rel += rel;
            if (rel > 1e-2) n_bad++;
        }
        double mean_rel = sum_rel / (double)n;
        printf("Correctness: max_rel_err=%.4e mean_rel_err=%.4e frac(rel>1e-2)=%.4f (max|ref|=%.3e)\n",
               max_rel, mean_rel, (double)n_bad/(double)n, max_abs_gpu);
        bool pass = (max_rel < 5e-2) && (mean_rel < 1e-2);
        printf("Result: %s\n", pass ? "PASS" : "FAIL");
        if (!pass) rc = 2;
    }

    if (cfg.perf) {
        cudaEvent_t a, b;
        CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
        for (int i = 0; i < cfg.perf_warmup; i++) run_pipeline(D, cfg);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(a, 0));
        for (int i = 0; i < cfg.perf_iters; i++) run_pipeline(D, cfg);
        CUDA_CHECK(cudaEventRecord(b, 0));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        double ms_iter = ms / cfg.perf_iters;
        double tok_s = (double)cfg.n_tokens / (ms_iter / 1e3);
        printf("Perf: %.4f ms/iter  %.1f tokens/s  (FFN stage, %d iters)\n", ms_iter, tok_s, cfg.perf_iters);
    }

    return rc;
}
