#pragma once
// ABOUTME: FP4 Type Definitions for Blackwell Tensor Cores
// Defines tile types for FP4 E2M1 (1 sign + 2 exponent + 1 mantissa)
// packed format and type traits for tensor core operations

#include "common.cuh"

#ifdef BLACKWELL_FP4_AVAILABLE

namespace ggml_cuda_mma {

// Forward declaration
template <int I, int J, typename T> struct tile;

// FP4 E2M1 packed format: 8 FP4 values per 32-bit register
// Each value is 4 bits: [S:1][E:2][M:1]
typedef uint32_t fp4_e2m1_packed;

// FP4 tile for 16x4 (A matrix in m16n8k32 MMA)
template <>
struct tile<16, 4, fp4_e2m1_packed> {
    static constexpr int I = 16;
    static constexpr int J = 4;
    static constexpr int ne = 2;  // 16 values in 2 registers (8 per register)

    uint32_t x[ne] = {0};  // 2 registers × 32 bits = 16 FP4 values

    static constexpr __device__ bool supported() {
        return BLACKWELL_FP4_AVAILABLE;
    }
};

// FP4 tile for 16x8 (A matrix in m16n8k32 MMA - full tile)
template <>
struct tile<16, 8, fp4_e2m1_packed> {
    static constexpr int I = 16;
    static constexpr int J = 8;
    static constexpr int ne = 4;  // 32 values in 4 registers (8 per register)

    uint32_t x[ne] = {0};  // 4 registers × 32 bits = 32 FP4 values

    static constexpr __device__ bool supported() {
        return BLACKWELL_FP4_AVAILABLE;
    }
};

// FP4 tile for 8x4 (B matrix in m16n8k32 MMA)
template <>
struct tile<8, 4, fp4_e2m1_packed> {
    static constexpr int I = 8;
    static constexpr int J = 4;
    static constexpr int ne = 2;  // 16 values in 2 registers (8 per register)

    uint32_t x[ne] = {0};  // 2 registers × 32 bits = 16 FP4 values

    static constexpr __device__ bool supported() {
        return BLACKWELL_FP4_AVAILABLE;
    }
};

// FP4 tile for 8x8 (B matrix in m16n8k32 MMA - full tile)
template <>
struct tile<8, 8, fp4_e2m1_packed> {
    static constexpr int I = 8;
    static constexpr int J = 8;
    static constexpr int ne = 4;  // 32 values in 4 registers (8 per register)

    uint32_t x[ne] = {0};  // 4 registers × 32 bits = 32 FP4 values

    static constexpr __device__ bool supported() {
        return BLACKWELL_FP4_AVAILABLE;
    }
};

} // namespace ggml_cuda_mma

#endif // BLACKWELL_FP4_AVAILABLE
