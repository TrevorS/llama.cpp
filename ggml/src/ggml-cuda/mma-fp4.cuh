#pragma once
// ABOUTME: FP4 Tensor Core MMA Operations for Blackwell
// Implements m16n8k32 matrix multiply-accumulate with FP4 E2M1 inputs
//
// PTX Instruction:
// mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32

#include "common.cuh"
#include "fp4-types.cuh"

#ifdef BLACKWELL_FP4_AVAILABLE

namespace ggml_cuda_mma {

// Forward declaration from mma.cuh
template <int I, int J, typename T> struct tile;

// ============================================================================
// FP4 E2M1 MMA: m16n8k32 (32 FP4 values in K dimension)
//
// Tile configuration:
//   A: 16×32 FP4 values (m16n8k32 - K dimension is 32)
//   B: 8×32 FP4 values (transposed, column-major)
//   D: 16×8 output accumulator in FP32
// ============================================================================

static __device__ __forceinline__ void mma(
        tile<16, 8, float> & D,
        const tile<16, 8, fp4_e2m1_packed> & A,
        const tile<8, 8, fp4_e2m1_packed> & B,
        const uint8_t scale_a,
        const uint8_t scale_b) {

#ifdef BLACKWELL_FP4_AVAILABLE
    // Cast tiles to int* for register manipulation (matches mma.cuh pattern)
    const int * Axi = (const int *) A.x;  // 4 registers, 8 FP4 values per register
    const int * Bxi = (const int *) B.x;  // 4 registers (note: B is 8×8 tile = 32 values)
    int       * Dxi = (int       *) D.x;  // 4 registers for FP32 output

    // Stage 1: Minimal PTX Assembly (basic m16n8k32 instruction)
    // Using standard register constraints from mma.cuh pattern
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
        "{%0, %1, %2, %3}, "           // D output: 4 FP32 registers
        "{%4, %5, %6, %7}, "           // A input: 4 packed FP4 registers
        "{%8, %9, %10, %11}, "         // B input: 4 packed FP4 registers
        "{%12, %13, %14, %15};"        // D accumulator (same as output, for in-place update)
        : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]),
          "r"(Bxi[0]), "r"(Bxi[1]), "r"(Bxi[2]), "r"(Bxi[3]),
          "r"(Dxi[0]), "r"(Dxi[1]), "r"(Dxi[2]), "r"(Dxi[3])
    );
#else
    GGML_UNUSED_VARS(D, A, B, scale_a, scale_b);
    NO_DEVICE_CODE;
#endif // BLACKWELL_FP4_AVAILABLE
}

} // namespace ggml_cuda_mma

#endif // BLACKWELL_FP4_AVAILABLE
