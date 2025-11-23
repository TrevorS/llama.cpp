#pragma once
// ABOUTME: FP4 Tensor Core MMA Operations for Blackwell GB10
// Implements m16n8k32 matrix multiply-accumulate with FP4 E2M1 inputs
//
// STATUS: Stage 3 Complete (Days 1-3 ✅)
//   ✅ Basic PTX m16n8k32 instruction working
//   ✅ Register passing validated (4A, 4B, 4D inputs)
//   ✅ Hardware execution confirmed on GB10 Blackwell
//   ✅ Scale factors integrated via post-MMA multiplication
//
// NEXT: Days 4-7 Integration (Scheduled)
//   TODO Day 6: Connect to kernel dispatch system (vecdotq.cuh)
//   TODO Day 7: Performance validation and benchmarking
//
// DOCUMENTATION:
//   - Implementation details: docs/PROGRESS.md (Section "Day 2: Stage 1-3")
//   - Test results: docs/TEST_RESULTS.md (All 5 FP4 tests passing)
//   - Quick reference: docs/QUICKSTART-DAY2.md (Step-by-step reproduction)

#include "common.cuh"
#include "fp4-types.cuh"
#include "mma.cuh"

#ifdef BLACKWELL_FP4_AVAILABLE

namespace ggml_cuda_mma {

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

    // Stage 2: Execute basic PTX m16n8k32 instruction
    // Using row-major A, column-major B, FP32 accumulator D
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

    // Stage 3: Post-MMA Scale Factor Multiplication
    // Scale factors are in E8M0 format (8-bit unsigned exponent)
    // Convert to FP32 and apply combined scale: output *= (2^exp_a) * (2^exp_b)

    // Convert E8M0 scale factors to FP32
    const float scale_a_fp32 = ggml_cuda_e8m0_to_fp32(scale_a);
    const float scale_b_fp32 = ggml_cuda_e8m0_to_fp32(scale_b);
    const float combined_scale = scale_a_fp32 * scale_b_fp32;

    // Apply scale to all 4 FP32 output registers
    // Cast to float* for element-wise multiplication
    float* D_float = (float*) D.x;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        D_float[i] *= combined_scale;
    }
#else
    GGML_UNUSED_VARS(D, A, B, scale_a, scale_b);
    NO_DEVICE_CODE;
#endif // BLACKWELL_FP4_AVAILABLE
}

} // namespace ggml_cuda_mma

#endif // BLACKWELL_FP4_AVAILABLE
