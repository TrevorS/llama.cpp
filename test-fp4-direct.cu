// ABOUTME: Direct Kernel Test for FP4 MMA Operations
// Low-level PTX debugging with printf for register inspection
// Compile: nvcc -arch=sm_121 -I. -I./ggml/include -I./ggml/src/ggml-cuda test-fp4-direct.cu -o test-fp4-direct
// Run: ./test-fp4-direct

#include <cstdio>
#include <cstring>

// Include CUDA GGML headers
#include "ggml.h"
#include "ggml/src/ggml-cuda/common.cuh"
#include "ggml/src/ggml-cuda/fp4-types.cuh"
#include "ggml/src/ggml-cuda/mma-fp4.cuh"

using namespace ggml_cuda_mma;

// ============================================================================
// Test Pattern: Zero Test (Baseline)
// ============================================================================
__global__ void test_fp4_zero_kernel() {
    // Initialize tiles
    tile<16, 8, fp4_e2m1_packed> A;  // FP4 input matrix A
    tile<8, 8, fp4_e2m1_packed> B;   // FP4 input matrix B
    tile<16, 8, float> D;             // FP32 output accumulator

    // Set all inputs to zero
    for (int i = 0; i < 4; i++) {
        A.x[i] = 0x00000000;  // All FP4 values = 0.0
        B.x[i] = 0x00000000;  // All FP4 values = 0.0
        D.x[i] = 0.0f;        // Output accumulator = 0.0
    }

    // Print input state
    printf("[ZERO TEST] Input State:\n");
    printf("  A.x[0] = 0x%08x, A.x[1] = 0x%08x, A.x[2] = 0x%08x, A.x[3] = 0x%08x\n",
           A.x[0], A.x[1], A.x[2], A.x[3]);
    printf("  B.x[0] = 0x%08x, B.x[1] = 0x%08x, B.x[2] = 0x%08x, B.x[3] = 0x%08x\n",
           B.x[0], B.x[1], B.x[2], B.x[3]);
    printf("  D.x[0] = %.6f, D.x[1] = %.6f, D.x[2] = %.6f, D.x[3] = %.6f\n",
           D.x[0], D.x[1], D.x[2], D.x[3]);

    // Call MMA operation
    mma(D, A, B, 0, 0);

    // Print output state
    printf("[ZERO TEST] Output State:\n");
    printf("  D.x[0] = %.6f, D.x[1] = %.6f, D.x[2] = %.6f, D.x[3] = %.6f\n",
           D.x[0], D.x[1], D.x[2], D.x[3]);
    printf("  Expected: All zeros (0 * anything = 0)\n");

    // Check correctness
    bool passed = true;
    for (int i = 0; i < 4; i++) {
        if (D.x[i] != 0.0f) {
            passed = false;
            printf("  ✗ ERROR: D.x[%d] = %.6f (expected 0.0)\n", i, D.x[i]);
        }
    }
    if (passed) {
        printf("  ✓ PASS: Zero test passed\n");
    }
    printf("\n");
}

// ============================================================================
// Test Pattern: All Ones Test (Simple Pattern)
// ============================================================================
__global__ void test_fp4_ones_kernel() {
    tile<16, 8, fp4_e2m1_packed> A;
    tile<8, 8, fp4_e2m1_packed> B;
    tile<16, 8, float> D;

    // Set A and B to pattern representing 1.0 in FP4 E2M1
    // E2M1: 1 sign + 2 exponent + 1 mantissa
    // +1.0 is typically: sign=0, exp=01, mantissa=0 = 0010 = 0x2
    // So 8 values of 0x2 in a uint32: 0x22222222

    for (int i = 0; i < 4; i++) {
        A.x[i] = 0x22222222;  // All FP4 values ≈ 1.0
        B.x[i] = 0x22222222;  // All FP4 values ≈ 1.0
        D.x[i] = 0.0f;        // Output accumulator = 0.0
    }

    printf("[ONES TEST] Input State:\n");
    printf("  A.x[0..3] = 0x22222222 (all ≈ 1.0 in E2M1)\n");
    printf("  B.x[0..3] = 0x22222222 (all ≈ 1.0 in E2M1)\n");
    printf("  D initial: [%.6f, %.6f, %.6f, %.6f]\n",
           D.x[0], D.x[1], D.x[2], D.x[3]);

    // Call MMA operation
    mma(D, A, B, 0, 0);

    printf("[ONES TEST] Output State:\n");
    printf("  D.x[0] = %.6f, D.x[1] = %.6f, D.x[2] = %.6f, D.x[3] = %.6f\n",
           D.x[0], D.x[1], D.x[2], D.x[3]);
    printf("  Expected: Sum of products of 1.0 across 32 k-dimension\n");
    printf("  Rough estimate: ~32.0 per element (1.0 * 32 FP4 inputs)\n");

    bool has_nonzero = false;
    for (int i = 0; i < 4; i++) {
        if (D.x[i] != 0.0f) {
            has_nonzero = true;
        }
    }
    if (has_nonzero) {
        printf("  ✓ PASS: Non-zero output detected (instruction executing)\n");
    } else {
        printf("  ✗ FAIL: All zeros (instruction may not be executing)\n");
    }
    printf("\n");
}

// ============================================================================
// Test Pattern: Identity Pattern
// ============================================================================
__global__ void test_fp4_identity_kernel() {
    tile<16, 8, fp4_e2m1_packed> A;
    tile<8, 8, fp4_e2m1_packed> B;
    tile<16, 8, float> D;

    // Create identity-like pattern
    // A = column vector of 1.0s repeated
    // B = row vector of 1.0s repeated
    for (int i = 0; i < 4; i++) {
        A.x[i] = 0x22222222;  // All FP4 values ≈ 1.0
        B.x[i] = 0x22222222;  // All FP4 values ≈ 1.0
        D.x[i] = 0.0f;
    }

    printf("[IDENTITY TEST] Input State:\n");
    printf("  A = column vector of 1.0s (16 rows × 32 K-dim)\n");
    printf("  B = row vector of 1.0s (8 rows × 32 K-dim)\n");
    printf("  Expected output: roughly 32.0 for each element\n");

    mma(D, A, B, 0, 0);

    printf("[IDENTITY TEST] Output State:\n");
    printf("  D = [%.6f, %.6f, %.6f, %.6f]\n", D.x[0], D.x[1], D.x[2], D.x[3]);

    bool reasonable = true;
    for (int i = 0; i < 4; i++) {
        // Reasonable output should be in range ~[20, 40] for 1.0 * 32
        if (D.x[i] < 10.0f || D.x[i] > 100.0f) {
            reasonable = false;
            printf("  ⚠ WARNING: D.x[%d] = %.6f (outside reasonable range)\n", i, D.x[i]);
        }
    }
    if (reasonable && D.x[0] != 0.0f) {
        printf("  ✓ PASS: Identity test produces reasonable output\n");
    }
    printf("\n");
}

// ============================================================================
// Main Test Launcher
// ============================================================================
int main() {
    printf("================================================================================\n");
    printf("FP4 Tensor Core Direct Kernel Test - PTX Debugging\n");
    printf("Testing NVIDIA Blackwell FP4 E2M1 MMA Operations\n");
    printf("================================================================================\n\n");

    // Enable printf in CUDA kernels
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024);

    printf("Running Test Suite...\n\n");

    // Test 1: Zero Test
    printf("--- TEST 1: ZERO TEST (Baseline) ---\n");
    test_fp4_zero_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();

    // Test 2: Ones Test
    printf("--- TEST 2: ONES TEST (All 1.0 Pattern) ---\n");
    test_fp4_ones_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();

    // Test 3: Identity Test
    printf("--- TEST 3: IDENTITY TEST (Row/Column Vectors) ---\n");
    test_fp4_identity_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();

    printf("================================================================================\n");
    printf("Direct Kernel Test Complete\n");
    printf("================================================================================\n");

    return 0;
}
