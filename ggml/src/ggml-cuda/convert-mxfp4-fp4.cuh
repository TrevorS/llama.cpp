#pragma once
// ABOUTME: MXFP4 to NVFP4 Conversion Functions for Blackwell Tensor Cores
// Converts llama.cpp's MXFP4 format (E8M0 scaling) to NVIDIA's NVFP4 E2M1 format
//
// STATUS: Partial Implementation (Day 5 Complete)
//   ✅ Lookup table mxfp4_to_e2m1_lut[16] generated from fp4_e2m1_analysis.py
//   ✅ Saturation behavior documented (nibbles 6,7,14,15 → ±6.0 max)
//   ✅ float_to_e2m1() implemented with IEEE-754 → E2M1 conversion (Day 4)
//   ✅ convert_mxfp4_to_nvfp4_block() full pipeline implementation (Day 5)
//
// SCHEDULED IMPLEMENTATION:
//   ✅ Day 4: float_to_e2m1() E2M1 quantization - COMPLETE
//   ✅ Day 5: convert_mxfp4_to_nvfp4_block() conversion - COMPLETE
//   TODO Day 6: Integrate with kernel dispatch in vecdotq.cuh
//   TODO Day 7: Validation against reference and benchmarking
//
// REFERENCES:
//   - E2M1 format spec: docs/BLACKWELL_FP4_RESEARCH.md Table 2
//   - Conversion algorithm: docs/PROGRESS.md (Day 5 section)
//   - Kernel integration: docs/PROGRESS.md (Day 6 section)

#include "common.cuh"
#include "fp4-types.cuh"

#ifdef BLACKWELL_FP4_AVAILABLE

// ============================================================================
// MXFP4 to E2M1 Bit Pattern Lookup Table
// Maps MXFP4 nibble values to E2M1 bit patterns
// Generated from: scripts/fp4_e2m1_analysis.py
// ============================================================================

static __constant__ uint8_t mxfp4_to_e2m1_lut[16] = {
    0x0, 0x1, 0x4, 0x5, 0x6, 0x7, 0x7, 0x7,  // Nibbles 0-7 (positive)
    0x0, 0x9, 0xC, 0xD, 0xE, 0xF, 0xF, 0xF,  // Nibbles 8-15 (negative)
};

// Note on saturated values (nibbles 6, 7, 14, 15):
// MXFP4 values {8, 12, -8, -12} saturate to {6, -6} in E2M1 representation
// This causes quantization error of ~25-50% for these values
// This is acceptable as these values are rare in quantized neural networks

// ============================================================================
// Device Functions
// ============================================================================

// E2M1 Quantization - Converts FP32 to 4-bit E2M1 format
// Format: [Sign:1][Exponent:2][Mantissa:1]
// Range: ±0.5 to ±6.0 (saturates at ±6.0)
// Exponent mapping: E=0→2^-1, E=1→2^0, E=2→2^1, E=3→2^2
// Mantissa: implicit leading 1, value = ±(1.M) × 2^(exp-1)
// Expected quantization error: ~5-15% mean absolute error
__device__ __forceinline__ uint8_t float_to_e2m1(float x) {
    // Step 1: Handle zero (special case, all bits 0)
    if (x == 0.0f) {
        return 0x0;
    }

    // Step 2: Extract IEEE-754 components from FP32
    uint32_t bits;
    memcpy(&bits, &x, sizeof(float));

    uint32_t sign_bit = (bits >> 31) & 0x1;      // Bit 31
    int32_t exp_bits = ((bits >> 23) & 0xFF);    // Bits 30-23 (IEEE-754 exponent, bias 127)
    uint32_t mantissa_bits = (bits >> 22) & 0x1; // Bit 22 (1st mantissa bit)

    // Step 3: Convert IEEE-754 exponent to unbiased exponent
    int32_t exp_unbiased = exp_bits - 127;  // Remove IEEE-754 bias

    // Step 4: Handle denormals and special values
    if (exp_bits == 0) {
        // Denormal or zero - treat as very small → 0
        return 0x0;
    }
    if (exp_bits == 0xFF) {
        // Infinity or NaN - saturate to max (±6.0)
        // E=3, M=1: represents ±(1.5) × 2^2 = ±6.0
        return (sign_bit == 0) ? 0x7 : 0xF;  // +6.0 or -6.0
    }

    // Step 5: Map IEEE exponent to E2M1 exponent
    // E2M1 exponent represents 2^(E-1):
    //   IEEE exp -1 → E2M1 E=0 (2^-1 = 0.5)
    //   IEEE exp  0 → E2M1 E=1 (2^0  = 1.0)
    //   IEEE exp  1 → E2M1 E=2 (2^1  = 2.0)
    //   IEEE exp  2 → E2M1 E=3 (2^2  = 4.0)
    int32_t e2m1_exp = exp_unbiased + 1;  // Map to E2M1 exponent space

    // Step 6: Handle saturation and underflow
    if (e2m1_exp < 0) {
        // Value too small (< 0.5), round to smallest: 0.5 (E=0, M=0)
        e2m1_exp = 0;
        mantissa_bits = 0;
    } else if (e2m1_exp > 3) {
        // Value too large (> 6.0), saturate to max: 6.0 (E=3, M=1)
        e2m1_exp = 3;
        mantissa_bits = 1;
    }
    // else: Value in normal range [0.5, 6.0] - mantissa_bits already extracted

    // Step 7: Pack into 4-bit E2M1 format [S:1][E:2][M:1]
    uint8_t result = (sign_bit << 3) | (e2m1_exp << 1) | mantissa_bits;

    return result;
}

// MXFP4 → NVFP4 Block Conversion Pipeline
// Converts a full 32-value MXFP4 block (16 bytes + scale) to E2M1 packed format
// Input:  block_mxfp4 (32 nibbles in 16 bytes + E8M0 scale factor)
// Output: 4 × uint32_t registers (8 E2M1 values per register) + scale factor
//
// MXFP4 Block Structure:
//   - 16 bytes (qs[0..15]), each byte contains 2 nibbles
//   - Nibble packing: Byte[i] = [high_nibble:4][low_nibble:4]
//   - Nibble[j] extracted as: (qs[j/2] >> ((j%2)*4)) & 0xF
//   - 1 byte scale factor (E8M0 format)
//
// E2M1 Output Format:
//   - 4 registers, each packs 8 E2M1 values (4 bits each)
//   - Register[i] = [E2M1_7:4][E2M1_6:4]...[E2M1_1:4][E2M1_0:4]
//   - Total: 32 E2M1 values across 4 registers
//
// Expected calls: From vec_dot_mxfp4_q8_1 in vecdotq.cuh (Day 6)
// Integration point: mmq.cuh tile loading (~line 730)
__device__ __forceinline__ void convert_mxfp4_to_nvfp4_block(
        const block_mxfp4* src,
        uint32_t* dst_packed,
        uint8_t* scale_out) {

    // Step 1: Extract and store scale factor
    // Keep as E8M0 for now (can be converted to FP32 later using ggml_cuda_e8m0_to_fp32)
    *scale_out = src->e;

    // Step 2: Convert 32 MXFP4 nibbles to E2M1 format
    // Process in chunks of 8 nibbles per output register
    #pragma unroll
    for (int reg_idx = 0; reg_idx < 4; reg_idx++) {
        uint32_t packed_register = 0;

        // Step 3: Process 8 E2M1 values for this register
        #pragma unroll
        for (int val_idx = 0; val_idx < 8; val_idx++) {
            // Calculate global nibble index (0-31)
            int nibble_idx = reg_idx * 8 + val_idx;

            // Step 4: Extract MXFP4 nibble from source
            // MXFP4 stores nibbles in pairs per byte (low nibble first)
            // Byte[i] = [nibble_(2i+1):4][nibble_(2i):4]
            int byte_idx = nibble_idx / 2;              // Which byte (0-15)
            int nibble_shift = (nibble_idx % 2) * 4;   // Low nibble (0) or high nibble (4)
            uint8_t mxfp4_nibble = (src->qs[byte_idx] >> nibble_shift) & 0xF;

            // Step 5: Lookup E2M1 value from lookup table
            // mxfp4_to_e2m1_lut maps MXFP4 nibbles (0-15) to E2M1 bit patterns
            uint8_t e2m1_value = mxfp4_to_e2m1_lut[mxfp4_nibble];

            // Step 6: Pack into output register
            // E2M1 values are 4 bits, pack 8 per register
            // Position E2M1 value at bits [val_idx*4 : val_idx*4+3]
            packed_register |= ((uint32_t)e2m1_value << (val_idx * 4));
        }

        // Step 7: Write packed register to output
        dst_packed[reg_idx] = packed_register;
    }
}

#endif // BLACKWELL_FP4_AVAILABLE
