#!/usr/bin/env python3
"""
FP4 E2M1 Format Analysis & Conversion Table Generator

This script analyzes the NVIDIA FP4 E2M1 format and generates conversion tables
for MXFP4 to NVFP4 transformation in llama.cpp Blackwell implementation.

E2M1 Format:
- 1 sign bit
- 2 exponent bits (bias = 1)
- 1 mantissa bit

Total: 4 bits

Usage:
    python fp4_e2m1_analysis.py
"""

import struct
import math
import random

# ============================================================================
# E2M1 Format Decoding
# ============================================================================

def e2m1_to_float(bits: int) -> float:
    """
    Convert 4-bit E2M1 value to IEEE FP32.

    E2M1 layout: SEEM (Sign, Exponent×2, Mantissa)
    - S: bit 3 (sign)
    - E: bits 2-1 (exponent, bias=1)
    - M: bit 0 (mantissa)

    Args:
        bits: 4-bit integer (0-15)

    Returns:
        IEEE FP32 equivalent
    """
    assert 0 <= bits <= 15, f"Invalid E2M1 bits: {bits}"

    sign = (bits >> 3) & 0x1
    exp  = (bits >> 1) & 0x3
    mant = (bits >> 0) & 0x1

    # Special case: zero
    if exp == 0 and mant == 0:
        return 0.0 if sign == 0 else -0.0

    # Denormal numbers (exp = 0, mant != 0)
    if exp == 0:
        value = 0.5 * (1.0 + mant)  # No implicit leading 1
    else:
        # Normal numbers
        # Value = (-1)^S × 2^(E-bias) × (1 + M/2^1)
        exponent = exp - 1  # bias = 1
        mantissa = 1.0 + mant * 0.5  # implicit 1 + explicit bits
        value = mantissa * (2.0 ** exponent)

    return -value if sign == 1 else value


def float_to_e2m1_nearest(value: float) -> int:
    """
    Quantize IEEE FP32 to nearest E2M1 value (round-to-nearest-even).

    Args:
        value: Input float

    Returns:
        4-bit E2M1 representation (0-15)
    """
    if value == 0.0:
        return 0  # +0

    # Handle negative
    sign_bit = 1 if value < 0 else 0
    abs_value = abs(value)

    # Find nearest representable value
    best_bits = 0
    best_error = float('inf')

    for bits in range(16):
        if (bits >> 3) != sign_bit:
            continue  # Skip wrong sign

        candidate = e2m1_to_float(bits)
        error = abs(abs_value - abs(candidate))

        if error < best_error:
            best_error = error
            best_bits = bits

    return best_bits


def float_to_e2m1_stochastic(value: float) -> int:
    """
    Stochastic rounding: randomly choose between floor and ceil based on distance.

    Args:
        value: Input float

    Returns:
        4-bit E2M1 representation
    """
    if value == 0.0:
        return 0

    sign_bit = 1 if value < 0 else 0
    abs_value = abs(value)

    # Find floor and ceil
    floor_bits, ceil_bits = None, None
    floor_val, ceil_val = -float('inf'), float('inf')

    for bits in range(16):
        if (bits >> 3) != sign_bit:
            continue

        candidate = abs(e2m1_to_float(bits))

        if candidate <= abs_value and candidate > floor_val:
            floor_val = candidate
            floor_bits = bits

        if candidate >= abs_value and candidate < ceil_val:
            ceil_val = candidate
            ceil_bits = bits

    if floor_bits is None:
        return ceil_bits
    if ceil_bits is None:
        return floor_bits

    # Probability proportional to distance
    if ceil_val == floor_val:
        return floor_bits

    p_ceil = (abs_value - floor_val) / (ceil_val - floor_val)

    return ceil_bits if random.random() < p_ceil else floor_bits


# ============================================================================
# E2M1 Value Table Generation
# ============================================================================

def generate_e2m1_table():
    """Generate complete E2M1 representable values table."""
    print("=" * 70)
    print("E2M1 Format: Complete Representable Values")
    print("=" * 70)
    print(f"{'Bits':>6} | {'Binary':>6} | {'Sign':>4} | {'Exp':>3} | {'Mant':>4} | {'Value':>10} | {'Hex':>10}")
    print("-" * 70)

    for bits in range(16):
        sign = (bits >> 3) & 0x1
        exp  = (bits >> 1) & 0x3
        mant = (bits >> 0) & 0x1
        value = e2m1_to_float(bits)

        binary = f"{bits:04b}"
        sign_str = "-" if sign else "+"
        value_str = f"{value:+.6f}"
        hex_str = f"0x{bits:X}"

        print(f"{bits:6d} | {binary:>6} | {sign_str:>4} | {exp:3d} | {mant:4d} | {value_str:>10} | {hex_str:>10}")

    print("=" * 70)


def analyze_e2m1_properties():
    """Analyze mathematical properties of E2M1 format."""
    print("\n" + "=" * 70)
    print("E2M1 Format: Mathematical Properties")
    print("=" * 70)

    values = [e2m1_to_float(i) for i in range(16)]
    pos_values = [v for v in values if v > 0]

    print(f"Total representable values: {len(set(values))}")
    print(f"Positive values: {len([v for v in values if v > 0])}")
    print(f"Negative values: {len([v for v in values if v < 0])}")
    print(f"Zero values: {len([v for v in values if v == 0])}")
    print(f"\nDynamic range: [{min(values):.4f}, {max(values):.4f}]")
    print(f"Smallest positive: {min(pos_values):.4f}")
    print(f"Largest magnitude: {max(abs(v) for v in values):.4f}")

    # Quantization analysis
    print("\nQuantization bins (positive side):")
    pos_sorted = sorted(set(v for v in values if v >= 0))
    for i in range(len(pos_sorted) - 1):
        gap = pos_sorted[i+1] - pos_sorted[i]
        mid = (pos_sorted[i] + pos_sorted[i+1]) / 2
        rel_error = gap / mid if mid != 0 else 0
        print(f"  [{pos_sorted[i]:6.4f}, {pos_sorted[i+1]:6.4f}]: gap={gap:.4f}, rel_err={rel_error*100:5.1f}%")


# ============================================================================
# MXFP4 to NVFP4 Conversion
# ============================================================================

def generate_mxfp4_to_nvfp4_lut():
    """
    Generate lookup table for MXFP4 signed integers to E2M1 bit patterns.

    MXFP4 uses signed int8 lookup table:
        kvalues_mxfp4 = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]

    Returns:
        List of 16 values: E2M1 bit patterns for each MXFP4 nibble
    """
    kvalues_mxfp4 = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]

    print("\n" + "=" * 70)
    print("MXFP4 to NVFP4 (E2M1) Lookup Table")
    print("=" * 70)
    print(f"{'Nibble':>6} | {'MXFP4 Int':>10} | {'E2M1 Bits':>10} | {'E2M1 Float':>12} | {'Error':>10}")
    print("-" * 70)

    lut = []
    for nibble, signed_int in enumerate(kvalues_mxfp4):
        # Convert to E2M1
        e2m1_bits = float_to_e2m1_nearest(float(signed_int))
        e2m1_value = e2m1_to_float(e2m1_bits)
        error = abs(signed_int - e2m1_value)

        lut.append(e2m1_bits)

        print(f"{nibble:6d} | {signed_int:10d} | 0x{e2m1_bits:X} ({e2m1_bits:04b}) | {e2m1_value:12.4f} | {error:10.4f}")

    print("=" * 70)

    return lut


def generate_cuda_lut_code(lut):
    """Generate CUDA code for the lookup table."""
    print("\n" + "=" * 70)
    print("CUDA Code Generation")
    print("=" * 70)

    print("\n// Lookup table: MXFP4 nibble → E2M1 bit pattern")
    print("static __constant__ uint8_t mxfp4_to_e2m1_lut[16] = {")
    for i in range(0, 16, 8):
        values = [f"0x{v:X}" for v in lut[i:i+8]]
        comment = f"  // Nibbles {i}-{i+7}"
        print(f"    {', '.join(values)},{comment}")
    print("};")

    print("\n// Conversion function: MXFP4 block → NVFP4 packed registers")
    print("""
__device__ __forceinline__ void convert_mxfp4_to_nvfp4_block(
    const block_mxfp4* src,
    uint32_t* dst_packed,  // 2× uint32 for 16× FP4 values
    uint8_t* scale_out     // E8M0 scale (or convert to E4M3)
) {
    // Extract scale
    *scale_out = src->e;  // Keep as E8M0 for now

    // Convert 16 nibbles to E2M1 bit patterns
    dst_packed[0] = 0;
    dst_packed[1] = 0;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        // Extract nibble
        uint8_t byte = src->qs[i / 2];
        uint8_t nibble = (i % 2 == 0) ? (byte & 0xF) : (byte >> 4);

        // Lookup E2M1 bits
        uint8_t e2m1 = mxfp4_to_e2m1_lut[nibble];

        // Pack into register (8 FP4 per uint32)
        int reg_idx = i / 8;
        int shift = (i % 8) * 4;
        dst_packed[reg_idx] |= (uint32_t)e2m1 << shift;
    }
}
""")


# ============================================================================
# Quantization Error Analysis
# ============================================================================

def analyze_quantization_error():
    """Analyze quantization error for various input distributions."""
    print("\n" + "=" * 70)
    print("Quantization Error Analysis")
    print("=" * 70)

    random.seed(42)

    # Test different distributions
    def uniform_samples(a, b, n):
        return [random.uniform(a, b) for _ in range(n)]

    def normal_samples(mu, sigma, n):
        # Box-Muller transform
        samples = []
        for _ in range(n):
            u1, u2 = random.random(), random.random()
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            samples.append(mu + sigma * z)
        return samples

    test_cases = [
        ("Uniform [-6, 6]", uniform_samples(-6, 6, 1000)),
        ("Normal(0, 2)", normal_samples(0, 2, 1000)),
    ]

    for name, samples in test_cases:
        # Clip to representable range
        samples = [max(-6, min(6, x)) for x in samples]

        # Quantize
        quantized = [e2m1_to_float(float_to_e2m1_nearest(x)) for x in samples]

        # Compute errors
        abs_errors = [abs(s - q) for s, q in zip(samples, quantized)]
        rel_errors = [abs_e / (abs(s) + 1e-8) for abs_e, s in zip(abs_errors, samples)]

        mean_abs = sum(abs_errors) / len(abs_errors)
        max_abs = max(abs_errors)
        mean_rel = sum(rel_errors) / len(rel_errors)
        max_rel = max(rel_errors)
        rmse = math.sqrt(sum(e**2 for e in abs_errors) / len(abs_errors))

        print(f"\n{name}:")
        print(f"  Mean absolute error: {mean_abs:.4f}")
        print(f"  Max absolute error:  {max_abs:.4f}")
        print(f"  Mean relative error: {mean_rel*100:.2f}%")
        print(f"  Max relative error:  {max_rel*100:.2f}%")
        print(f"  RMSE:                {rmse:.4f}")


# ============================================================================
# E8M0 vs E4M3 Scaling Comparison
# ============================================================================

def compare_scale_formats():
    """Compare E8M0 (MXFP4) vs E4M3 (NVFP4) scaling accuracy."""
    print("\n" + "=" * 70)
    print("E8M0 vs E4M3 Scale Factor Comparison")
    print("=" * 70)

    # Simulate some realistic scale factors
    test_scales = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]

    print(f"{'Original':>12} | {'E8M0':>12} | {'E4M3*':>12} | {'E8M0 Error':>12} | {'E4M3 Error':>12}")
    print("-" * 70)

    for scale in test_scales:
        # E8M0: Quantize to power of 2
        e8m0_quantized = 2 ** round(math.log2(max(scale, 2**-127)))

        # E4M3: Approximate (would need actual FP8 implementation)
        # For now, use higher precision as proxy
        e4m3_quantized = round(scale * 8) / 8.0  # Rough approximation

        e8m0_error = abs(scale - e8m0_quantized) / scale * 100
        e4m3_error = abs(scale - e4m3_quantized) / scale * 100

        print(f"{scale:12.4f} | {e8m0_quantized:12.4f} | {e4m3_quantized:12.4f} | {e8m0_error:11.2f}% | {e4m3_error:11.2f}%")

    print("\n*Note: E4M3 values are approximate (actual FP8 implementation needed)")


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print(" FP4 E2M1 Format Analysis for Blackwell Tensor Cores")
    print(" NVIDIA GB10 - llama.cpp Implementation Helper")
    print("=" * 70)

    # 1. Generate E2M1 value table
    generate_e2m1_table()

    # 2. Analyze properties
    analyze_e2m1_properties()

    # 3. MXFP4 → NVFP4 conversion
    lut = generate_mxfp4_to_nvfp4_lut()

    # 4. Generate CUDA code
    generate_cuda_lut_code(lut)

    # 5. Quantization error analysis
    analyze_quantization_error()

    # 6. Scale format comparison
    compare_scale_formats()

    print("\n" + "=" * 70)
    print(" Analysis Complete!")
    print(" See generated CUDA code above for implementation.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
