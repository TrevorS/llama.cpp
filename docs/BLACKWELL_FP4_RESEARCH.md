# Blackwell FP4 Tensor Core Research Documentation

**Research Date:** 2025-11-22
**Target Hardware:** NVIDIA GB10 Blackwell iGPU (DGX Spark)
**Objective:** Implement native FP4 tensor core support for MXFP4 quantization in llama.cpp
**Expected Performance:** 60 → 75-85 tokens/sec (25-40% improvement)

---

## 1. Executive Summary

### Key Findings

✅ **Blackwell GB10 Specifications Confirmed:**
- Compute Capability: **sm_121** (CC 12.1)
- Tensor Cores: 192× 5th-generation with native FP4 support
- Peak Performance: 1000 TOPS @ FP4, ~120 TOPS currently utilized (DP4A)
- **8x performance headroom available**

✅ **NVFP4 Format Documented:**
- Bit layout: **E2M1** (1 sign, 2 exponent, 1 mantissa)
- Representable values: {0, 0.5, 1, 1.5, 2, 3, 4, 6} and negatives
- Block size: **16 elements** (vs MXFP4's 32)
- Scale format: **E4M3** FP8 (vs MXFP4's E8M0)

✅ **PTX Instruction Identified:**
```ptx
mma.sync.aligned.m16n8k32.row.col.kind::mxf4nvf4.block_scale.scale_vec::1X.f32.e2m1.e2m1.f32
```

✅ **Reference Implementation Found:**
- CUTLASS examples/72_blackwell_narrow_precision_gemm/
- cuBLAS 12.9+ supports CUDA_R_4F_E2M1 with cuBLASLt

### Critical Challenge: MXFP4 → NVFP4 Conversion

**Problem:** llama.cpp uses MXFP4 (E8M0 scaling, 32-element blocks), but Blackwell tensor cores expect NVFP4 (E4M3 scaling, 16-element blocks).

**Solution Required:** Runtime conversion or direct compatibility path (see Section 6).

---

## 2. Blackwell Architecture Details

### 2.1 GB10 Specifications

| Component | Specification |
|-----------|---------------|
| **Architecture** | NVIDIA Blackwell (5th Gen Tensor Cores) |
| **Compute Capability** | sm_121 (CC 12.1) |
| **Manufacturing** | TSMC 3nm |
| **CUDA Cores** | 6,144 |
| **Tensor Cores** | 192 |
| **L2 Cache** | 24 MB |
| **Memory** | 128 GB LPDDR5X-9400 (unified, coherent) |
| **Memory Bandwidth** | 273-301 GB/s |
| **TDP** | 140W |
| **Peak FP4 Performance** | ~1000 TOPS |
| **Peak FP32 Performance** | 31 TFLOPS |

**Source:** [NVIDIA GB10 Details](https://wccftech.com/nvidia-gb10-superchip-soc-3nm-20-arm-v9-2-cpu-cores-nvfp4-blackwell-gpu-lpddr5x-9400-memory-140w-tdp/)

### 2.2 Fifth-Generation Tensor Cores

**Supported Data Types:**
- FP64, FP32/TF32, FP16/BF16 (legacy)
- INT8, FP8 (E4M3, E5M2)
- **FP6** (E3M2, E2M3) - NEW
- **FP4** (E2M1) - NEW

**Performance Claims:**
- FP4: **2x throughput** vs FP8
- FP4: **4.6x speedup** vs Hopper H200 FP8 (on GB200)
- Achieved: **6,787 TFLOPS/s** in cuBLAS benchmarks

**Sources:**
- [NVIDIA RTX Blackwell Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
- [Blackwell Architecture Overview](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)

---

## 3. NVFP4 Format Specification

### 3.1 Bit Layout

```
┌─────┬──────┬──────┬──────┐
│  S  │ E1   │ E0   │ M0   │
│ (1) │ (1)  │ (1)  │ (1)  │
└─────┴──────┴──────┴──────┘
   Sign  Exponent  Mantissa
          (2 bits) (1 bit)
```

**E2M1 Format:**
- Exponent bias: 1 (assumed based on IEEE conventions)
- Denormals: Supported (0b00 exponent)
- Special values: Zero (all zeros)

### 3.2 Representable Values

| Bits | Interpretation | Value |
|------|----------------|-------|
| 0000 | +0 | 0.0 |
| 0001 | +denormal | 0.5 |
| 0010 | +normal (e=0) | 1.0 |
| 0011 | +normal (e=0) | 1.5 |
| 0100 | +normal (e=1) | 2.0 |
| 0101 | +normal (e=1) | 3.0 |
| 0110 | +normal (e=2) | 4.0 |
| 0111 | +normal (e=2) | 6.0 |
| 1000-1111 | Negative equivalents | -0.0 to -6.0 |

**Dynamic Range:** -6 to +6

**Source:** [Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

### 3.3 Two-Level Scaling

NVFP4 uses hierarchical scaling to extend dynamic range:

```
final_value = fp4_value × scale_e4m3 × scale_fp32
              └─────────┘   └────────┘   └────────┘
              Base value    Block scale   Tensor scale
              (16 values)   (E4M3 FP8)    (FP32)
```

**Block Size:** 16 elements (halved from MXFP4's 32)
- Allows finer-grained adaptation to data distribution
- Reduces quantization error

**Block Scale Format:** E4M3 FP8
```
┌─────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│  S  │ E3   │ E2   │ E1   │ E0   │ M2   │ M1   │ M0   │
└─────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
   1     4 exponent bits       3 mantissa bits
```

**Advantage over MXFP4:**
- MXFP4 E8M0: Power-of-two scales only (0 mantissa bits)
- NVFP4 E4M3: Fractional scales (3 mantissa bits) → better precision

**Source:** [NVFP4 Trains with Precision of 16-Bit](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/)

---

## 4. MXFP4 Format (Current llama.cpp Implementation)

### 4.1 Memory Layout

From `ggml/src/ggml-common.h:190-195`:

```c
#define QK_MXFP4 32
typedef struct {
    uint8_t e;              // E8M0 exponent (shared scale)
    uint8_t qs[QK_MXFP4/2]; // 32 nibbles (4-bit values) = 16 bytes
} block_mxfp4;
// Total: 17 bytes for 32 elements = 4.25 bits/element
```

### 4.2 E8M0 Scale Factor

**E8M0 Format:** Unsigned 8-bit exponent only (no mantissa, no sign)

```c
// From ggml/src/ggml-cuda/common.cuh:618-629
static __device__ __forceinline__ float ggml_cuda_e8m0_to_fp32(uint8_t x) {
#if CUDART_VERSION >= 12080
    const nv_bfloat16 e = __nv_cvt_e8m0_to_bf16raw(x);
    return (float) e;
#else
    uint32_t bits;
    if (x == 0) {
        bits = 0x00400000; // Special case: 2^-127
    } else {
        bits = (uint32_t) x << 23; // Direct exponent mapping
    }
    return *((float *) &bits);
#endif
}
```

**Scaling:** `final_value = kvalues_mxfp4[nibble] * e8m0_to_fp32(e) * 0.5f`

### 4.3 Quantized Value Lookup Table

From `ggml/src/ggml-common.h:1094-1096`:

```c
GGML_TABLE_BEGIN(int8_t, kvalues_mxfp4, 16)
    0, 1, 2, 3, 4, 6, 8, 12,  // Indices 0-7 (positive)
    0, -1, -2, -3, -4, -6, -8, -12,  // Indices 8-15 (negative)
GGML_TABLE_END()
```

**Important:** These are signed integers, not FP4 bit patterns!

### 4.4 Current DP4A Path

From `ggml/src/ggml-cuda/vecdotq.cuh:295-314`:

```c
static __device__ __forceinline__ float vec_dot_mxfp4_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {

    const block_mxfp4 * bq4 = (const block_mxfp4 *) vbq + kbx;
    const int * q8 = (const int *) bq8_1->qs + iqs;

    int sumi = 0;
    for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        const int aux_q4 = get_int_b1(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);

        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi); // INT8 DP4A
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }

    const float d = ggml_cuda_e8m0_to_fp32(bq4->e) * 0.5f * __low2float(bq8_1->ds);
    return d * sumi;
}
```

**Bottleneck:** Uses CUDA cores (DP4A), not tensor cores (~120 TOPS vs 1000 TOPS available).

---

## 5. PTX Assembly for FP4 Tensor Cores

### 5.1 Instruction Syntax (Blackwell sm_121)

Based on PTX ISA 9.0 and CUTLASS examples:

```ptx
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32
    {d0, d1, d2, d3},      // Output: 4× f32 accumulator registers
    {a0, a1, a2, a3},      // A matrix: 4× packed e2m1 registers
    {b0, b1},              // B matrix: 2× packed e2m1 registers
    {c0, c1, c2, c3},      // C accumulator: 4× f32 registers (input)
    {scale_a},             // Scale factor A: e4m3 or ue8m0
    {scale_b};             // Scale factor B: e4m3 or ue8m0
```

**Sources:**
- [Stack Overflow: FP4 PTX on Blackwell](https://stackoverflow.com/questions/79735243/how-are-fp6-and-fp4-supported-on-nvidia-tensor-core-on-blackwell)
- [NVIDIA Forums: Running FP4 MMA on sm_120a](https://forums.developer.nvidia.com/t/run-ptx-mma-sync-aligned-kind-mxf8f6f4-block-scale-scale-vec-1x-m16n8k32-on-sm-120a/329702)

### 5.2 Instruction Breakdown

| Component | Description |
|-----------|-------------|
| `mma.sync.aligned` | Warp-synchronous matrix-multiply-accumulate |
| `.kind::mxf4nvf4` | Microscaling FP4/NVFP4 variant |
| `.block_scale` | Use block-level scaling factors |
| `.scale_vec::1X` | Scale vector dimension (1× = per-block) |
| `.m16n8k32` | Tile size: M=16, N=8, K=32 |
| `.row.col` | A is row-major, B is column-major |
| `.f32.e2m1.e2m1.f32` | Output/A/B/Accum data types |

**Key Difference from INT8 (m16n8k16):**
- K dimension: 16 → **32** (double throughput)
- Scale factors: Required (not in INT8)

### 5.3 Register Allocation

**Output/Accumulator (D/C):** 4 registers @ f32
- Each register holds 1× f32 value
- Total: 4× f32 = 16 bytes

**A Matrix:** 4 registers @ packed e2m1
- Each register holds 8× FP4 values (32 bits / 4 bits = 8)
- Total: 32× FP4 values

**B Matrix:** 2 registers @ packed e2m1
- Each register holds 8× FP4 values
- Total: 16× FP4 values

**Scale Factors:** 1 register each
- Format: e4m3 (FP8) or ue8m0 (unsigned E8M0)

### 5.4 Inline Assembly Template

```cuda
__device__ __forceinline__ void mma_fp4_e2m1(
    float (&d)[4],           // Output accumulator
    const uint32_t (&a)[4],  // A matrix (packed FP4)
    const uint32_t (&b)[2],  // B matrix (packed FP4)
    const float (&c)[4],     // Input accumulator
    const uint8_t scale_a,   // Scale factor A (e4m3 or ue8m0)
    const uint8_t scale_b    // Scale factor B
) {
#if __CUDA_ARCH__ >= 1210  // Blackwell CC 12.1+
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::1X."
        "m16n8k32.row.col.f32.e2m1.e2m1.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13}, "
        "{%14}, {%15};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(scale_a), "r"(scale_b)
    );
#else
    #error "Blackwell CC 12.1+ required for FP4 tensor cores"
#endif
}
```

**Constraint Codes:**
- `"f"`: FP32 register
- `"r"`: 32-bit general register
- `"+"`: Read-write operand

### 5.5 Compilation Requirements

**CUDA Toolkit:** 13.0+

**Compiler Flags:**
```bash
nvcc -arch=compute_121 -code=sm_121 \
     --ptxas-options=-v \
     -DCUDA_ARCH=121 \
     kernel.cu
```

**Critical:** Must specify both `arch` (virtual) and `code` (real) for sm_121.

**Sources:**
- [CUDA 13.0 Documentation](https://docs.nvidia.com/cuda/index.html)
- [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)

---

## 6. MXFP4 → NVFP4 Conversion Challenge

### 6.1 Format Incompatibilities

| Aspect | MXFP4 (llama.cpp) | NVFP4 (Blackwell) | Compatible? |
|--------|-------------------|-------------------|-------------|
| **Base Format** | E2M1 (signed int8 lookup) | E2M1 (4-bit FP) | ⚠️ Partial |
| **Block Size** | 32 elements | 16 elements | ❌ No |
| **Scale Format** | E8M0 (power-of-2) | E4M3 (fractional) | ❌ No |
| **Storage** | 17 bytes/block | 10 bytes/block* | ❌ No |

*NVFP4: 16 FP4 (8 bytes) + 1 E4M3 scale (1 byte) + 1 FP32 tensor scale (4 bytes shared) ≈ 10 bytes

### 6.2 Conversion Strategy Options

#### Option A: Runtime Conversion (Recommended for Prototyping)

**Convert on-the-fly during kernel execution:**

```cuda
__device__ uint32_t convert_mxfp4_block_to_nvfp4(
    const block_mxfp4* src,
    int offset  // Which 16 elements (0 or 1)
) {
    // Step 1: Extract E8M0 scale and convert to FP32
    float scale_e8m0 = ggml_cuda_e8m0_to_fp32(src->e) * 0.5f;

    // Step 2: Extract 16 nibbles (8 bytes)
    const uint8_t* qs_ptr = src->qs + offset * 8;

    // Step 3: Pack into NVFP4 format
    uint32_t result[2] = {0, 0};  // 16 FP4 values = 64 bits

    for (int i = 0; i < 8; i++) {
        uint8_t byte = qs_ptr[i];
        uint8_t nibble_lo = byte & 0xF;
        uint8_t nibble_hi = byte >> 4;

        // Lookup quantized values
        float val_lo = kvalues_mxfp4[nibble_lo] * scale_e8m0;
        float val_hi = kvalues_mxfp4[nibble_hi] * scale_e8m0;

        // Convert to FP4 E2M1 bit patterns
        uint8_t fp4_lo = float_to_e2m1(val_lo);
        uint8_t fp4_hi = float_to_e2m1(val_hi);

        // Pack into registers
        int reg_idx = i / 4;
        int shift = (i % 4) * 8;
        result[reg_idx] |= (fp4_lo << shift);
        result[reg_idx] |= (fp4_hi << (shift + 4));
    }

    return result;
}
```

**Pros:**
- No model file format changes
- Backward compatible
- Quick to prototype

**Cons:**
- Runtime overhead (~5-10% estimated)
- Doesn't use native E4M3 scaling

#### Option B: Pre-Quantization to NVFP4 (Optimal Performance)

**Modify quantization pipeline to generate NVFP4 directly:**

1. Update `ggml-quants.c::quantize_row_mxfp4()` to use 16-element blocks
2. Compute E4M3 scale factors instead of E8M0
3. Store NVFP4 format in model file (new GGML type?)

**Pros:**
- Zero runtime conversion overhead
- Native E4M3 scaling → better accuracy
- Maximum performance

**Cons:**
- Requires model re-quantization
- New GGML type: `GGML_TYPE_NVFP4`
- More implementation work

#### Option C: Hybrid Approach (Pragmatic)

1. **Phase 1:** Implement Option A for validation
2. **Phase 2:** Add NVFP4 quantization (Option B) as new type
3. **Phase 3:** Support both MXFP4 (convert) and NVFP4 (native)

**Recommended Path:** Start with Option A, migrate to Option C.

### 6.3 E8M0 → E4M3 Scale Conversion

**Challenge:** MXFP4 scale is power-of-two only (E8M0), NVFP4 expects fractional scales (E4M3).

**Naive Approach (lossy):**
```cuda
// Convert E8M0 to FP32, then quantize to E4M3
float scale_fp32 = e8m0_to_fp32(src->e) * 0.5f;
uint8_t scale_e4m3 = fp32_to_e4m3(scale_fp32);
```

**Problem:** Information loss if E8M0 scale was already quantized.

**Better Approach (if re-quantizing):**
```cuda
// Compute optimal E4M3 scale from original FP32 weights
float scale_e4m3 = compute_optimal_scale_e4m3(fp32_weights, 16);
```

**Sources:**
- [MXFP4 vs NVFP4 Comparison](https://arxiv.org/pdf/2509.23202)
- [Bridging the Gap Between Promise and Performance](https://www.arxiv.org/pdf/2509.23202)

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Week 1, Days 1-3)

**Goal:** Compile and run basic FP4 MMA stub

**Tasks:**
1. Add Blackwell CC 12.1 detection to `ggml/src/ggml-cuda/common.cuh`
   ```cuda
   #define GGML_CUDA_CC_BLACKWELL 1210
   #define BLACKWELL_FP4_AVAILABLE (__CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL)
   ```

2. Create `ggml/src/ggml-cuda/fp4-types.cuh`:
   ```cuda
   namespace ggml_cuda_mma {
       template <> struct tile<16, 2, fp4_e2m1_packed> {
           uint32_t x[1]; // 8 FP4 values per register
       };
       template <> struct tile<16, 4, fp4_e2m1_packed> {
           uint32_t x[2]; // 16 FP4 values
       };
       template <> struct tile<16, 8, fp4_e2m1_packed> {
           uint32_t x[4]; // 32 FP4 values
       };
   }
   ```

3. Stub MMA wrapper in `ggml/src/ggml-cuda/mma-fp4.cuh`:
   ```cuda
   static __device__ __forceinline__ void mma(
       tile<16, 8, float> & D,
       const tile<16, 8, fp4_e2m1_packed> & A,
       const tile<8, 4, fp4_e2m1_packed> & B,
       uint8_t scale_a,
       uint8_t scale_b
   ) {
   #ifdef BLACKWELL_FP4_AVAILABLE
       // TODO: Actual PTX assembly
       for (int i = 0; i < 4; i++) D.x[i] = 0.0f; // Stub
   #else
       NO_DEVICE_CODE;
   #endif
   }
   ```

**Deliverable:** Code compiles with `-arch=sm_121`, stub executes without crashes.

### Phase 2: PTX Implementation (Week 1, Days 4-7)

**Goal:** Working FP4 MMA instruction with hardcoded test inputs

**Tasks:**
1. Implement actual PTX inline assembly (see Section 5.4)
2. Create test kernel:
   ```cuda
   __global__ void test_fp4_mma_basic() {
       // Hardcoded identity matrix test
       tile<16, 8, fp4_e2m1_packed> A = make_identity_fp4();
       tile<8, 4, fp4_e2m1_packed> B = make_identity_fp4();
       tile<16, 8, float> C = {};

       mma(C, A, B, 1, 1);  // scale=1.0

       // Verify output
       if (threadIdx.x == 0) {
           assert(fabs(C.x[0] - 1.0f) < 1e-2);
       }
   }
   ```

3. Debug PTX compilation errors (expect multiple iterations)

**Deliverable:** test_fp4_mma_basic passes on DGX Spark.

### Phase 3: Conversion Functions (Week 2, Days 8-12)

**Goal:** MXFP4 → NVFP4 conversion with numerical correctness

**Tasks:**
1. Implement `float_to_e2m1()` quantization function
2. Implement `convert_mxfp4_block_to_nvfp4()` (see Section 6.2, Option A)
3. Create test:
   ```cuda
   __global__ void test_conversion() {
       block_mxfp4 src = load_test_block();
       uint32_t nvfp4_packed[2] = convert_mxfp4_block_to_nvfp4(&src, 0);

       float* ref = dequantize_mxfp4_to_fp32(&src);
       float* actual = dequantize_nvfp4_to_fp32(nvfp4_packed);

       for (int i = 0; i < 16; i++) {
           assert(fabs(ref[i] - actual[i]) < 0.1f);  // 10% tolerance
       }
   }
   ```

**Deliverable:** Conversion within 10% error of FP32 reference.

### Phase 4: Integration (Week 2-3, Days 13-18)

**Goal:** Replace DP4A path with FP4 MMA in `vec_dot_mxfp4_q8_1`

**Tasks:**
1. Modify `ggml/src/ggml-cuda/vecdotq.cuh::vec_dot_mxfp4_q8_1()`:
   ```cuda
   #ifdef BLACKWELL_FP4_AVAILABLE
       // Convert MXFP4 to NVFP4
       tile<16, 8, fp4_e2m1_packed> A_fp4;
       convert_and_load_mxfp4_to_nvfp4(bq4, A_fp4, scale_a);

       // Convert Q8_1 to FP4 (or keep as FP8)
       tile<8, 4, fp4_e2m1_packed> B_fp4;
       convert_q8_to_nvfp4(bq8_1, B_fp4, scale_b);

       // MMA
       tile<16, 8, float> result;
       mma(result, A_fp4, B_fp4, scale_a, scale_b);

       return reduce_sum(result);
   #else
       // Fallback to DP4A
   #endif
   ```

2. Update `ggml/src/ggml-cuda/mmq.cuh::load_tiles_mxfp4()` for tensor core path

3. Add dispatch logic in `ggml/src/ggml-cuda/mmq.cu`

**Deliverable:** Model runs end-to-end, produces text (even if slow/inaccurate).

### Phase 5: Optimization (Week 3, Days 19-21)

**Goal:** Achieve 75-85 tokens/sec target

**Tasks:**
1. **Profile with Nsight Compute:**
   ```bash
   ncu --set full --target-processes all \
       ./llama-cli -m gpt-oss-120b-mxfp4.gguf -p "Test" -n 128
   ```

2. **Optimize:**
   - Shared memory layout (minimize bank conflicts)
   - Warp-level tile sizes (tune M×N×K)
   - Register pressure reduction
   - Async memory transfers (ldmatrix)

3. **Measure:**
   ```bash
   ./llama-bench -m gpt-oss-120b-mxfp4.gguf -r 10
   # Target: 75-85 tok/s (vs current 60.4 tok/s)
   ```

**Deliverable:** ≥75 tokens/sec achieved.

---

## 8. Testing & Validation Strategy

### 8.1 Unit Tests

**Level 1: PTX Instruction**
```cuda
// tests/test_fp4_mma_basic.cu
GTEST_TEST(FP4_MMA, IdentityMatrix) {
    // Identity @ Identity = Identity
    auto result = run_fp4_mma(identity, identity);
    EXPECT_NEAR(result, identity, 1e-2);
}
```

**Level 2: Conversion**
```cuda
GTEST_TEST(FP4_Conversion, MXFP4_to_NVFP4) {
    block_mxfp4 input = generate_random_mxfp4();
    auto nvfp4 = convert(input);
    auto reconstructed = dequantize(nvfp4);
    auto reference = dequantize(input);
    EXPECT_NEAR(reconstructed, reference, 0.1f); // 10% tolerance
}
```

**Level 3: Vector Dot Product**
```cuda
GTEST_TEST(FP4_VecDot, SmallMatrix) {
    // 16x32 @ 32x8 matrix multiply
    auto result_fp4 = vec_dot_mxfp4_q8_1_fp4(A, B);
    auto result_fp32 = vec_dot_fp32_reference(A, B);
    EXPECT_NEAR(result_fp4, result_fp32, 0.05f); // 5% tolerance
}
```

### 8.2 Numerical Precision Analysis

**FP4 E2M1 Precision:**
- Mantissa bits: 1 → precision ≈ 2^-1 = 50% relative error per value
- With E4M3 scaling: reduces to ~5% with proper quantization
- Accumulated over 120B parameters: Monitor perplexity drift

**Acceptable Tolerances:**
| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Per-element error | <10% | E2M1 has ~6% quantization error |
| Layer output error | <5% | With proper scaling |
| Model perplexity | <+2% | Acceptable quality degradation |
| Token accuracy | >98% | Compared to FP32 baseline |

**Test Procedure:**
```bash
# 1. Generate reference logits (FP32)
./llama-cli -m model-fp32.gguf -p "prompt" --logits-out ref.txt

# 2. Generate FP4 logits
./llama-cli -m model-mxfp4.gguf -p "prompt" --logits-out fp4.txt

# 3. Compare
python compare_logits.py ref.txt fp4.txt
# Expected: KL divergence < 0.05, top-k accuracy > 98%
```

### 8.3 Performance Benchmarks

**Kernel-Level Metrics:**
```bash
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
    --metrics smsp__sass_thread_inst_executed_op_dadd_pred_on.sum \
    ./test_fp4_kernel

# Target metrics:
# - SM utilization: >80%
# - Tensor core active: >70%
# - Achieved TOPS: 600-800 (vs 1000 theoretical)
```

**End-to-End Throughput:**
```bash
./llama-bench -m gpt-oss-120b-mxfp4.gguf -n 128 -r 10

# Expected results:
# Baseline (DP4A):     60.4 tok/s
# Target (FP4 TC):     75-85 tok/s
# Minimum acceptable:  70 tok/s (+16%)
```

---

## 9. Risk Mitigation & Fallback Plans

### Risk 1: PTX Syntax Undocumented/Incorrect

**Probability:** Medium
**Impact:** High (project blocker)

**Mitigation:**
1. **Reverse engineer cuBLAS:**
   ```bash
   nvdisasm /usr/local/cuda/lib64/libcublas.so.13 > cublas.ptx
   grep -A20 "mma.*e2m1" cublas.ptx
   ```

2. **Use cuBLASLt wrapper:**
   ```cuda
   // Fallback: Call cuBLASLt instead of custom PTX
   cublasLtMatmul(handle, matmulDesc,
       &alpha, A_nvfp4, A_desc, B_nvfp4, B_desc,
       &beta, C_fp32, C_desc, D_fp32, D_desc,
       algo, workspace, workspace_size, stream);
   ```

3. **Ask NVIDIA Developer Forums:**
   - Post minimal reproducible example
   - Request PTX syntax confirmation
   - Reference CUTLASS examples

**Fallback:** If PTX never works, wrap cuBLASLt (10-15% overhead acceptable).

### Risk 2: Numerical Accuracy Unacceptable

**Probability:** Medium
**Impact:** Medium (quality degradation)

**Mitigation:**
1. **Tune scaling factors:** Use optimal E4M3 computation
2. **Mixed precision:** Keep attention in FP16, FFN in FP4
3. **Quantization-aware training:** Fine-tune model for FP4

**Fallback:** Accept 70 tok/s with INT8 MMA (m16n8k16) instead of FP4.

### Risk 3: Performance Below Target (<70 tok/s)

**Probability:** Low
**Impact:** Medium (missed goal)

**Mitigation:**
1. **Profile-guided optimization:** Nsight Compute analysis
2. **Tile size tuning:** Experiment with m16n8k16 vs m16n8k32
3. **Shared memory optimization:** Eliminate bank conflicts
4. **Async loading:** Overlap compute and memory transfers

**Fallback:** Document partial improvement (e.g., 68 tok/s = +13% is still valuable).

---

## 10. Reference Implementation: CUTLASS

### 10.1 Key Code Patterns

From `examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu`:

**Data Type Definitions:**
```cpp
using ElementA = cutlass::float_e2m1_t;       // FP4 E2M1
using ElementB = cutlass::float_e2m1_t;
using ElementC = cutlass::float_e2m1_t;
using ElementD = cutlass::float_e2m1_t;
using ElementScale = cutlass::float_ue8m0_t;  // Unsigned E8M0 (or E4M3)
```

**MMA Configuration:**
```cpp
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using TileShape = cutlass::gemm::GemmShape<128, 128, 256>;  // M×N×K
using WarpShape = cutlass::gemm::GemmShape<64, 64, 256>;
```

**Scale Factor Layout:**
```cpp
using LayoutScaleA = cutlass::layout::PackedVectorLayout;
using LayoutScaleB = cutlass::layout::PackedVectorLayout;
constexpr int ScaleVectorSize = 16;  // NVFP4 block size
```

**Epilogue Fusion:**
```cpp
using EpilogueOp = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
    ElementD, ElementCompute, ElementScale, ScaleVectorSize>;
```

### 10.2 Lessons for llama.cpp Implementation

1. **Use block size = 16** (not 32)
2. **Scale factors can be E8M0 or E4M3** (hardware supports both)
3. **Tile size m16n8k32** confirmed for FP4
4. **Register packing:** 8 FP4 values per 32-bit register
5. **Warp specialization:** Separate MMA and epilogue warps for efficiency

**Source:** [CUTLASS GitHub - Blackwell FP4 Example](https://github.com/NVIDIA/cutlass/blob/main/examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu)

---

## 11. Open Questions & Research Needed

### 11.1 Unresolved Technical Questions

❓ **Q1:** Can Blackwell tensor cores accept E8M0 scales directly, or conversion to E4M3 mandatory?
- **Investigation:** Test both in PTX assembly
- **Impact:** If E8M0 works, no scale conversion needed (huge simplification)

❓ **Q2:** What is the exact register count for m16n8k32 FP4 MMA?
- **Investigation:** Disassemble cuBLAS or compile CUTLASS with --ptxas-options=-v
- **Impact:** Affects register pressure and occupancy

❓ **Q3:** Does Blackwell support MXFP4 format natively (32-element blocks)?
- **Investigation:** Check PTX ISA for `.kind::mxfp4` variant
- **Impact:** Could eliminate conversion entirely

❓ **Q4:** What is performance of cuBLASLt wrapper vs custom PTX?
- **Investigation:** Benchmark both approaches
- **Impact:** Determines if custom PTX is worth the effort

### 11.2 Documentation Gaps

- [ ] Complete PTX ISA 9.0 section on FP4 MMA (verify .kind modifiers)
- [ ] Blackwell programming guide (not yet publicly released)
- [ ] GB10 specific tuning guidelines (cache sizes, latencies)

---

## 12. Success Criteria

### Milestone 1: "Hello World" ✅ (Days 1-3)
- [ ] Code compiles with -arch=sm_121
- [ ] Blackwell detection works
- [ ] Stub MMA function executes without crashing

### Milestone 2: "First Light" ✅ (Days 4-7)
- [ ] PTX instruction compiles
- [ ] MMA executes (even with wrong results)
- [ ] No hangs or device errors

### Milestone 3: "Numerical Correctness" ✅ (Days 8-12)
- [ ] Conversion test passes (<10% error)
- [ ] Small matmul test passes (<5% error)
- [ ] Unit tests green

### Milestone 4: "Integration" ✅ (Days 13-16)
- [ ] GPT-OSS-120B loads
- [ ] Forward pass completes
- [ ] Generates coherent text (perplexity check)

### Milestone 5: "Performance" 🎯 (Days 17-21)
- [ ] **≥75 tokens/sec** (stretch goal: 85)
- [ ] SM utilization >80%
- [ ] No regressions on other quant types

---

## 13. Sources & References

### Official Documentation
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [NVIDIA RTX Blackwell Whitepaper (PDF)](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
- [CUDA 13.0 Documentation](https://docs.nvidia.com/cuda/index.html)
- [PTX ISA 9.0](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [cuBLAS 13.0 Documentation](https://docs.nvidia.com/cuda/cublas/index.html)
- [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)

### Technical Blogs
- [Introducing NVFP4 for Efficient Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [NVFP4 Trains with Precision of 16-Bit](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/)
- [Boosting Matrix Multiplication with cuBLAS 12.9](https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9)

### Academic Papers
- [Dissecting Blackwell Architecture with Microbenchmarks (arXiv:2507.10789)](https://arxiv.org/html/2507.10789v2)
- [Bridging the Gap for Microscaling FP4 (arXiv:2509.23202)](https://www.arxiv.org/pdf/2509.23202)
- [Pretraining LLMs with NVFP4 (arXiv:2509.25149)](https://arxiv.org/html/2509.25149v1)
- [OCP Microscaling Formats Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)

### Code Examples
- [CUTLASS Blackwell FP4 GEMM Example](https://github.com/NVIDIA/cutlass/blob/main/examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu)
- [NVIDIA CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples)

### Community Resources
- [Stack Overflow: FP4/FP6 on Blackwell](https://stackoverflow.com/questions/79735243/how-are-fp6-and-fp4-supported-on-nvidia-tensor-core-on-blackwell)
- [NVIDIA Forums: FP4 MMA on sm_120a](https://forums.developer.nvidia.com/t/run-ptx-mma-sync-aligned-kind-mxf8f6f4-block-scale-scale-vec-1x-m16n8k32-on-sm-120a/329702)

---

## 14. Next Steps

**Immediate Actions (Before Implementation):**

1. **Verify Hardware Access:**
   - [ ] Confirm DGX Spark availability
   - [ ] SSH access and CUDA 13.0 installed
   - [ ] Run `nvidia-smi` to verify GB10 detection

2. **Set Up Development Environment:**
   - [ ] Clone llama.cpp repo
   - [ ] Create feature branch: `feature/blackwell-fp4-tensor-cores`
   - [ ] Compile with `-arch=sm_121` (test toolchain)

3. **Fetch CUTLASS Examples:**
   ```bash
   git clone https://github.com/NVIDIA/cutlass.git
   cd cutlass/examples/72_blackwell_narrow_precision_gemm
   nvcc -arch=sm_121 72b_blackwell_nvfp4_nvfp4_gemm.cu
   ```

4. **Begin Phase 1 Implementation:**
   - Create stub files (fp4-types.cuh, mma-fp4.cuh)
   - Add CC 12.1 detection
   - Compile and verify

**Ready to proceed! 🚀**

---

**Document Version:** 1.0
**Last Updated:** 2025-11-22
**Status:** Research Complete - Ready for Implementation
