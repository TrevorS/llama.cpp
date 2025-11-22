# FP4 Tensor Core Implementation Plan

## Quick Reference Guide

**Target:** 75-85 tokens/sec for GPT-OSS-120B on DGX Spark GB10
**Timeline:** 3 weeks
**Difficulty:** ⭐⭐⭐⭐⭐ Expert

---

## 📋 3-Week Implementation Checklist

### Week 1: Foundation & PTX

#### Days 1-3: Setup & Stubs
- [ ] **Environment Setup**
  ```bash
  # On DGX Spark
  nvcc --version  # Verify CUDA 13.0+
  nvidia-smi      # Verify GB10 detected
  git checkout -b feature/blackwell-fp4-implementation
  ```

- [ ] **Code Structure**
  - [ ] Create `ggml/src/ggml-cuda/fp4-types.cuh`
  - [ ] Create `ggml/src/ggml-cuda/mma-fp4.cuh`
  - [ ] Create `ggml/src/ggml-cuda/convert-mxfp4-fp4.cuh`

- [ ] **Add Blackwell Detection** (`ggml/src/ggml-cuda/common.cuh`)
  ```cuda
  #define GGML_CUDA_CC_BLACKWELL 1210
  #define BLACKWELL_FP4_AVAILABLE (__CUDA_ARCH__ >= 1210)
  ```

- [ ] **Stub MMA Function**
  ```cuda
  template<> void mma(
      tile<16, 8, float> & D,
      const tile<16, 8, fp4_packed> & A,
      const tile<8, 4, fp4_packed> & B
  ) {
      #ifdef BLACKWELL_FP4_AVAILABLE
          // TODO: PTX
          for (int i = 0; i < 4; i++) D.x[i] = 0.0f;
      #endif
  }
  ```

- [ ] **Compile Test**
  ```bash
  cd build && cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121
  make -j
  ```

**Deliverable:** Clean compile, stub executes.

---

#### Days 4-7: PTX Implementation

- [ ] **Implement PTX Assembly** (`mma-fp4.cuh`)
  ```cuda
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::1X."
      "m16n8k32.row.col.f32.e2m1.e2m1.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %11, %12, %13}, {%14}, {%15};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
        "r"(scale_a), "r"(scale_b)
  );
  ```

- [ ] **Create Test Kernel** (`tests/test-fp4-mma.cu`)
  ```cuda
  __global__ void test_fp4_identity() {
      tile_fp4 A = make_identity();
      tile_fp4 B = make_identity();
      tile_float C = {};
      mma(C, A, B, 1, 1);
      assert(fabs(C.x[0] - 1.0f) < 0.1f);
  }
  ```

- [ ] **Debug Compilation**
  - [ ] Try variations if syntax fails
  - [ ] Check register allocation (`--ptxas-options=-v`)
  - [ ] Consult CUTLASS examples if stuck

- [ ] **Fallback: Reverse Engineer cuBLAS**
  ```bash
  nvdisasm /usr/local/cuda/lib64/libcublas.so.13 | grep -A10 "mma.*e2m1" > mma_fp4.ptx
  ```

**Deliverable:** test_fp4_identity passes on GB10.

---

### Week 2: Conversion & Integration

#### Days 8-10: MXFP4 → NVFP4 Conversion

- [ ] **Implement E2M1 Quantization** (`convert-mxfp4-fp4.cuh`)
  ```cuda
  __device__ uint8_t float_to_e2m1(float val) {
      // Quantize float to 4-bit E2M1
      // Handle: sign, exponent, mantissa, rounding
      // Return: 4-bit pattern (0-15)
  }
  ```

- [ ] **Implement Block Conversion**
  ```cuda
  __device__ void convert_mxfp4_to_nvfp4_block(
      const block_mxfp4* src,
      uint32_t* dst_packed,  // 2 registers for 16 FP4 values
      uint8_t* scale_e4m3    // E4M3 or E8M0 scale
  ) {
      float scale_fp32 = ggml_cuda_e8m0_to_fp32(src->e) * 0.5f;

      for (int i = 0; i < 16; i++) {
          uint8_t nibble = (src->qs[i/2] >> ((i%2)*4)) & 0xF;
          float val = kvalues_mxfp4[nibble] * scale_fp32;
          uint8_t fp4 = float_to_e2m1(val);

          int reg_idx = i / 8;
          int shift = (i % 8) * 4;
          dst_packed[reg_idx] |= (fp4 << shift);
      }

      *scale_e4m3 = fp32_to_e8m0(scale_fp32);  // Or E4M3 if needed
  }
  ```

- [ ] **Create Conversion Test**
  ```cuda
  __global__ void test_conversion() {
      block_mxfp4 input = {{128}, {0x01, 0x23, ...}};  // e=128, qs=...
      uint32_t nvfp4[2];
      uint8_t scale;

      convert_mxfp4_to_nvfp4_block(&input, nvfp4, &scale);

      // Verify by dequantizing both
      float ref[16], actual[16];
      dequantize_mxfp4(input, ref);
      dequantize_nvfp4(nvfp4, scale, actual);

      for (int i = 0; i < 16; i++) {
          assert(fabs(ref[i] - actual[i]) / fabs(ref[i]) < 0.15f);  // 15% tolerance
      }
  }
  ```

**Deliverable:** Conversion test passes with <15% error.

---

#### Days 11-14: Integration into llama.cpp

- [ ] **Modify `vec_dot_mxfp4_q8_1`** (`vecdotq.cuh`)
  ```cuda
  static __device__ __forceinline__ float vec_dot_mxfp4_q8_1_fp4(
      const void * __restrict__ vbq,
      const block_q8_1 * __restrict__ bq8_1,
      const int & kbx, const int & iqs
  ) {
  #ifdef BLACKWELL_FP4_AVAILABLE
      const block_mxfp4 * bq4 = (const block_mxfp4 *) vbq + kbx;

      // Convert MXFP4 to NVFP4 (tile A)
      tile<16, 8, fp4_packed> A;
      uint8_t scale_a;
      convert_mxfp4_to_tile(bq4, A, scale_a);

      // Convert Q8 to FP4 (tile B)
      tile<8, 4, fp4_packed> B;
      uint8_t scale_b;
      convert_q8_to_fp4_tile(bq8_1, iqs, B, scale_b);

      // MMA
      tile<16, 8, float> result = {};
      mma(result, A, B, scale_a, scale_b);

      // Reduce sum
      return warp_reduce_sum(result);
  #else
      // Fallback to DP4A
      return vec_dot_mxfp4_q8_1_dp4a(vbq, bq8_1, kbx, iqs);
  #endif
  }
  ```

- [ ] **Update Dispatch** (`mmq.cu`)
  ```cuda
  #if defined(BLACKWELL_FP4_AVAILABLE)
      mul_mat_q_case<GGML_TYPE_MXFP4, use_fp4_path>();
  #else
      mul_mat_q_case<GGML_TYPE_MXFP4, use_dp4a_path>();
  #endif
  ```

- [ ] **End-to-End Test**
  ```bash
  ./llama-cli -m gpt-oss-120b-mxfp4.gguf -p "Hello world" -n 32
  # Should generate coherent text (quality check)
  ```

**Deliverable:** Model runs, generates text, no crashes.

---

### Week 3: Optimization & Validation

#### Days 15-17: Performance Tuning

- [ ] **Baseline Measurement**
  ```bash
  ./llama-bench -m gpt-oss-120b-mxfp4.gguf -n 128 -r 10 > baseline.txt
  # Record: tokens/sec, latency, perplexity
  ```

- [ ] **Profile with Nsight**
  ```bash
  ncu --set full --target-processes all \
      --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
      --metrics dram__bytes_read.sum,dram__bytes_write.sum \
      --section MemoryWorkloadAnalysis \
      ./llama-bench -m gpt-oss-120b-mxfp4.gguf -n 32
  ```

- [ ] **Optimization Checklist**
  - [ ] **Shared Memory Bank Conflicts**
    - Use `__align__(16)` for tile data
    - Pad shared memory arrays to avoid conflicts

  - [ ] **Register Pressure**
    - Check occupancy in ncu report
    - Reduce intermediate variables if <50%

  - [ ] **Tile Size Tuning**
    - Try M=16/32, N=8/16, K=32/64
    - Measure throughput for each

  - [ ] **Async Memory Transfers**
    - Use `ldmatrix.sync.aligned` for loading
    - Overlap compute and memory ops

  - [ ] **Warp Specialization**
    - Separate warps for loading vs MMA (if beneficial)

- [ ] **Iterative Tuning**
  ```bash
  # After each optimization
  ./llama-bench -m gpt-oss-120b-mxfp4.gguf -n 128 -r 5
  # Target: Incremental improvement toward 75 tok/s
  ```

**Deliverable:** ≥70 tokens/sec achieved.

---

#### Days 18-21: Validation & Documentation

- [ ] **Numerical Validation**
  ```bash
  # Compare FP4 vs FP32 logits
  ./llama-cli -m gpt-oss-120b-fp32.gguf -p "prompt" --logits-out ref.txt
  ./llama-cli -m gpt-oss-120b-mxfp4.gguf -p "prompt" --logits-out fp4.txt
  python compare_logits.py ref.txt fp4.txt
  # Target: KL divergence < 0.05
  ```

- [ ] **Quality Tests**
  - [ ] Run perplexity evaluation on WikiText-2
  - [ ] Generate 1000 tokens, manual review
  - [ ] Compare to DP4A baseline (should be similar quality)

- [ ] **Regression Tests**
  ```bash
  # Ensure other quant types still work
  ./llama-bench -m model-q4_0.gguf
  ./llama-bench -m model-q8_0.gguf
  ```

- [ ] **Performance Report**
  ```
  | Metric           | Baseline (DP4A) | FP4 TC | Improvement |
  |------------------|-----------------|--------|-------------|
  | Tokens/sec       | 60.4            | 78.2   | +29.5%      |
  | SM Utilization   | 42%             | 84%    | +100%       |
  | Achieved TOPS    | 120             | 720    | +500%       |
  | Perplexity       | 12.34           | 12.48  | +1.1%       |
  ```

- [ ] **Code Review Prep**
  - [ ] Add comments to PTX assembly
  - [ ] Document tile layout assumptions
  - [ ] Add error handling for non-Blackwell GPUs

- [ ] **Create Pull Request**
  ```bash
  git add ggml/src/ggml-cuda/{fp4-types,mma-fp4,convert-mxfp4-fp4}.cuh
  git commit -m "Add FP4 tensor core support for Blackwell GB10

  - Implements m16n8k32 FP4 E2M1 MMA instruction
  - MXFP4 to NVFP4 runtime conversion
  - +29% performance improvement for GPT-OSS-120B on DGX Spark
  - Falls back to DP4A on non-Blackwell GPUs"

  git push origin feature/blackwell-fp4-implementation
  ```

**Deliverable:** Production-ready code, ≥75 tok/s, comprehensive docs.

---

## 🚨 Troubleshooting Guide

### Issue: PTX Compilation Fails

**Error:** `error: identifier "mma" is undefined`

**Solutions:**
1. Verify `-arch=sm_121` is set
2. Check `#ifdef BLACKWELL_FP4_AVAILABLE` guard
3. Try explicit `--generate-code arch=compute_121,code=sm_121`
4. Consult CUTLASS for exact syntax

### Issue: Wrong Results from MMA

**Symptoms:** Output all zeros or garbage

**Debug Steps:**
1. Print register values before MMA
2. Verify input packing (8 FP4 per register)
3. Check scale factor format (E8M0 vs E4M3)
4. Compare to CUTLASS example

### Issue: Performance Below Target

**If <70 tok/s:**
1. Profile with `ncu --set full`
2. Check SM utilization (<80% = problem)
3. Check tensor core active cycles
4. Look for memory bottlenecks
5. Try different tile sizes

### Issue: Numerical Errors Too High

**If error >20%:**
1. Check E2M1 quantization function
2. Verify scale factor conversion
3. Test with FP32 scale (no quantization)
4. Compare intermediate values to FP32 reference

---

## 📊 Success Metrics

| Milestone | Metric | Target |
|-----------|--------|--------|
| **Week 1** | PTX compiles | ✅ Clean build |
| **Week 1** | MMA executes | ✅ No crashes |
| **Week 2** | Conversion accuracy | <15% error |
| **Week 2** | Model runs | ✅ Generates text |
| **Week 3** | Performance | ≥75 tok/s |
| **Week 3** | Quality | <+2% perplexity |
| **Week 3** | SM Utilization | >80% |

---

## 🎯 Daily Checklist Template

```markdown
### Day X: [Task Name]

**Morning:**
- [ ] Review yesterday's progress
- [ ] Read relevant documentation
- [ ] Plan today's implementation

**Work:**
- [ ] Task 1: [specific goal]
- [ ] Task 2: [specific goal]
- [ ] Run tests: `./test-fp4-xxx`

**Evening:**
- [ ] Commit progress (even if incomplete)
- [ ] Document blockers
- [ ] Plan tomorrow

**Blockers:**
- Issue: [description]
- Mitigation: [what you'll try]
```

---

## 🔗 Quick Links

- **Full Research Doc:** `/home/user/llama.cpp/docs/BLACKWELL_FP4_RESEARCH.md`
- **CUTLASS Example:** [72b_blackwell_nvfp4_nvfp4_gemm.cu](https://github.com/NVIDIA/cutlass/blob/main/examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu)
- **PTX ISA:** [Section 9.7.14 (MMA Instructions)](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- **cuBLAS FP4:** [cuBLAS 12.9 Blog](https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9)

---

**Ready to start? Begin with Week 1, Day 1! 🚀**
