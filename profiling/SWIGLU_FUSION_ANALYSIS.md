# SwiGLU Fusion Deep Dive for gpt-oss-20b

## 1. What is SwiGLU?

### Standard SwiGLU (Swish-Gated Linear Unit)

SwiGLU is an activation function used in transformer FFN (Feed-Forward Network) layers. Introduced by Shazeer (2020), it combines:
- **Swish activation**: `swish(x) = x * sigmoid(x)`
- **Gating mechanism**: Element-wise multiplication with a gate tensor

**Mathematical formula:**
```
SwiGLU(x, gate) = Swish(gate) * x
               = (gate * sigmoid(gate)) * x
```

### OpenAI Variant (SwiGLU-OAI)

gpt-oss-20b uses a modified version with additional parameters:
- `alpha = 1.702` (steeper sigmoid)
- `limit = 7.0` (clamping for numerical stability)

**Formula from `unary.cuh:103-110`:**
```c
float ggml_cuda_op_swiglu_oai_single(float x, float g, float alpha = 1.702f, float limit = 7.0f) {
    x = fminf(x, limit);                        // Clamp input
    g = fmaxf(fminf(g, limit), -limit);         // Clamp gate symmetrically

    float out_glu = x / (1.0f + expf(-x * alpha));  // Modified swish: x * sigmoid(alpha*x)
    out_glu = out_glu * (1.0f + g);             // Multiply by (1 + gate)
    return out_glu;
}
```

**Key differences from standard SwiGLU:**
1. Uses `sigmoid(alpha * x)` instead of `sigmoid(x)` (steeper gradient)
2. Uses `(1 + gate)` instead of just `gate` (bias towards keeping signal)
3. Clamping for numerical stability with large values

---

## 2. Current Implementation in llama.cpp

### Data Flow for gpt-oss-20b MoE FFN

From `src/llama-graph.cpp:1094-1161`:

```
Input: cur [n_embd, 1, n_tokens]
           │
           ├─────────────────────────────────┐
           │                                 │
           ▼                                 ▼
┌─────────────────────┐         ┌─────────────────────┐
│ Gate Projection     │         │ Up Projection       │
│ build_lora_mm_id    │         │ build_lora_mm_id    │
│ (gate_exps)         │         │ (up_exps)           │
└─────────────────────┘         └─────────────────────┘
           │                                 │
           │ gate [n_ff, n_expert_used, n_tokens]
           │                                 │ up [n_ff, n_expert_used, n_tokens]
           │                                 │
           └──────────┬──────────────────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ ggml_swiglu_oai     │  ◄── SEPARATE KERNEL
           │ (cur, up, alpha,    │
           │  limit)             │
           └─────────────────────┘
                      │
                      │ activated [n_ff, n_expert_used, n_tokens]
                      │
                      ▼
           ┌─────────────────────┐
           │ Down Projection     │
           │ build_lora_mm_id    │
           │ (down_exps)         │
           └─────────────────────┘
                      │
                      ▼
           Output: [n_embd, n_expert_used, n_tokens]
```

### Current Kernel Sequence

For each token, the CUDA execution is:
1. **Kernel 1**: `mul_mat_id` for gate projection
2. **Kernel 2**: `mul_mat_id` for up projection
3. **Kernel 3**: `swiglu_oai_kernel` for activation ◄── **FUSION TARGET**
4. **Kernel 4**: `mul_mat_id` for down projection

**Problems with current approach:**
- **Memory bandwidth waste**: Kernel 3 reads gate+up from global memory, writes result back
- **Kernel launch overhead**: Extra kernel launch for relatively simple operation
- **No data reuse**: Gate and up tensors computed in kernels 1&2, stored, then re-read in kernel 3

---

## 3. What "Fusion" Means

### Definition
Kernel fusion combines multiple GPU operations into a single kernel, eliminating:
1. **Intermediate memory traffic**: No write-then-read between fused operations
2. **Kernel launch overhead**: Single dispatch instead of multiple
3. **Synchronization points**: GPU stays busy without idle gaps

### Fusion Opportunity for SwiGLU

**Without fusion (current):**
```
gate_proj output ──► Global Memory ──► swiglu ──► Global Memory
up_proj output   ──► Global Memory ──► swiglu ──►
```

**With fusion:**
```
gate_proj ──► Registers ──► swiglu ──► Global Memory (or directly to down_proj)
up_proj   ──► Registers ──►
```

### Memory Savings Calculation

For gpt-oss-20b with n_ff=5120, n_tokens=1, n_expert_used=2:
- Without fusion: 2 writes + 2 reads = 4 × 5120 × 2 × 4 bytes = 163,840 bytes per layer
- With fusion: 0 intermediate traffic

At 273 GB/s bandwidth (DGX Spark), this saves ~0.6 µs per layer × 28 layers = ~17 µs per token.

---

## 4. Existing Fusion Infrastructure

### Good News: Fusion Already Exists!

llama.cpp already has SwiGLU fusion infrastructure in the matmul-vector kernels:

**From `ggml/src/ggml-cuda/common.cuh:1249-1260`:**
```c
struct ggml_cuda_mm_fusion_args_host {
    const ggml_tensor * x_bias = nullptr;
    const ggml_tensor * gate = nullptr;          // ◄── Gate tensor for fusion
    const ggml_tensor * gate_bias = nullptr;
    ggml_glu_op glu_op;                          // ◄── GGML_GLU_OP_SWIGLU_OAI
};
```

**From `ggml/src/ggml-cuda/mmvf.cu:328-342`:**
```c
switch (glu_op) {
    case GGML_GLU_OP_SWIGLU:
        value *= ggml_cuda_op_silu_single(gate_value);
        break;
    case GGML_GLU_OP_GEGLU:
        value *= ggml_cuda_op_gelu_single(gate_value);
        break;
    case GGML_GLU_OP_SWIGLU_OAI: {
        value = ggml_cuda_op_swiglu_oai_single(gate_value, value);  // ◄── Already implemented!
        break;
    }
}
```

### Current Limitation

The fusion is implemented in:
- `mmvf.cu` (float matmul-vector)
- `mmvq.cu` (quantized matmul-vector)

But **NOT** in:
- `mul_mat_id` operations (used by MoE for expert dispatch)
- The larger batched matrix multiplications

---

## 5. Implementation Strategy

### Option A: Fuse SwiGLU into mul_mat_id (Recommended)

**What to modify:** `ggml/src/ggml-cuda/` matmul operations used by MoE

**Steps:**
1. Add fusion args to `ggml_cuda_mul_mat_id` dispatch
2. Extend `mmvq.cu` ID kernels to support fusion
3. Modify `build_moe_ffn` to pass gate tensor for fusion

**Pros:**
- Leverages existing fusion infrastructure
- Works with all quantization types
- Clean integration with MoE dispatch

**Cons:**
- Need to handle ID-based indexing in fusion
- Gate and up projections share input, but fusion assumes gate is already computed

### Option B: Fused Gate+Up Projection Kernel (Higher Impact)

**What to modify:** Create new fused kernel that computes both projections

**Current (2 kernels):**
```
cur ──► gate_exps ──► gate_out
cur ──► up_exps   ──► up_out
```

**Fused (1 kernel):**
```
cur ──► [gate_exps, up_exps] ──► [gate_out, up_out]
```

**Pros:**
- Eliminates redundant input reads
- More significant speedup (~2x reduction in input memory traffic)

**Cons:**
- Requires new kernel template
- More complex weight tensor layout

### Option C: Full MoE FFN Fusion (Maximum Impact)

Fuse entire sequence: gate_proj + up_proj + SwiGLU + down_proj

**This is what Liger Kernel does**, achieving 22-24% speedup.

**Challenges:**
- Very large shared memory requirements
- Complex implementation for MoE with expert dispatch
- May not fit in registers for large hidden dimensions

---

## 6. Recommended Implementation Plan

### Phase 1: Enable SwiGLU Fusion in Existing Kernels (Quick Win)

1. **Modify graph construction** (`llama-graph.cpp`):
   - Change `ggml_swiglu_oai(cur, up, ...)` to be a fusion hint
   - Pass fusion args through `mul_mat_id` for down projection

2. **Extend `mmvq.cu` ID path**:
   - Add fusion support similar to `mmvf.cu`
   - Handle MXFP4 quantization path

**Expected improvement:** 5-10% on decode, 2-5% on prefill

### Phase 2: Fused Gate+Up Projection

1. **Create new op type**: `GGML_OP_MUL_MAT_DUAL_ID`
2. **Implement fused kernel**: Reads input once, writes two outputs
3. **Modify `build_moe_ffn`**: Use new dual matmul

**Expected improvement:** Additional 5-10%

### Phase 3: Full FFN Fusion (Advanced)

1. **Analyze memory requirements** for full fusion
2. **Implement tiled approach** if shared memory insufficient
3. **Special case for decode** (single token, smaller tiles)

**Expected improvement:** Additional 5-15%

---

## 7. Files to Modify

```
ggml/src/ggml-cuda/
├── mmvq.cu           # Add fusion to ID-based quantized matmul
├── mmid.cu           # Modify ID helper for fusion
├── ggml-cuda.cu      # Wire up fusion dispatch for MoE ops
└── common.cuh        # Potentially extend fusion args

src/
├── llama-graph.cpp   # Modify build_moe_ffn to request fusion
└── llama-graph.h     # Add fusion flag to FFN builder
```

---

## 8. Validation Checklist

After implementing fusion:

1. **Correctness**:
   - [ ] Run perplexity test, ensure PPL within 0.1% of baseline
   - [ ] Compare output tensors numerically (max diff < 1e-5)

2. **Performance**:
   - [ ] Run baseline benchmark before/after
   - [ ] Profile with Nsight to verify kernel reduction
   - [ ] Measure actual memory bandwidth savings

3. **Compatibility**:
   - [ ] Test with different batch sizes (1, 8, 32)
   - [ ] Test with different context lengths (512, 4096, 16384)
   - [ ] Verify MXFP4 quantization still works

---

## References

- [SwiGLU Paper (Shazeer 2020)](https://arxiv.org/abs/2002.05202)
- [Liger Kernel GitHub](https://github.com/linkedin/Liger-Kernel)
- [gpt-oss Architecture](https://github.com/openai/gpt-oss)
- [llama.cpp CUDA Backend](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda)
