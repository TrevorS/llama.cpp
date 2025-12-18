# Claude Code Guide for gpt-oss-20b Optimization

This guide explains how to work with Claude Code for iterative CUDA kernel optimization.

## Understanding the Profiling Data

### What to Look For in Baseline Results

When Claude Code runs `01_baseline_benchmark.sh`, examine:

```
Key Metrics:
- pp (prompt processing) tokens/sec: Prefill throughput
- tg (token generation) tokens/sec: Decode throughput
- Standard deviation: Consistency of measurements
```

**Expected ranges for gpt-oss-20b on DGX Spark:**
- Prefill (pp2048): ~2000 t/s
- Decode (tg32): ~60 t/s

### Interpreting Nsight Systems Reports

After `02_nsight_systems_profile.sh`:

1. **Timeline View** (`nsys-ui`):
   - Look for: Gaps between kernel launches (CPU bottleneck)
   - Look for: Synchronization points (unnecessary waits)
   - Look for: Memory copies (should be minimal with unified memory)

2. **Kernel Summary CSV**:
   ```
   Kernel Name          | Time %  | Avg Duration | Count
   ---------------------|---------|--------------|------
   mmq_xxx              | 55%     | 125 µs       | 1024
   swiglu_oai_kernel    | 15%     | 45 µs        | 512
   fattn_xxx            | 20%     | 200 µs       | 256
   topk_moe_cuda        | 3%      | 8 µs         | 512
   ```

   **Analysis:**
   - High time % kernels are optimization targets
   - Many small kernels may indicate fusion opportunities
   - Compare `Count` vs expected (should match layer count)

### Interpreting Nsight Compute Reports

After `03_nsight_compute_profile.sh`:

1. **Speed of Light (SOL) Metrics**:
   - `Compute (SM) Throughput`: How much of GPU compute capacity used
   - `Memory Throughput`: How much of memory bandwidth used
   - Target: >70% for the limiting resource

2. **Roofline Position**:
   - Below ceiling = room for optimization
   - Left of ridge point = memory-bound
   - Right of ridge point = compute-bound

3. **Occupancy**:
   - Theoretical vs achieved occupancy
   - Low occupancy = consider launch configuration changes

4. **Warp State Statistics**:
   - `Stall Wait` = memory latency issues
   - `Stall Barrier` = synchronization overhead
   - `Stall Not Selected` = insufficient parallelism

## Optimization Strategies for gpt-oss-20b

### 1. MoE Routing (topk_moe.cu)

**Current Implementation** (`ggml/src/ggml-cuda/topk-moe.cu`):
- Fused softmax + top-k + normalization
- Template-based for different expert counts

**Potential Optimizations:**
- [ ] Expert batching for better memory coalescing
- [ ] Prefetch expert weights based on routing prediction
- [ ] Reduce warp divergence in argmax reduction

**How to ask Claude Code:**
```
"Analyze topk-moe.cu:63-167 for warp divergence patterns.
The model uses 64 experts with top-2 selection."
```

### 2. Matrix Multiplication (mmq.cuh, mmvq.cu)

**Current Implementation:**
- MXFP4 quantized matmul
- Uses dp4a and tensor core paths

**Potential Optimizations:**
- [ ] Tune tile sizes for GB10's cache hierarchy
- [ ] Optimize shared memory layout for bank conflicts
- [ ] Use async copy (cp.async) for prefetching

**How to ask Claude Code:**
```
"The MXFP4 matmul shows 45% compute throughput.
Examine mmq.cuh and suggest tile size optimizations
for 128GB unified memory with 273 GB/s bandwidth."
```

### 3. SwiGLU Activation (unary.cu)

**Current Implementation** (`ggml/src/ggml-cuda/unary.cu:349-416`):
- Separate kernel for SwiGLU-OAI variant
- Takes gate and input tensors separately

**Potential Optimizations:**
- [ ] Fuse with preceding matmul (gate projection)
- [ ] Fuse with following matmul (down projection)
- [ ] Use vectorized loads/stores

**How to ask Claude Code:**
```
"The SwiGLU kernel is 15% of runtime.
Propose a fused kernel that combines:
1. Gate projection matmul
2. Up projection matmul
3. SwiGLU activation
4. Down projection matmul

Reference: Liger Kernel achieves 22% speedup with this fusion."
```

### 4. Attention (fattn*.cu)

**Current Implementation:**
- Flash attention variants
- MMA-based and vector-based paths

**Potential Optimizations:**
- [ ] KV cache management for long contexts
- [ ] Tune for unified memory access patterns
- [ ] Optimize softmax for numerical stability

**How to ask Claude Code:**
```
"Attention is memory-bound at 85% bandwidth utilization.
With 128K context support, how can we optimize
KV cache access patterns for DGX Spark's unified memory?"
```

## Step-by-Step Optimization Session

### Session Start
```
Human: Let's optimize gpt-oss-20b for DGX Spark.
       First, run the baseline benchmark.

Claude: I'll run the baseline benchmark to establish our starting point.
        [Runs ./profiling/01_baseline_benchmark.sh]

        Results:
        - Prefill (pp2048): 1856 t/s
        - Decode (tg32): 58.3 t/s

        These are good starting numbers. Let me run profiling
        to identify optimization opportunities.
```

### Profile Analysis
```
Human: Profile the key kernels.

Claude: [Runs ./profiling/02_nsight_systems_profile.sh]
        [Runs ./profiling/03_nsight_compute_profile.sh quick]

        Key findings from profiling:
        1. mmq kernels: 52% of time, 67% compute SOL
        2. swiglu_oai_kernel: 18% of time, unfused
        3. fattn: 22% of time, memory-bound at 78% BW
        4. topk_moe: 4% of time, some warp divergence

        Recommendation: Start with SwiGLU fusion for quick win.
```

### Make Changes
```
Human: Let's implement the SwiGLU fusion.

Claude: I'll start an optimization iteration.
        [Runs ./profiling/05_optimization_workflow.sh start "Fuse SwiGLU"]

        Now let me modify unary.cu to implement the fusion...
        [Makes code changes]

        Changes made to:
        - ggml/src/ggml-cuda/unary.cu
        - ggml/src/ggml-cuda/mmvf.cu

        Rebuilding...
        [Runs cmake --build build-cuda -j]
```

### Validate
```
Human: Test the optimization.

Claude: [Runs ./profiling/05_optimization_workflow.sh benchmark opt_xxx]
        [Runs ./profiling/04_compare_runs.sh baseline_xxx opt_xxx]

        Results:
        - Prefill: 1856 -> 2012 t/s (+8.4%)
        - Decode: 58.3 -> 61.2 t/s (+5.0%)

        The SwiGLU fusion improved performance.
        Recording changes...
```

## Common Issues and Solutions

### Issue: Build Fails After Changes
```bash
# Check CUDA compilation errors
cmake --build build-cuda -j 2>&1 | grep error

# Common fixes:
# - Template instantiation errors: Check type deduction
# - Shared memory overflow: Reduce tile sizes
# - Register pressure: Use __launch_bounds__
```

### Issue: Performance Regression
```
1. Verify correctness first:
   ./build-cuda/bin/llama-perplexity -m model.gguf ...

2. Check Nsight for new bottlenecks:
   - Did we introduce sync points?
   - Did memory access patterns change?

3. Compare kernel-level metrics:
   ncu --import old.ncu-rep --import new.ncu-rep
```

### Issue: Numerical Accuracy Changes
```
# Run perplexity comparison
./build-cuda/bin/llama-perplexity -m model.gguf -f wiki.test.raw

# Expected: PPL should be within 0.1% of baseline
# If worse: Check float->half conversions, accumulation order
```

## Benchmarking Best Practices

1. **Warm up** before timing (skip first few iterations)
2. **Multiple runs** (at least 5) for statistical significance
3. **Control variables**: Same prompt length, context depth
4. **Check for throttling**: Monitor GPU temperature/clocks
5. **Isolate changes**: One optimization at a time

## Reference Performance Targets

| Metric | Baseline | Good | Excellent |
|--------|----------|------|-----------|
| Prefill pp2048 | 2000 t/s | 2500 t/s | 3000 t/s |
| Decode tg32 | 60 t/s | 70 t/s | 80 t/s |
| MM Compute SOL | 50% | 70% | 85% |
| Attention BW SOL | 75% | 85% | 95% |

## Files Modified During Optimization

Track all files that may be modified:

```
ggml/src/ggml-cuda/
├── topk-moe.cu          # MoE routing changes
├── mmid.cu              # Expert dispatch changes
├── unary.cu             # Activation fusion
├── mmq.cuh              # Matmul optimization
├── mmvq.cu              # Vector matmul
├── mmvf.cu              # Float matmul with fusion
├── fattn*.cu            # Attention optimization
├── common.cuh           # Shared utilities
└── ggml-cuda.cu         # Kernel dispatch

src/models/
└── openai-moe-iswa.cpp  # Model graph construction
```

Always run `git diff` before committing to review all changes.
