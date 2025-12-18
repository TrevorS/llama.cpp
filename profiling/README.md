# gpt-oss-20b Profiling Harness for DGX Spark

A comprehensive profiling and optimization framework for running gpt-oss-20b on NVIDIA DGX Spark (Grace Blackwell GB10). Designed for iterative optimization with Claude Code.

## Overview

### Model: gpt-oss-20b
- **Architecture**: Mixture of Experts (MoE)
- **Total Parameters**: ~20B
- **Active Parameters**: ~3.6-4B per inference
- **Activation**: SwiGLU (OpenAI variant)
- **Native Quantization**: MXFP4
- **Context Length**: 128K tokens

### Hardware: DGX Spark (GB10)
- **Memory**: 128GB unified LPDDR5x (273 GB/s)
- **AI Performance**: 1 PFLOP (FP4 with sparsity)
- **Tensor Cores**: 5th generation (FP4/FP8 support)
- **Interconnect**: NVLink-C2C (900 GB/s CPU-GPU)

## Quick Start

```bash
# 1. Set model path
export MODEL_PATH=/path/to/gpt-oss-20b.gguf

# 2. Build llama.cpp with CUDA
cmake -B build-cuda -DGGML_CUDA=ON
cmake --build build-cuda -j

# 3. Run baseline benchmarks
./profiling/01_baseline_benchmark.sh

# 4. Run application-level profiling
./profiling/02_nsight_systems_profile.sh

# 5. Run kernel-level profiling
./profiling/03_nsight_compute_profile.sh quick
```

## Directory Structure

```
profiling/
├── profile_config.sh           # Configuration and settings
├── 01_baseline_benchmark.sh    # Baseline throughput measurements
├── 02_nsight_systems_profile.sh # Application-level GPU profiling
├── 03_nsight_compute_profile.sh # Deep kernel analysis
├── 04_compare_runs.sh          # Compare optimization iterations
├── 05_optimization_workflow.sh  # Guided optimization workflow
├── README.md                   # This file
└── results/                    # Generated results
    ├── baseline/               # Baseline benchmark results
    ├── optimization/           # Optimization tracking
    └── nsight_reports/         # Nsight profiling reports
```

## Key CUDA Kernels to Optimize

For gpt-oss-20b MoE model, these are the critical kernels:

| Kernel Category | Files | Expected % Time | Notes |
|----------------|-------|-----------------|-------|
| MoE Routing | `topk-moe.cu` | <5% | Expert selection |
| Matrix Multiply | `mmq.cuh`, `mmvq.cu` | 50-70% | MXFP4 quantized |
| SwiGLU | `unary.cu` | 10-20% | Can be fused |
| Attention | `fattn*.cu` | 15-25% | Memory-bound |
| RoPE | `rope.cu` | <5% | Position encoding |
| RMS Norm | `norm.cu` | <2% | Normalization |

### Key Files

```
ggml/src/ggml-cuda/
├── topk-moe.cu          # MoE top-k routing kernel
├── mmid.cu              # MoE expert dispatch
├── unary.cu             # SwiGLU and other activations
├── mmq.cuh              # Quantized matrix multiply
├── mmvq.cu              # Matrix-vector quantized multiply
├── fattn.cu             # Flash attention
├── fattn-common.cuh     # Attention common code
├── softmax.cuh          # Softmax kernel
└── rope.cu              # Rotary position embeddings
```

## Optimization Workflow with Claude Code

### Step 1: Establish Baseline
```bash
# Run complete baseline
./profiling/01_baseline_benchmark.sh

# Note your baseline RUN_ID from the output
```

### Step 2: Profile to Find Bottlenecks
```bash
# Application-level timeline
./profiling/02_nsight_systems_profile.sh

# Deep kernel analysis
./profiling/03_nsight_compute_profile.sh

# View results
nsys-ui profiling/nsight_reports/nsys_*.nsys-rep
ncu-ui profiling/nsight_reports/ncu_*.ncu-rep
```

### Step 3: Start Optimization Iteration
```bash
# Start tracking an optimization
./profiling/05_optimization_workflow.sh start "Fuse SwiGLU with gate projection"

# Make code changes...
# Edit ggml/src/ggml-cuda/unary.cu or other files

# Rebuild
cmake --build build-cuda -j

# Benchmark the optimization
./profiling/05_optimization_workflow.sh benchmark opt_YYYYMMDD_HHMMSS
```

### Step 4: Compare Results
```bash
# Compare with baseline
./profiling/04_compare_runs.sh baseline_YYYYMMDD opt_YYYYMMDD

# Or use workflow
./profiling/05_optimization_workflow.sh compare baseline_id opt_id
```

### Step 5: Iterate or Accept
- If improved: Record changes and continue
- If regressed: Revert and try different approach

```bash
# Record what files changed
./profiling/05_optimization_workflow.sh record opt_YYYYMMDD "file1.cu,file2.cu"

# View history
./profiling/05_optimization_workflow.sh history
```

## Claude Code Integration

### Asking Claude Code to Optimize

**Good prompts:**
```
"Profile the SwiGLU kernel and suggest optimizations based on the Nsight Compute report"

"The mmq kernel shows 45% memory bandwidth utilization. How can we improve it?"

"Implement kernel fusion for the gate and up projection in the MoE FFN layer"

"The topk_moe kernel has warp divergence. Propose a fix."
```

**Providing context:**
```
"Here are the profiling results: [paste nsight output]
Current kernel implementation: [paste code]
Suggest specific optimizations for DGX Spark's unified memory architecture"
```

### Reading Profiling Results

Claude Code can help interpret:
1. **Nsight Systems timelines** - Look for gaps, CPU overhead
2. **Nsight Compute metrics** - SOL%, occupancy, memory throughput
3. **Roofline analysis** - Compute vs memory bound determination

### Example Optimization Session

```
Human: Run the baseline benchmark for gpt-oss-20b

Claude: [Runs 01_baseline_benchmark.sh, reports results]