# Vision-Language Benchmark Suite for llama.cpp

A comprehensive profiling and benchmarking toolkit designed to identify performance bottlenecks in llama.cpp inference, with special focus on Vision-Language models like Qwen3-VL.

## Quick Start

```bash
# Set model paths
export MODEL_PATH=/path/to/qwen3vl-30b-a3b-q8.gguf
export MMPROJ_PATH=/path/to/qwen3vl-mmproj.gguf

# Check status
./claude-driver.sh status

# Run quick comparison benchmark
./claude-driver.sh bench_compare 512 128

# Run full benchmark suite
./run-bench.sh full
```

## Components

### 1. claude-driver.sh - Simple CLI for Automation

Provides atomic commands with JSON output for easy parsing:

| Command | Description |
|---------|-------------|
| `status` | Check prerequisites and system info |
| `bench_text <pp> <tg> [flags]` | Run text benchmark |
| `bench_compare <pp> <tg>` | Compare baseline vs optimized configs |
| `profile_quick [pp] [tg]` | Quick Nsight Systems profile |
| `analyze [dir]` | Analyze results |
| `gpu_monitor` | Get current GPU stats |
| `list_results` | List previous runs |

**Output format**: All output is wrapped in `@@JSON_START@@` and `@@JSON_END@@` markers.

### 2. run-bench.sh - Full Benchmark Suite

Runs comprehensive tests across multiple scenarios and configurations:

```bash
./run-bench.sh quick    # Smoke test (2-3 tests)
./run-bench.sh full     # Full matrix of tests
./run-bench.sh profile text_only_medium flash_attn  # Profile specific combo
./run-bench.sh list     # Show available scenarios/configs
```

### 3. analyze.py - Results Analysis

Parses benchmark results and identifies bottlenecks:

```bash
python3 analyze.py results/full_20240123/
python3 analyze.py --kernel-summary results/profile_xxx/kernels.json
```

## Test Scenarios

| ID | Type | Description |
|----|------|-------------|
| `text_only_short` | Text | 128 prompt, 64 gen tokens |
| `text_only_medium` | Text | 512 prompt, 128 gen tokens |
| `text_only_long` | Text | 2048 prompt, 128 gen tokens |
| `vl_small_image` | Vision | 512x512 image |
| `vl_medium_image` | Vision | 1024x1024 image |
| `vl_large_image` | Vision | 2048x2048 high-res image |
| `vl_detailed_analysis` | Vision | Complex analysis task |

## Configurations

| ID | Description | Flags |
|----|-------------|-------|
| `baseline` | Default | `-ngl 99` |
| `flash_attn` | Flash Attention | `-ngl 99 -fa 1` |
| `moe_cpu` | MoE experts on CPU | `-ngl 99 -fa 1 -cmoe` |
| `kv_q8` | Quantized KV cache | `-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0` |
| `moe_cpu_kv_q8` | Combined | `-ngl 99 -fa 1 -cmoe -ctk q8_0 -ctv q8_0` |
| `unified_memory` | Unified memory | `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` |
| `unified_moe_cpu` | Unified + MoE CPU | Both combined |

## Example: Claude-Driven Profiling Session

```bash
# Step 1: Check system status
./claude-driver.sh status

# Step 2: Run baseline benchmark
./claude-driver.sh bench_text 512 128 "-ngl 99"

# Step 3: Run with Flash Attention
./claude-driver.sh bench_text 512 128 "-ngl 99 -fa 1"

# Step 4: Run with MoE on CPU (for MoE models)
./claude-driver.sh bench_text 512 128 "-ngl 99 -fa 1 -cmoe"

# Step 5: Quick profile to identify kernel bottlenecks
./claude-driver.sh profile_quick 512 64

# Step 6: Analyze results
./claude-driver.sh analyze results/

# Step 7: Monitor GPU during inference
./claude-driver.sh gpu_monitor
```

## Output Structure

### Benchmark Result

```json
{
  "command": "bench_text",
  "params": {
    "prompt_tokens": 512,
    "gen_tokens": 128,
    "flags": "-ngl 99 -fa 1"
  },
  "results": [
    {
      "n_prompt": 512,
      "n_gen": 0,
      "avg_ts": 5432.10,
      "stddev_ts": 123.45
    },
    {
      "n_prompt": 0,
      "n_gen": 128,
      "avg_ts": 45.67,
      "stddev_ts": 1.23
    }
  ]
}
```

### Bottleneck Analysis

```json
{
  "type": "benchmark_analysis",
  "bottlenecks": [
    {
      "category": "moe",
      "severity": "high",
      "description": "MoE kernels account for 65% of GPU time",
      "recommendations": [
        "Try -cmoe flag to offload MoE experts to CPU"
      ]
    }
  ],
  "action_items": [
    {
      "priority": "high",
      "category": "moe",
      "action": "Try -cmoe flag to offload MoE experts to CPU"
    }
  ]
}
```

## Requirements

- llama.cpp built with CUDA support
- jq (for JSON processing)
- Python 3.6+ (for analyze.py)
- NVIDIA Nsight Systems (optional, for profiling)

## Adding Test Images

Place test images in `test_images/`:
- `512x512.jpg` - Small image
- `1024x1024.jpg` - Medium image
- `2048x2048.jpg` - Large/high-res image

## Tips for DGX Spark (GB10)

1. **Start with baseline**: Run without any optimizations first
2. **Test Flash Attention**: Should help with attention-heavy workloads
3. **Test MoE CPU offload**: Critical for large MoE models
4. **Monitor GPU utilization**: Low utilization suggests CPU bottleneck
5. **Check unified memory**: May help or hurt depending on workload

## Interpreting Results

| Metric | Good | Concerning |
|--------|------|------------|
| Prefill (pp) t/s | >1000 for small models | <500 |
| Decode (tg) t/s | >20 for 30B models | <10 |
| GPU Utilization | >70% | <30% |
| Memory Bandwidth | >200 GB/s | <100 GB/s |
