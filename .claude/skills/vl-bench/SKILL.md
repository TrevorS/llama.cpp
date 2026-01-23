---
name: vl-bench
description: Profile and benchmark Vision-Language model inference in llama.cpp. Use for performance analysis, bottleneck detection, and optimization testing on Qwen3-VL and similar models.
allowed-tools: Bash, Read, Write, Grep, Glob
argument-hint: <command> [args]
---

# Vision-Language Benchmark Suite

Profile and optimize VL model inference in llama.cpp.

## Commands

Run with: `/vl-bench <command> [args]`

| Command | Description |
|---------|-------------|
| `status` | Check prerequisites (model paths, GPU, build) |
| `bench_quick` | Fast smoke test with baseline config |
| `bench_sweep` | Full benchmark matrix across all configs |
| `profile [scenario]` | GPU kernel profiling with Nsight Systems |
| `compare <config1> <config2>` | Side-by-side configuration comparison |
| `gpu` | Real-time GPU utilization monitoring |
| `help` | Show available commands |

## Quick Start

```bash
# 1. Check system readiness
/vl-bench status

# 2. Run quick baseline benchmark
/vl-bench bench_quick

# 3. Compare Flash Attention vs baseline
/vl-bench compare baseline flash_attn

# 4. Profile to find bottlenecks
/vl-bench profile text_only_medium
```

## Environment Variables

Set before running:
```bash
export MODEL_PATH=/path/to/qwen3vl-30b-a3b-q8.gguf
export MMPROJ_PATH=/path/to/qwen3vl-mmproj.gguf
```

## Available Configurations

| ID | Description | Flags |
|----|-------------|-------|
| `baseline` | Default | `-ngl 99` |
| `flash_attn` | Flash Attention | `-ngl 99 -fa 1` |
| `moe_cpu` | MoE experts on CPU | `-ngl 99 -fa 1 -cmoe` |
| `kv_q8` | Quantized KV cache | `-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0` |
| `moe_cpu_kv_q8` | Combined optimizations | `-ngl 99 -fa 1 -cmoe -ctk q8_0 -ctv q8_0` |

## Test Scenarios

| ID | Type | Description |
|----|------|-------------|
| `text_only_short` | Text | 128 prompt, 64 gen tokens |
| `text_only_medium` | Text | 512 prompt, 128 gen tokens |
| `text_only_long` | Text | 2048 prompt, 128 gen tokens |
| `vl_small_image` | Vision | 512x512 image |
| `vl_medium_image` | Vision | 1024x1024 image |
| `vl_large_image` | Vision | 2048x2048 high-res image |

## Workflow for Optimization

1. **Baseline**: Run `status` then `bench_quick` to establish baseline
2. **Compare**: Use `compare` to test different configurations
3. **Profile**: Run `profile` on scenarios showing issues
4. **Iterate**: Adjust flags based on bottleneck analysis

## Output Format

Results include:
- Human-readable summary
- Suggested next steps
- JSON data between `@@JSON@@` markers

## Implementation

This skill uses `tools/vl-bench/claude-driver.sh` which is optimized for Claude Code automation with:
- Feedback loops (summary + next steps)
- State persistence for session resumption
- Graceful error handling with fix suggestions

## Interpreting Results

| Metric | Good | Concerning |
|--------|------|------------|
| Prefill (pp) t/s | >1000 | <500 |
| Decode (tg) t/s | >20 for 30B | <10 |
| GPU Utilization | >70% | <30% |
