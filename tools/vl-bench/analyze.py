#!/usr/bin/env python3
"""
analyze.py - Analyze benchmark results from vl-bench suite

This script parses benchmark outputs and generates insights that Claude can use
to identify bottlenecks and recommend optimizations.

Usage:
    python analyze.py <results_dir>
    python analyze.py --compare <dir1> <dir2>
    python analyze.py --kernel-summary <kernel_summary.json>
"""

import json
import sys
import os
import re
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import statistics


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    scenario_id: str
    config_id: str
    prompt_tokens: int
    gen_tokens: int
    prompt_ts: float  # tokens/second for prompt
    gen_ts: float     # tokens/second for generation
    stddev_prompt: float
    stddev_gen: float
    status: str


@dataclass
class KernelStats:
    """CUDA kernel statistics"""
    name: str
    total_time_ns: int
    count: int
    avg_time_ns: float
    percentage: float


@dataclass
class BottleneckAnalysis:
    """Analysis of potential bottlenecks"""
    category: str  # "memory", "compute", "kernel_launch", "cpu", "unknown"
    severity: str  # "high", "medium", "low"
    description: str
    evidence: List[str]
    recommendations: List[str]


def parse_llama_bench_json(filepath: Path) -> List[BenchmarkResult]:
    """Parse llama-bench JSON output"""
    results = []
    try:
        with open(filepath) as f:
            data = json.load(f)

        for entry in data:
            # llama-bench outputs separate entries for pp and tg
            scenario_id = Path(filepath).stem.split('_')[0] if '_' in filepath.stem else "unknown"
            config_id = Path(filepath).stem.split('_')[-1] if '_' in filepath.stem else "unknown"

            result = BenchmarkResult(
                scenario_id=scenario_id,
                config_id=config_id,
                prompt_tokens=entry.get('n_prompt', 0),
                gen_tokens=entry.get('n_gen', 0),
                prompt_ts=entry.get('avg_ts', 0) if entry.get('n_prompt', 0) > 0 else 0,
                gen_ts=entry.get('avg_ts', 0) if entry.get('n_gen', 0) > 0 else 0,
                stddev_prompt=entry.get('stddev_ts', 0) if entry.get('n_prompt', 0) > 0 else 0,
                stddev_gen=entry.get('stddev_ts', 0) if entry.get('n_gen', 0) > 0 else 0,
                status="success"
            )
            results.append(result)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}", file=sys.stderr)

    return results


def parse_kernel_summary(filepath: Path) -> List[KernelStats]:
    """Parse Nsight Systems kernel summary"""
    kernels = []
    try:
        with open(filepath) as f:
            data = json.load(f)

        # nsys stats outputs in various formats, handle common ones
        if isinstance(data, list):
            for entry in data:
                kernel = KernelStats(
                    name=entry.get('Name', entry.get('kernel_name', 'unknown')),
                    total_time_ns=entry.get('Total Time (ns)', entry.get('total_time', 0)),
                    count=entry.get('Instances', entry.get('count', 0)),
                    avg_time_ns=entry.get('Avg (ns)', entry.get('avg_time', 0)),
                    percentage=entry.get('Time (%)', entry.get('percentage', 0))
                )
                kernels.append(kernel)
    except Exception as e:
        print(f"Error parsing kernel summary {filepath}: {e}", file=sys.stderr)

    return kernels


def analyze_results(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Analyze benchmark results and identify patterns"""

    analysis = {
        "summary": {},
        "comparisons": [],
        "bottlenecks": [],
        "recommendations": []
    }

    if not results:
        return analysis

    # Group by configuration
    by_config = {}
    for r in results:
        if r.config_id not in by_config:
            by_config[r.config_id] = []
        by_config[r.config_id].append(r)

    # Calculate summary statistics
    for config_id, config_results in by_config.items():
        prompt_speeds = [r.prompt_ts for r in config_results if r.prompt_ts > 0]
        gen_speeds = [r.gen_ts for r in config_results if r.gen_ts > 0]

        analysis["summary"][config_id] = {
            "avg_prompt_ts": statistics.mean(prompt_speeds) if prompt_speeds else 0,
            "avg_gen_ts": statistics.mean(gen_speeds) if gen_speeds else 0,
            "test_count": len(config_results)
        }

    # Compare configurations
    configs = list(by_config.keys())
    if "baseline" in configs:
        baseline_summary = analysis["summary"].get("baseline", {})
        baseline_prompt = baseline_summary.get("avg_prompt_ts", 0)
        baseline_gen = baseline_summary.get("avg_gen_ts", 0)

        for config_id, summary in analysis["summary"].items():
            if config_id == "baseline":
                continue

            prompt_speedup = summary["avg_prompt_ts"] / baseline_prompt if baseline_prompt > 0 else 0
            gen_speedup = summary["avg_gen_ts"] / baseline_gen if baseline_gen > 0 else 0

            analysis["comparisons"].append({
                "config": config_id,
                "vs_baseline": {
                    "prompt_speedup": round(prompt_speedup, 2),
                    "gen_speedup": round(gen_speedup, 2)
                }
            })

    return analysis


def analyze_kernels(kernels: List[KernelStats]) -> List[BottleneckAnalysis]:
    """Analyze kernel statistics to identify bottlenecks"""
    bottlenecks = []

    if not kernels:
        return bottlenecks

    total_time = sum(k.total_time_ns for k in kernels)

    # Sort by time
    sorted_kernels = sorted(kernels, key=lambda k: k.total_time_ns, reverse=True)

    # Identify dominant kernels
    top_kernels = sorted_kernels[:5]
    top_time = sum(k.total_time_ns for k in top_kernels)
    top_percentage = (top_time / total_time * 100) if total_time > 0 else 0

    # Check for specific patterns
    moe_kernels = [k for k in kernels if 'moe' in k.name.lower() or 'topk' in k.name.lower()]
    attn_kernels = [k for k in kernels if 'attn' in k.name.lower() or 'flash' in k.name.lower()]
    matmul_kernels = [k for k in kernels if 'mul_mat' in k.name.lower() or 'gemm' in k.name.lower()]
    conv_kernels = [k for k in kernels if 'conv' in k.name.lower()]

    # MoE analysis
    if moe_kernels:
        moe_time = sum(k.total_time_ns for k in moe_kernels)
        moe_pct = (moe_time / total_time * 100) if total_time > 0 else 0

        if moe_pct > 30:
            bottlenecks.append(BottleneckAnalysis(
                category="moe",
                severity="high" if moe_pct > 50 else "medium",
                description=f"MoE kernels account for {moe_pct:.1f}% of GPU time",
                evidence=[f"{k.name}: {k.percentage:.1f}%" for k in moe_kernels[:3]],
                recommendations=[
                    "Try -cmoe flag to offload MoE experts to CPU",
                    "Check expert utilization balance",
                    "Consider quantized expert weights"
                ]
            ))

    # Attention analysis
    if attn_kernels:
        attn_time = sum(k.total_time_ns for k in attn_kernels)
        attn_pct = (attn_time / total_time * 100) if total_time > 0 else 0

        if attn_pct > 40:
            has_flash = any('flash' in k.name.lower() for k in attn_kernels)
            bottlenecks.append(BottleneckAnalysis(
                category="attention",
                severity="medium",
                description=f"Attention kernels account for {attn_pct:.1f}% of GPU time",
                evidence=[f"{k.name}: {k.percentage:.1f}%" for k in attn_kernels[:3]],
                recommendations=[
                    "Verify Flash Attention is enabled (-fa 1)" if not has_flash else "Flash Attention is active",
                    "Consider quantized KV cache (-ctk q8_0 -ctv q8_0)",
                    "Check for optimal batch size"
                ]
            ))

    # Many small kernels = launch overhead
    small_kernels = [k for k in kernels if k.avg_time_ns < 10000]  # < 10 microseconds
    if len(small_kernels) > len(kernels) * 0.5:
        bottlenecks.append(BottleneckAnalysis(
            category="kernel_launch",
            severity="medium",
            description=f"{len(small_kernels)} kernels with <10µs avg runtime detected",
            evidence=[f"Total kernels: {len(kernels)}, Small kernels: {len(small_kernels)}"],
            recommendations=[
                "Consider larger batch sizes",
                "Check for kernel fusion opportunities",
                "CUDA graphs may help (if supported)"
            ]
        ))

    # Vision encoder analysis
    if conv_kernels:
        conv_time = sum(k.total_time_ns for k in conv_kernels)
        conv_pct = (conv_time / total_time * 100) if total_time > 0 else 0

        if conv_pct > 10:
            bottlenecks.append(BottleneckAnalysis(
                category="vision_encoder",
                severity="low" if conv_pct < 20 else "medium",
                description=f"Conv2D kernels account for {conv_pct:.1f}% of GPU time",
                evidence=[f"{k.name}: {k.percentage:.1f}%" for k in conv_kernels[:3]],
                recommendations=[
                    "Conv2D is used for patch embedding - usually not the main bottleneck",
                    "Check if vision encoder is using Flash Attention"
                ]
            ))

    return bottlenecks


def generate_report(analysis: Dict[str, Any], bottlenecks: List[BottleneckAnalysis]) -> Dict[str, Any]:
    """Generate a structured report for Claude to process"""

    report = {
        "type": "benchmark_analysis",
        "summary": analysis.get("summary", {}),
        "configuration_comparisons": analysis.get("comparisons", []),
        "bottlenecks": [asdict(b) for b in bottlenecks],
        "action_items": []
    }

    # Generate prioritized action items
    high_severity = [b for b in bottlenecks if b.severity == "high"]
    medium_severity = [b for b in bottlenecks if b.severity == "medium"]

    for b in high_severity:
        for rec in b.recommendations[:2]:
            report["action_items"].append({
                "priority": "high",
                "category": b.category,
                "action": rec
            })

    for b in medium_severity:
        for rec in b.recommendations[:1]:
            report["action_items"].append({
                "priority": "medium",
                "category": b.category,
                "action": rec
            })

    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze vl-bench results")
    parser.add_argument("results_dir", nargs="?", help="Results directory to analyze")
    parser.add_argument("--compare", nargs=2, metavar=("DIR1", "DIR2"), help="Compare two result sets")
    parser.add_argument("--kernel-summary", help="Analyze kernel summary JSON")
    parser.add_argument("--output", "-o", default="-", help="Output file (default: stdout)")

    args = parser.parse_args()

    results = []
    bottlenecks = []

    if args.results_dir:
        results_path = Path(args.results_dir)
        if results_path.is_dir():
            for json_file in results_path.glob("*.json"):
                if json_file.name != "summary.json":
                    results.extend(parse_llama_bench_json(json_file))

    if args.kernel_summary:
        kernels = parse_kernel_summary(Path(args.kernel_summary))
        bottlenecks = analyze_kernels(kernels)

    analysis = analyze_results(results)
    report = generate_report(analysis, bottlenecks)

    # Output
    output_json = json.dumps(report, indent=2)

    if args.output == "-":
        print("@@JSON_START@@")
        print(output_json)
        print("@@JSON_END@@")
    else:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
