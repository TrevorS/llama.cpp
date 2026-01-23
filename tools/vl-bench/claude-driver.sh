#!/bin/bash
# claude-driver.sh - Simplified interface for Claude to drive benchmarking
#
# This script provides atomic operations that Claude can call sequentially
# to profile and analyze llama.cpp performance.
#
# All output is structured with @@JSON_START@@ / @@JSON_END@@ markers
# for easy parsing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${SCRIPT_DIR}/../.."
BUILD_DIR="${LLAMA_DIR}/build"

# JSON output helper
json_out() {
    echo "@@JSON_START@@"
    echo "$1"
    echo "@@JSON_END@@"
}

cmd_status() {
    # Check system status and prerequisites
    local model_ok="false"
    local mmproj_ok="false"
    local llama_bench_ok="false"
    local llama_cli_ok="false"
    local nsys_ok="false"

    [[ -n "${MODEL_PATH:-}" && -f "${MODEL_PATH:-}" ]] && model_ok="true"
    [[ -n "${MMPROJ_PATH:-}" && -f "${MMPROJ_PATH:-}" ]] && mmproj_ok="true"
    [[ -x "${BUILD_DIR}/bin/llama-bench" ]] && llama_bench_ok="true"
    [[ -x "${BUILD_DIR}/bin/llama-cli" ]] && llama_cli_ok="true"
    command -v nsys &>/dev/null && nsys_ok="true"

    local gpu_name="unknown"
    local gpu_memory="unknown"
    if command -v nvidia-smi &>/dev/null; then
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
        gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    fi

    json_out "{
        \"command\": \"status\",
        \"ready\": $([ "$model_ok" = "true" ] && [ "$llama_bench_ok" = "true" ] && echo "true" || echo "false"),
        \"checks\": {
            \"model_path\": $model_ok,
            \"mmproj_path\": $mmproj_ok,
            \"llama_bench\": $llama_bench_ok,
            \"llama_cli\": $llama_cli_ok,
            \"nsys\": $nsys_ok
        },
        \"paths\": {
            \"model\": \"${MODEL_PATH:-not_set}\",
            \"mmproj\": \"${MMPROJ_PATH:-not_set}\",
            \"build_dir\": \"$BUILD_DIR\"
        },
        \"gpu\": {
            \"name\": \"$gpu_name\",
            \"memory\": \"$gpu_memory\"
        }
    }"
}

cmd_bench_text() {
    # Run text-only benchmark
    # Args: prompt_tokens gen_tokens [extra_flags]
    local prompt_tokens="${1:-512}"
    local gen_tokens="${2:-128}"
    local extra_flags="${3:--ngl 99 -fa 1}"

    if [[ -z "${MODEL_PATH:-}" ]]; then
        json_out '{"error": "MODEL_PATH not set"}'
        return 1
    fi

    local output
    output=$("${BUILD_DIR}/bin/llama-bench" \
        -m "$MODEL_PATH" \
        -p "$prompt_tokens" \
        -n "$gen_tokens" \
        -r 3 \
        $extra_flags \
        -o json 2>&1)

    json_out "{
        \"command\": \"bench_text\",
        \"params\": {
            \"prompt_tokens\": $prompt_tokens,
            \"gen_tokens\": $gen_tokens,
            \"flags\": \"$extra_flags\"
        },
        \"results\": $output
    }"
}

cmd_bench_compare() {
    # Run comparative benchmark: baseline vs optimized
    local prompt_tokens="${1:-512}"
    local gen_tokens="${2:-128}"

    if [[ -z "${MODEL_PATH:-}" ]]; then
        json_out '{"error": "MODEL_PATH not set"}'
        return 1
    fi

    # Baseline
    local baseline
    baseline=$("${BUILD_DIR}/bin/llama-bench" \
        -m "$MODEL_PATH" \
        -p "$prompt_tokens" -n "$gen_tokens" \
        -r 3 -ngl 99 \
        -o json 2>&1)

    # With Flash Attention
    local flash_attn
    flash_attn=$("${BUILD_DIR}/bin/llama-bench" \
        -m "$MODEL_PATH" \
        -p "$prompt_tokens" -n "$gen_tokens" \
        -r 3 -ngl 99 -fa 1 \
        -o json 2>&1)

    # With MoE on CPU (if applicable)
    local moe_cpu
    moe_cpu=$("${BUILD_DIR}/bin/llama-bench" \
        -m "$MODEL_PATH" \
        -p "$prompt_tokens" -n "$gen_tokens" \
        -r 3 -ngl 99 -fa 1 -cmoe \
        -o json 2>&1) || moe_cpu='[]'

    json_out "{
        \"command\": \"bench_compare\",
        \"params\": {
            \"prompt_tokens\": $prompt_tokens,
            \"gen_tokens\": $gen_tokens
        },
        \"baseline\": $baseline,
        \"flash_attn\": $flash_attn,
        \"moe_cpu\": $moe_cpu
    }"
}

cmd_profile_quick() {
    # Quick Nsight Systems profile
    local prompt_tokens="${1:-256}"
    local gen_tokens="${2:-32}"

    if ! command -v nsys &>/dev/null; then
        json_out '{"error": "nsys not found"}'
        return 1
    fi

    local output_dir="${SCRIPT_DIR}/results/profile_$(date +%s)"
    mkdir -p "$output_dir"

    local nsys_file="${output_dir}/profile"

    nsys profile \
        --trace=cuda,nvtx \
        --output="$nsys_file" \
        --force-overwrite=true \
        "${BUILD_DIR}/bin/llama-bench" \
        -m "$MODEL_PATH" \
        -p "$prompt_tokens" -n "$gen_tokens" \
        -r 1 -ngl 99 -fa 1 \
        2>&1 >/dev/null

    # Generate kernel stats
    nsys stats "${nsys_file}.nsys-rep" \
        --report cuda_gpu_kern_sum \
        --format json > "${output_dir}/kernels.json" 2>/dev/null || echo '[]' > "${output_dir}/kernels.json"

    # Extract top 10 kernels
    local top_kernels
    top_kernels=$(head -c 10000 "${output_dir}/kernels.json" 2>/dev/null || echo '[]')

    json_out "{
        \"command\": \"profile_quick\",
        \"output_dir\": \"$output_dir\",
        \"nsys_file\": \"${nsys_file}.nsys-rep\",
        \"top_kernels\": $top_kernels
    }"
}

cmd_analyze() {
    # Analyze results directory
    local results_dir="${1:-${SCRIPT_DIR}/results}"

    if [[ ! -d "$results_dir" ]]; then
        json_out "{\"error\": \"Results directory not found: $results_dir\"}"
        return 1
    fi

    python3 "${SCRIPT_DIR}/analyze.py" "$results_dir"
}

cmd_gpu_monitor() {
    # Get current GPU status
    if ! command -v nvidia-smi &>/dev/null; then
        json_out '{"error": "nvidia-smi not found"}'
        return 1
    fi

    local gpu_util
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "0,0,0,0,0")

    IFS=',' read -r gpu_pct mem_pct mem_used mem_total temp <<< "$gpu_util"

    json_out "{
        \"command\": \"gpu_monitor\",
        \"gpu_utilization_pct\": ${gpu_pct:-0},
        \"memory_utilization_pct\": ${mem_pct:-0},
        \"memory_used_mb\": ${mem_used:-0},
        \"memory_total_mb\": ${mem_total:-0},
        \"temperature_c\": ${temp:-0}
    }"
}

cmd_list_results() {
    # List available result directories
    local results_base="${SCRIPT_DIR}/results"

    if [[ ! -d "$results_base" ]]; then
        json_out '{"command": "list_results", "results": []}'
        return 0
    fi

    local dirs
    dirs=$(find "$results_base" -maxdepth 1 -type d -name "*_*" -printf '%f\n' 2>/dev/null | sort -r | head -20 || echo "")

    local json_dirs="["
    local first=true
    while IFS= read -r dir; do
        [[ -z "$dir" ]] && continue
        [[ "$first" == "true" ]] && first=false || json_dirs+=","
        json_dirs+="\"$dir\""
    done <<< "$dirs"
    json_dirs+="]"

    json_out "{
        \"command\": \"list_results\",
        \"results_dir\": \"$results_base\",
        \"results\": $json_dirs
    }"
}

cmd_help() {
    json_out '{
        "command": "help",
        "available_commands": {
            "status": "Check system status and prerequisites",
            "bench_text": "Run text-only benchmark (args: prompt_tokens gen_tokens [flags])",
            "bench_compare": "Run comparative benchmark with different configs",
            "profile_quick": "Run quick Nsight Systems profile",
            "analyze": "Analyze results directory",
            "gpu_monitor": "Get current GPU utilization",
            "list_results": "List available result directories",
            "help": "Show this help"
        },
        "environment_variables": {
            "MODEL_PATH": "Path to model file (required)",
            "MMPROJ_PATH": "Path to mmproj file (for VL models)"
        },
        "usage": "MODEL_PATH=/path/to/model.gguf ./claude-driver.sh <command> [args]"
    }'
}

# Main dispatcher
main() {
    local cmd="${1:-help}"
    shift || true

    case "$cmd" in
        status)       cmd_status ;;
        bench_text)   cmd_bench_text "$@" ;;
        bench_compare) cmd_bench_compare "$@" ;;
        profile_quick) cmd_profile_quick "$@" ;;
        analyze)      cmd_analyze "$@" ;;
        gpu_monitor)  cmd_gpu_monitor ;;
        list_results) cmd_list_results ;;
        help|--help|-h) cmd_help ;;
        *)
            json_out "{\"error\": \"Unknown command: $cmd\"}"
            exit 1
            ;;
    esac
}

main "$@"
