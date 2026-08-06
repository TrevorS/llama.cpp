#!/bin/bash
# claude-driver.sh - Optimized for Claude Code automation
#
# Design principles (from Anthropic Claude Code best practices):
# 1. Feedback loops - Show what happened + suggest next actions
# 2. Dual output - Human-readable summary + machine-parseable JSON
# 3. Graceful errors - Clear messages with fix suggestions
# 4. State persistence - Save progress for resumption
# 5. Iteration support - Designed for run/analyze/iterate cycles
#
# Output format:
#   Human-readable text first (for Claude to understand context)
#   Then JSON between @@JSON@@ markers (for structured parsing)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${SCRIPT_DIR}/../.."

# Docker detection and path configuration
IN_DOCKER="false"
if [[ -f /.dockerenv ]] || grep -q docker /proc/1/cgroup 2>/dev/null; then
    IN_DOCKER="true"
fi

# Paths adjust based on Docker vs host execution
if [[ "$IN_DOCKER" == "true" ]]; then
    BUILD_DIR="${BUILD_DIR:-/llama.cpp/build}"
    RESULTS_DIR="${RESULTS_DIR:-/results}"
    STATE_FILE="${RESULTS_DIR}/.bench_state.json"
else
    BUILD_DIR="${LLAMA_DIR}/build"
    RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
    STATE_FILE="${SCRIPT_DIR}/.bench_state.json"
fi

# Ensure results directory exists
mkdir -p "$RESULTS_DIR" 2>/dev/null || true

# Dual output: human summary + structured JSON
emit_result() {
    local summary="$1"
    local json="$2"
    local next_steps="${3:-}"

    # Human-readable section
    echo ""
    echo "=== Result ==="
    echo "$summary"

    if [[ -n "$next_steps" ]]; then
        echo ""
        echo "=== Suggested Next Steps ==="
        echo "$next_steps"
    fi

    # Machine-parseable section
    echo ""
    echo "@@JSON@@"
    echo "$json"
    echo "@@JSON@@"
}

emit_error() {
    local error_msg="$1"
    local suggestion="$2"

    echo ""
    echo "=== Error ==="
    echo "$error_msg"
    echo ""
    echo "=== How to Fix ==="
    echo "$suggestion"
    echo ""
    echo "@@JSON@@"
    echo "{\"success\": false, \"error\": \"$error_msg\", \"suggestion\": \"$suggestion\"}"
    echo "@@JSON@@"
}

emit_progress() {
    local step="$1"
    local total="$2"
    local message="$3"
    echo "[${step}/${total}] $message"
}

save_state() {
    local key="$1"
    local value="$2"

    # Create or update state file
    if [[ -f "$STATE_FILE" ]]; then
        local tmp=$(mktemp)
        jq --arg k "$key" --arg v "$value" '.[$k] = $v' "$STATE_FILE" > "$tmp" 2>/dev/null || echo "{\"$key\": \"$value\"}" > "$tmp"
        mv "$tmp" "$STATE_FILE"
    else
        echo "{\"$key\": \"$value\"}" > "$STATE_FILE"
    fi
}

get_state() {
    local key="$1"
    local default="${2:-}"

    if [[ -f "$STATE_FILE" ]]; then
        jq -r --arg k "$key" '.[$k] // empty' "$STATE_FILE" 2>/dev/null || echo "$default"
    else
        echo "$default"
    fi
}

# ============================================================
# COMMANDS
# ============================================================

cmd_status() {
    echo "Checking system status..."

    local issues=()
    local model_ok="false"
    local mmproj_ok="false"
    local llama_bench_ok="false"
    local llama_cli_ok="false"
    local nsys_ok="false"
    local model_size=""
    local model_type=""

    # Check model
    if [[ -z "${MODEL_PATH:-}" ]]; then
        issues+=("MODEL_PATH environment variable not set")
    elif [[ ! -f "$MODEL_PATH" ]]; then
        issues+=("Model file not found: $MODEL_PATH")
    else
        model_ok="true"
        model_size=$(ls -lh "$MODEL_PATH" 2>/dev/null | awk '{print $5}' || echo "unknown")
        # Try to detect model type from filename
        if [[ "$MODEL_PATH" == *"moe"* ]] || [[ "$MODEL_PATH" == *"mixtral"* ]] || [[ "$MODEL_PATH" == *"qwen"*"vl"* ]]; then
            model_type="MoE"
        else
            model_type="Dense"
        fi
    fi

    # Check mmproj (for VL models)
    if [[ -n "${MMPROJ_PATH:-}" ]]; then
        if [[ -f "$MMPROJ_PATH" ]]; then
            mmproj_ok="true"
        else
            issues+=("MMPROJ file not found: $MMPROJ_PATH")
        fi
    fi

    # Check binaries
    if [[ -x "${BUILD_DIR}/bin/llama-bench" ]]; then
        llama_bench_ok="true"
    else
        issues+=("llama-bench not found. Run: cmake --build build --target llama-bench")
    fi

    if [[ -x "${BUILD_DIR}/bin/llama-cli" ]]; then
        llama_cli_ok="true"
    else
        issues+=("llama-cli not found. Run: cmake --build build --target llama-cli")
    fi

    # Check nsys
    if command -v nsys &>/dev/null; then
        nsys_ok="true"
    fi

    # Get GPU info
    local gpu_name="not detected"
    local gpu_memory="unknown"
    local gpu_driver="unknown"
    if command -v nvidia-smi &>/dev/null; then
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "error")
        gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "error")
        gpu_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "error")
    fi

    # Determine readiness and next steps
    local ready="false"
    local next_steps=""

    if [[ ${#issues[@]} -eq 0 && "$model_ok" == "true" && "$llama_bench_ok" == "true" ]]; then
        ready="true"
        if [[ "$model_type" == "MoE" ]]; then
            next_steps="Ready! For MoE models, recommended first test:
  ./claude-driver.sh bench_sweep
This will test baseline, flash_attn, and moe_cpu configurations."
        else
            next_steps="Ready! Recommended first test:
  ./claude-driver.sh bench_quick"
        fi
    else
        next_steps="Fix these issues first:
$(printf '  - %s\n' "${issues[@]}")"
    fi

    # Build summary
    local docker_status=""
    if [[ "$IN_DOCKER" == "true" ]]; then
        docker_status="Environment: Docker container
"
    fi

    local summary="System Status: $([ "$ready" == "true" ] && echo "READY" || echo "NOT READY")
${docker_status}
Model: ${MODEL_PATH:-not set} ($model_size, $model_type)
GPU: $gpu_name ($gpu_memory)
Driver: $gpu_driver

Components:
  llama-bench: $([ "$llama_bench_ok" == "true" ] && echo "OK" || echo "MISSING")
  llama-cli: $([ "$llama_cli_ok" == "true" ] && echo "OK" || echo "MISSING")
  nsys: $([ "$nsys_ok" == "true" ] && echo "OK" || echo "not found (optional)")
  mmproj: $([ "$mmproj_ok" == "true" ] && echo "OK" || echo "not configured")"

    local json="{
  \"success\": true,
  \"ready\": $ready,
  \"in_docker\": $IN_DOCKER,
  \"model\": {
    \"path\": \"${MODEL_PATH:-}\",
    \"size\": \"$model_size\",
    \"type\": \"$model_type\",
    \"exists\": $model_ok
  },
  \"gpu\": {
    \"name\": \"$gpu_name\",
    \"memory\": \"$gpu_memory\",
    \"driver\": \"$gpu_driver\"
  },
  \"components\": {
    \"llama_bench\": $llama_bench_ok,
    \"llama_cli\": $llama_cli_ok,
    \"nsys\": $nsys_ok,
    \"mmproj\": $mmproj_ok
  },
  \"issues\": $(printf '%s\n' "${issues[@]:-}" | jq -R . | jq -s . 2>/dev/null || echo "[]")
}"

    emit_result "$summary" "$json" "$next_steps"
}

cmd_bench_quick() {
    # Quick single benchmark - good for iteration
    local flags="${1:--ngl 99 -fa 1}"
    local pp="${2:-512}"
    local tg="${3:-128}"

    if [[ -z "${MODEL_PATH:-}" || ! -f "${MODEL_PATH:-}" ]]; then
        emit_error "Model not found" "Set MODEL_PATH to your model file: export MODEL_PATH=/path/to/model.gguf"
        return 1
    fi

    echo "Running quick benchmark: pp=$pp, tg=$tg, flags=$flags"
    echo ""

    local output
    local start_time=$(date +%s)

    if ! output=$("${BUILD_DIR}/bin/llama-bench" \
        -m "$MODEL_PATH" \
        -p "$pp" \
        -n "$tg" \
        -r 3 \
        $flags \
        -o json 2>&1); then
        emit_error "Benchmark failed" "Check if model is valid and GPU has enough memory. Try: nvidia-smi"
        return 1
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Parse results
    local pp_ts=$(echo "$output" | jq -r '.[0].avg_ts // 0' 2>/dev/null || echo "0")
    local tg_ts=$(echo "$output" | jq -r '.[1].avg_ts // 0' 2>/dev/null || echo "0")

    # Save to state for comparison
    save_state "last_pp_ts" "$pp_ts"
    save_state "last_tg_ts" "$tg_ts"
    save_state "last_flags" "$flags"

    local summary="Benchmark completed in ${duration}s

Results:
  Prompt processing: ${pp_ts} tokens/sec (${pp} tokens)
  Token generation:  ${tg_ts} tokens/sec (${tg} tokens)

Configuration: $flags"

    local next_steps="To compare with different settings:
  ./claude-driver.sh bench_quick \"-ngl 99 -fa 1 -cmoe\"

To run a full configuration sweep:
  ./claude-driver.sh bench_sweep

To profile and find bottlenecks:
  ./claude-driver.sh profile"

    local json="{
  \"success\": true,
  \"command\": \"bench_quick\",
  \"duration_sec\": $duration,
  \"config\": {
    \"prompt_tokens\": $pp,
    \"gen_tokens\": $tg,
    \"flags\": \"$flags\"
  },
  \"results\": {
    \"prompt_tokens_per_sec\": $pp_ts,
    \"gen_tokens_per_sec\": $tg_ts
  },
  \"raw\": $output
}"

    emit_result "$summary" "$json" "$next_steps"
}

cmd_bench_sweep() {
    # Sweep through key configurations
    local pp="${1:-512}"
    local tg="${2:-128}"

    if [[ -z "${MODEL_PATH:-}" || ! -f "${MODEL_PATH:-}" ]]; then
        emit_error "Model not found" "Set MODEL_PATH: export MODEL_PATH=/path/to/model.gguf"
        return 1
    fi

    echo "Running configuration sweep (pp=$pp, tg=$tg)..."
    echo ""

    local configs=(
        "baseline:-ngl 99"
        "flash_attn:-ngl 99 -fa 1"
        "moe_cpu:-ngl 99 -fa 1 -cmoe"
        "kv_q8:-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0"
    )

    local results="["
    local summary="Configuration Sweep Results (pp=$pp, tg=$tg)\n"
    summary+="================================================\n"

    local best_pp_config=""
    local best_pp_ts=0
    local best_tg_config=""
    local best_tg_ts=0
    local total=${#configs[@]}
    local i=0

    for config in "${configs[@]}"; do
        i=$((i + 1))
        local name="${config%%:*}"
        local flags="${config#*:}"

        emit_progress $i $total "Testing $name..."

        local output
        if output=$("${BUILD_DIR}/bin/llama-bench" \
            -m "$MODEL_PATH" \
            -p "$pp" -n "$tg" \
            -r 3 $flags \
            -o json 2>&1); then

            local pp_ts=$(echo "$output" | jq -r '.[0].avg_ts // 0' 2>/dev/null || echo "0")
            local tg_ts=$(echo "$output" | jq -r '.[1].avg_ts // 0' 2>/dev/null || echo "0")

            summary+="  $name: pp=${pp_ts} t/s, tg=${tg_ts} t/s\n"

            # Track best
            if (( $(echo "$pp_ts > $best_pp_ts" | bc -l 2>/dev/null || echo 0) )); then
                best_pp_ts=$pp_ts
                best_pp_config=$name
            fi
            if (( $(echo "$tg_ts > $best_tg_ts" | bc -l 2>/dev/null || echo 0) )); then
                best_tg_ts=$tg_ts
                best_tg_config=$name
            fi

            [[ $i -gt 1 ]] && results+=","
            results+="{\"config\": \"$name\", \"flags\": \"$flags\", \"pp_ts\": $pp_ts, \"tg_ts\": $tg_ts}"
        else
            summary+="  $name: FAILED\n"
            [[ $i -gt 1 ]] && results+=","
            results+="{\"config\": \"$name\", \"flags\": \"$flags\", \"error\": true}"
        fi
    done

    results+="]"

    summary+="\nBest for prompt processing: $best_pp_config (${best_pp_ts} t/s)"
    summary+="\nBest for token generation: $best_tg_config (${best_tg_ts} t/s)"

    # Save best config
    save_state "best_config" "$best_tg_config"
    save_state "best_tg_ts" "$best_tg_ts"

    local next_steps="Best configuration: $best_tg_config

To profile this configuration and find remaining bottlenecks:
  ./claude-driver.sh profile \"$best_tg_config\"

To test with different prompt/generation lengths:
  ./claude-driver.sh bench_sweep 1024 256"

    local json="{
  \"success\": true,
  \"command\": \"bench_sweep\",
  \"params\": {\"pp\": $pp, \"tg\": $tg},
  \"results\": $results,
  \"best\": {
    \"for_prefill\": \"$best_pp_config\",
    \"prefill_ts\": $best_pp_ts,
    \"for_decode\": \"$best_tg_config\",
    \"decode_ts\": $best_tg_ts
  }
}"

    emit_result "$(echo -e "$summary")" "$json" "$next_steps"
}

cmd_profile() {
    # Profile with Nsight Systems
    local config="${1:-flash_attn}"
    local pp="${2:-256}"
    local tg="${3:-32}"

    if ! command -v nsys &>/dev/null; then
        emit_error "nsys not found" "Install NVIDIA Nsight Systems or run on a system with CUDA toolkit"
        return 1
    fi

    if [[ -z "${MODEL_PATH:-}" || ! -f "${MODEL_PATH:-}" ]]; then
        emit_error "Model not found" "Set MODEL_PATH: export MODEL_PATH=/path/to/model.gguf"
        return 1
    fi

    # Map config name to flags
    local flags=""
    case "$config" in
        baseline)   flags="-ngl 99" ;;
        flash_attn) flags="-ngl 99 -fa 1" ;;
        moe_cpu)    flags="-ngl 99 -fa 1 -cmoe" ;;
        kv_q8)      flags="-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0" ;;
        *)          flags="$config" ;;  # Allow passing raw flags
    esac

    local output_dir="${RESULTS_DIR}/profile_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$output_dir"

    echo "Profiling with config: $config"
    echo "Output directory: $output_dir"
    echo ""

    emit_progress 1 3 "Running nsys profile..."

    local nsys_file="${output_dir}/profile"
    nsys profile \
        --trace=cuda,nvtx \
        --cuda-memory-usage=true \
        --output="$nsys_file" \
        --force-overwrite=true \
        "${BUILD_DIR}/bin/llama-bench" \
        -m "$MODEL_PATH" \
        -p "$pp" -n "$tg" \
        -r 1 $flags \
        2>&1 | grep -v "^Generating" || true

    emit_progress 2 3 "Extracting kernel statistics..."

    nsys stats "${nsys_file}.nsys-rep" \
        --report cuda_gpu_kern_sum \
        --format json > "${output_dir}/kernels.json" 2>/dev/null || echo "[]" > "${output_dir}/kernels.json"

    emit_progress 3 3 "Analyzing results..."

    # Analyze top kernels
    local top_kernels=$(jq -r 'sort_by(-.["Time (%)"]) | .[0:10] | .[] | "\(.Name): \(.["Time (%)"])%"' "${output_dir}/kernels.json" 2>/dev/null || echo "Analysis failed")

    # Detect patterns
    local has_moe=$(grep -qi "moe\|topk" "${output_dir}/kernels.json" && echo "true" || echo "false")
    local has_flash=$(grep -qi "flash" "${output_dir}/kernels.json" && echo "true" || echo "false")

    local summary="Profile completed: $output_dir

Top kernels by time:
$top_kernels

Detected patterns:
  MoE kernels present: $has_moe
  Flash attention present: $has_flash"

    local recommendations=""
    if [[ "$has_moe" == "true" && "$config" != "moe_cpu" ]]; then
        recommendations+="- MoE kernels detected. Try: ./claude-driver.sh bench_quick \"-ngl 99 -fa 1 -cmoe\"\n"
    fi
    if [[ "$has_flash" == "false" && "$config" != *"-fa"* ]]; then
        recommendations+="- Flash attention not detected. Try: ./claude-driver.sh bench_quick \"-ngl 99 -fa 1\"\n"
    fi

    local next_steps="Profile saved to: $output_dir

To view detailed profile in Nsight Systems GUI:
  nsys-ui ${nsys_file}.nsys-rep

$([ -n "$recommendations" ] && echo -e "Recommendations:\n$recommendations")"

    local json="{
  \"success\": true,
  \"command\": \"profile\",
  \"config\": \"$config\",
  \"output_dir\": \"$output_dir\",
  \"nsys_file\": \"${nsys_file}.nsys-rep\",
  \"analysis\": {
    \"moe_detected\": $has_moe,
    \"flash_attn_detected\": $has_flash
  },
  \"kernels_file\": \"${output_dir}/kernels.json\"
}"

    emit_result "$summary" "$json" "$next_steps"
}

cmd_compare() {
    # Compare current results with previous
    local current_pp=$(get_state "last_pp_ts" "0")
    local current_tg=$(get_state "last_tg_ts" "0")
    local current_flags=$(get_state "last_flags" "unknown")
    local best_tg=$(get_state "best_tg_ts" "0")
    local best_config=$(get_state "best_config" "unknown")

    if [[ "$current_tg" == "0" ]]; then
        emit_error "No benchmark results to compare" "Run a benchmark first: ./claude-driver.sh bench_quick"
        return 1
    fi

    local improvement="N/A"
    if [[ "$best_tg" != "0" ]]; then
        improvement=$(echo "scale=1; ($current_tg - $best_tg) / $best_tg * 100" | bc 2>/dev/null || echo "N/A")
    fi

    local summary="Comparison with best known result:

Current run ($current_flags):
  Prompt: $current_pp t/s
  Decode: $current_tg t/s

Best recorded ($best_config):
  Decode: $best_tg t/s

Difference: ${improvement}%"

    local json="{
  \"success\": true,
  \"current\": {
    \"flags\": \"$current_flags\",
    \"pp_ts\": $current_pp,
    \"tg_ts\": $current_tg
  },
  \"best\": {
    \"config\": \"$best_config\",
    \"tg_ts\": $best_tg
  },
  \"improvement_pct\": \"$improvement\"
}"

    emit_result "$summary" "$json"
}

cmd_gpu() {
    # Monitor GPU status
    if ! command -v nvidia-smi &>/dev/null; then
        emit_error "nvidia-smi not found" "This command requires NVIDIA GPU and drivers"
        return 1
    fi

    local stats=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || echo "0,0,0,0,0,0")

    IFS=',' read -r gpu_pct mem_pct mem_used mem_total temp power <<< "$stats"

    local summary="GPU Status:
  Utilization: ${gpu_pct}%
  Memory: ${mem_used}MB / ${mem_total}MB (${mem_pct}%)
  Temperature: ${temp}°C
  Power: ${power}W"

    local json="{
  \"success\": true,
  \"gpu_util_pct\": ${gpu_pct:-0},
  \"mem_util_pct\": ${mem_pct:-0},
  \"mem_used_mb\": ${mem_used:-0},
  \"mem_total_mb\": ${mem_total:-0},
  \"temp_c\": ${temp:-0},
  \"power_w\": ${power:-0}
}"

    emit_result "$summary" "$json"
}

cmd_help() {
    local docker_info=""
    if [[ "$IN_DOCKER" == "true" ]]; then
        docker_info="
Running in: Docker container"
    else
        docker_info="
Docker Usage:
  # Build image
  docker build -t vl-bench -f tools/vl-bench/Dockerfile .

  # Run with docker-compose
  MODEL_DIR=/path/to/models docker compose -f tools/vl-bench/docker-compose.yml run --rm bench status

  # Or run directly
  docker run --gpus all -v /path/to/models:/models -e MODEL_PATH=/models/model.gguf vl-bench status"
    fi

    local summary="VL-Bench: Vision-Language Benchmark Suite for llama.cpp

Commands:
  status      - Check system readiness and prerequisites
  bench_quick - Run single benchmark (fast iteration)
  bench_sweep - Test multiple configurations
  profile     - Profile with Nsight Systems
  compare     - Compare with previous results
  gpu         - Show GPU status
  help        - Show this help

Quick Start (Host):
  1. export MODEL_PATH=/path/to/model.gguf
  2. ./claude-driver.sh status
  3. ./claude-driver.sh bench_sweep
$docker_info

Environment:
  MODEL_PATH  - Path to GGUF model (required)
  MMPROJ_PATH - Path to vision projector (for VL models)
  RESULTS_DIR - Output directory (default: ./results)"

    local json="{
  \"success\": true,
  \"in_docker\": $IN_DOCKER,
  \"commands\": [\"status\", \"bench_quick\", \"bench_sweep\", \"profile\", \"compare\", \"gpu\", \"help\"]
}"

    emit_result "$summary" "$json"
}

# ============================================================
# MAIN
# ============================================================

main() {
    local cmd="${1:-help}"
    shift || true

    case "$cmd" in
        status)      cmd_status ;;
        bench_quick) cmd_bench_quick "$@" ;;
        bench_sweep) cmd_bench_sweep "$@" ;;
        profile)     cmd_profile "$@" ;;
        compare)     cmd_compare ;;
        gpu)         cmd_gpu ;;
        help|--help|-h) cmd_help ;;
        *)
            emit_error "Unknown command: $cmd" "Run ./claude-driver.sh help for available commands"
            exit 1
            ;;
    esac
}

main "$@"
