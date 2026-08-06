#!/bin/bash
# run-bench.sh - Vision-Language Benchmark Suite for llama.cpp
# Designed to be driven by Claude or run standalone
#
# Usage:
#   ./run-bench.sh [command] [options]
#
# Commands:
#   full          Run full benchmark suite
#   quick         Run quick smoke test
#   profile       Run with Nsight Systems profiling
#   single        Run single scenario/config combination
#   list          List available scenarios and configurations
#   analyze       Analyze results from previous run
#   compare       Compare two result sets

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${SCRIPT_DIR}/../.."
BUILD_DIR="${LLAMA_DIR}/build"
CONFIG_FILE="${SCRIPT_DIR}/config.json"
RESULTS_DIR="${SCRIPT_DIR}/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default values (can be overridden by environment)
MODEL_PATH="${MODEL_PATH:-}"
MMPROJ_PATH="${MMPROJ_PATH:-}"
REPETITIONS="${REPETITIONS:-3}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# JSON output helper - outputs structured data for Claude to parse
json_output() {
    local type="$1"
    local data="$2"
    echo "@@JSON_START@@"
    echo "{\"type\": \"$type\", \"timestamp\": \"$(date -Iseconds)\", \"data\": $data}"
    echo "@@JSON_END@@"
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    local errors=0

    # Check for llama-bench
    if [[ ! -x "${BUILD_DIR}/bin/llama-bench" ]]; then
        log_error "llama-bench not found at ${BUILD_DIR}/bin/llama-bench"
        errors=$((errors + 1))
    fi

    # Check for llama-cli
    if [[ ! -x "${BUILD_DIR}/bin/llama-cli" ]]; then
        log_error "llama-cli not found at ${BUILD_DIR}/bin/llama-cli"
        errors=$((errors + 1))
    fi

    # Check for model
    if [[ -z "$MODEL_PATH" ]]; then
        log_warn "MODEL_PATH not set, will need to be provided"
    elif [[ ! -f "$MODEL_PATH" ]]; then
        log_error "Model not found: $MODEL_PATH"
        errors=$((errors + 1))
    fi

    # Check for jq (for JSON processing)
    if ! command -v jq &> /dev/null; then
        log_error "jq is required but not installed"
        errors=$((errors + 1))
    fi

    # Check for nsys (optional)
    if ! command -v nsys &> /dev/null; then
        log_warn "nsys not found - Nsight Systems profiling will be disabled"
    fi

    if [[ $errors -gt 0 ]]; then
        log_error "Prerequisites check failed with $errors errors"
        return 1
    fi

    log_success "Prerequisites check passed"
    return 0
}

get_system_info() {
    log_info "Gathering system information..."

    local gpu_info=""
    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "N/A")
    fi

    local cpu_info=$(lscpu 2>/dev/null | grep "Model name" | cut -d: -f2 | xargs || echo "N/A")
    local mem_info=$(free -h 2>/dev/null | grep Mem | awk '{print $2}' || echo "N/A")
    local cuda_version=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | tr -d ',' || echo "N/A")

    json_output "system_info" "{
        \"gpu\": \"$gpu_info\",
        \"cpu\": \"$cpu_info\",
        \"memory\": \"$mem_info\",
        \"cuda_version\": \"$cuda_version\",
        \"hostname\": \"$(hostname)\",
        \"kernel\": \"$(uname -r)\"
    }"
}

run_llama_bench() {
    local config_flags="$1"
    local prompt_tokens="$2"
    local gen_tokens="$3"
    local env_vars="${4:-}"
    local output_file="$5"

    local cmd="${BUILD_DIR}/bin/llama-bench"
    cmd+=" -m $MODEL_PATH"
    cmd+=" -p $prompt_tokens"
    cmd+=" -n $gen_tokens"
    cmd+=" -r $REPETITIONS"
    cmd+=" $config_flags"
    cmd+=" -o json"

    log_info "Running: $cmd"

    if [[ -n "$env_vars" ]]; then
        env $env_vars $cmd > "$output_file" 2>&1
    else
        $cmd > "$output_file" 2>&1
    fi

    return $?
}

run_vl_bench() {
    local config_flags="$1"
    local image_path="$2"
    local prompt="$3"
    local gen_tokens="$4"
    local env_vars="${5:-}"
    local output_file="$6"

    local cmd="${BUILD_DIR}/bin/llama-cli"
    cmd+=" -m $MODEL_PATH"
    cmd+=" --mmproj $MMPROJ_PATH"
    cmd+=" --image $image_path"
    cmd+=" -p \"$prompt\""
    cmd+=" -n $gen_tokens"
    cmd+=" $config_flags"
    cmd+=" --no-display-prompt"

    log_info "Running VL benchmark..."

    local start_time=$(date +%s.%N)

    if [[ -n "$env_vars" ]]; then
        env $env_vars timeout 300 bash -c "$cmd" > "$output_file" 2>&1 || true
    else
        timeout 300 bash -c "$cmd" > "$output_file" 2>&1 || true
    fi

    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)

    echo "$duration"
}

run_single_test() {
    local scenario_id="$1"
    local config_id="$2"
    local run_dir="$3"

    mkdir -p "$run_dir"

    # Parse scenario from config
    local scenario=$(jq -r ".scenarios[] | select(.id == \"$scenario_id\")" "$CONFIG_FILE")
    local config=$(jq -r ".configurations[] | select(.id == \"$config_id\")" "$CONFIG_FILE")

    if [[ -z "$scenario" || "$scenario" == "null" ]]; then
        log_error "Scenario not found: $scenario_id"
        return 1
    fi

    if [[ -z "$config" || "$config" == "null" ]]; then
        log_error "Configuration not found: $config_id"
        return 1
    fi

    local scenario_type=$(echo "$scenario" | jq -r '.type')
    local config_flags=$(echo "$config" | jq -r '.flags')
    local env_vars=$(echo "$config" | jq -r '.env // empty')
    local description=$(echo "$scenario" | jq -r '.description')

    log_info "Running: $scenario_id with $config_id"
    log_info "Description: $description"

    local result_file="${run_dir}/${scenario_id}_${config_id}.json"
    local status="success"
    local metrics="{}"

    if [[ "$scenario_type" == "text" ]]; then
        local prompt_tokens=$(echo "$scenario" | jq -r '.prompt_tokens')
        local gen_tokens=$(echo "$scenario" | jq -r '.gen_tokens')

        if run_llama_bench "$config_flags" "$prompt_tokens" "$gen_tokens" "$env_vars" "$result_file"; then
            # Parse llama-bench JSON output
            if [[ -f "$result_file" ]]; then
                metrics=$(jq '.[0] | {
                    prompt_tokens: .n_prompt,
                    gen_tokens: .n_gen,
                    avg_prompt_ts: .avg_ts,
                    stddev_prompt_ts: .stddev_ts
                }' "$result_file" 2>/dev/null || echo '{"error": "parse_failed"}')
            fi
        else
            status="failed"
        fi

    elif [[ "$scenario_type" == "vision" ]]; then
        local image_key=$(echo "$scenario" | jq -r '.image')
        local image_path=$(jq -r ".test_images.$image_key" "$CONFIG_FILE")
        local prompt=$(echo "$scenario" | jq -r '.prompt')
        local gen_tokens=$(echo "$scenario" | jq -r '.gen_tokens')

        # Check if image exists
        if [[ ! -f "${SCRIPT_DIR}/$image_path" ]]; then
            log_warn "Test image not found: $image_path, using placeholder"
            image_path="/dev/null"
            status="skipped_no_image"
        else
            image_path="${SCRIPT_DIR}/$image_path"
        fi

        local duration=$(run_vl_bench "$config_flags" "$image_path" "$prompt" "$gen_tokens" "$env_vars" "$result_file")

        metrics=$(echo "{
            \"duration_seconds\": $duration,
            \"gen_tokens\": $gen_tokens,
            \"tokens_per_second\": $(echo "scale=2; $gen_tokens / $duration" | bc 2>/dev/null || echo "0")
        }")
    fi

    # Output structured result
    json_output "test_result" "{
        \"scenario_id\": \"$scenario_id\",
        \"config_id\": \"$config_id\",
        \"status\": \"$status\",
        \"metrics\": $metrics,
        \"result_file\": \"$result_file\"
    }"

    return 0
}

run_quick() {
    log_info "Running quick smoke test..."

    local run_dir="${RESULTS_DIR}/quick_${TIMESTAMP}"
    mkdir -p "$run_dir"

    json_output "run_start" "{\"type\": \"quick\", \"output_dir\": \"$run_dir\"}"

    get_system_info

    # Run minimal set of tests
    run_single_test "text_only_short" "baseline" "$run_dir"
    run_single_test "text_only_short" "flash_attn" "$run_dir"

    if [[ -n "$MMPROJ_PATH" && -f "$MMPROJ_PATH" ]]; then
        run_single_test "vl_small_image" "flash_attn" "$run_dir" || true
    fi

    json_output "run_complete" "{\"output_dir\": \"$run_dir\", \"status\": \"complete\"}"
}

run_full() {
    log_info "Running full benchmark suite..."

    local run_dir="${RESULTS_DIR}/full_${TIMESTAMP}"
    mkdir -p "$run_dir"

    json_output "run_start" "{\"type\": \"full\", \"output_dir\": \"$run_dir\"}"

    get_system_info

    # Get all scenarios and configurations
    local scenarios=$(jq -r '.scenarios[].id' "$CONFIG_FILE")
    local configs=$(jq -r '.configurations[].id' "$CONFIG_FILE")

    local total_tests=$(echo "$scenarios" | wc -l)
    total_tests=$((total_tests * $(echo "$configs" | wc -l)))
    local current_test=0

    for scenario in $scenarios; do
        for config in $configs; do
            current_test=$((current_test + 1))
            log_info "Test $current_test/$total_tests: $scenario + $config"

            run_single_test "$scenario" "$config" "$run_dir" || {
                log_warn "Test failed: $scenario + $config"
            }
        done
    done

    json_output "run_complete" "{\"output_dir\": \"$run_dir\", \"total_tests\": $total_tests, \"status\": \"complete\"}"

    # Generate summary
    generate_summary "$run_dir"
}

run_profile() {
    local scenario_id="${1:-text_only_medium}"
    local config_id="${2:-flash_attn}"

    log_info "Running Nsight Systems profile for $scenario_id with $config_id..."

    if ! command -v nsys &> /dev/null; then
        log_error "nsys not found - cannot run profiling"
        return 1
    fi

    local run_dir="${RESULTS_DIR}/profile_${TIMESTAMP}"
    mkdir -p "$run_dir"

    local scenario=$(jq -r ".scenarios[] | select(.id == \"$scenario_id\")" "$CONFIG_FILE")
    local config=$(jq -r ".configurations[] | select(.id == \"$config_id\")" "$CONFIG_FILE")
    local config_flags=$(echo "$config" | jq -r '.flags')
    local env_vars=$(echo "$config" | jq -r '.env // empty')

    local nsys_output="${run_dir}/nsys_${scenario_id}_${config_id}"

    local cmd=""
    local scenario_type=$(echo "$scenario" | jq -r '.type')

    if [[ "$scenario_type" == "text" ]]; then
        local prompt_tokens=$(echo "$scenario" | jq -r '.prompt_tokens')
        local gen_tokens=$(echo "$scenario" | jq -r '.gen_tokens')
        cmd="${BUILD_DIR}/bin/llama-bench -m $MODEL_PATH -p $prompt_tokens -n $gen_tokens -r 1 $config_flags"
    else
        local image_key=$(echo "$scenario" | jq -r '.image')
        local image_path="${SCRIPT_DIR}/$(jq -r ".test_images.$image_key" "$CONFIG_FILE")"
        local prompt=$(echo "$scenario" | jq -r '.prompt')
        local gen_tokens=$(echo "$scenario" | jq -r '.gen_tokens')
        cmd="${BUILD_DIR}/bin/llama-cli -m $MODEL_PATH --mmproj $MMPROJ_PATH --image $image_path -p \"$prompt\" -n $gen_tokens $config_flags"
    fi

    local nsys_cmd="nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true --output=$nsys_output --force-overwrite=true"

    log_info "Running: $nsys_cmd $cmd"

    if [[ -n "$env_vars" ]]; then
        env $env_vars $nsys_cmd bash -c "$cmd"
    else
        $nsys_cmd bash -c "$cmd"
    fi

    # Generate stats
    log_info "Generating kernel statistics..."
    nsys stats "${nsys_output}.nsys-rep" \
        --report cuda_gpu_kern_sum \
        --format csv > "${run_dir}/kernel_summary.csv" 2>/dev/null || true

    nsys stats "${nsys_output}.nsys-rep" \
        --report cuda_gpu_kern_sum \
        --format json > "${run_dir}/kernel_summary.json" 2>/dev/null || true

    json_output "profile_complete" "{
        \"output_dir\": \"$run_dir\",
        \"nsys_file\": \"${nsys_output}.nsys-rep\",
        \"kernel_summary\": \"${run_dir}/kernel_summary.json\"
    }"
}

generate_summary() {
    local run_dir="$1"

    log_info "Generating summary..."

    local summary_file="${run_dir}/summary.json"

    # Aggregate all results
    echo "{" > "$summary_file"
    echo "  \"timestamp\": \"$(date -Iseconds)\"," >> "$summary_file"
    echo "  \"results\": [" >> "$summary_file"

    local first=true
    for result_file in "${run_dir}"/*.json; do
        if [[ "$result_file" == *"summary.json" ]]; then
            continue
        fi

        if [[ "$first" == true ]]; then
            first=false
        else
            echo "," >> "$summary_file"
        fi

        local basename=$(basename "$result_file" .json)
        echo "    {\"test\": \"$basename\", \"file\": \"$result_file\"}" >> "$summary_file"
    done

    echo "  ]" >> "$summary_file"
    echo "}" >> "$summary_file"

    log_success "Summary written to $summary_file"
}

cmd_list() {
    log_info "Available scenarios:"
    jq -r '.scenarios[] | "  \(.id): \(.description)"' "$CONFIG_FILE"

    echo ""
    log_info "Available configurations:"
    jq -r '.configurations[] | "  \(.id): \(.description)"' "$CONFIG_FILE"
}

cmd_compare() {
    local result1="$1"
    local result2="$2"

    log_info "Comparing $result1 vs $result2..."

    # This would be expanded with actual comparison logic
    json_output "comparison" "{
        \"baseline\": \"$result1\",
        \"comparison\": \"$result2\",
        \"status\": \"not_implemented\"
    }"
}

show_help() {
    cat << EOF
Vision-Language Benchmark Suite for llama.cpp

Usage: $0 [command] [options]

Commands:
  quick                     Run quick smoke test (2-3 tests)
  full                      Run full benchmark suite
  profile [scenario] [cfg]  Run Nsight Systems profiling
  single <scenario> <cfg>   Run single scenario+config combination
  list                      List available scenarios and configurations
  compare <dir1> <dir2>     Compare two result sets
  help                      Show this help message

Environment Variables:
  MODEL_PATH      Path to the main model file (required)
  MMPROJ_PATH     Path to the mmproj file (required for VL tests)
  REPETITIONS     Number of repetitions per test (default: 3)
  WARMUP_RUNS     Number of warmup runs (default: 1)

Examples:
  # Quick test
  MODEL_PATH=./model.gguf ./run-bench.sh quick

  # Full suite with vision
  MODEL_PATH=./model.gguf MMPROJ_PATH=./mmproj.gguf ./run-bench.sh full

  # Profile specific scenario
  MODEL_PATH=./model.gguf ./run-bench.sh profile text_only_medium moe_cpu

  # Single test
  MODEL_PATH=./model.gguf ./run-bench.sh single text_only_short baseline

Output:
  Results are saved to: ${RESULTS_DIR}/
  JSON markers (@@JSON_START@@/@@JSON_END@@) enable machine parsing
EOF
}

# Main entry point
main() {
    local command="${1:-help}"
    shift || true

    case "$command" in
        quick)
            check_prerequisites && run_quick
            ;;
        full)
            check_prerequisites && run_full
            ;;
        profile)
            check_prerequisites && run_profile "$@"
            ;;
        single)
            if [[ $# -lt 2 ]]; then
                log_error "Usage: $0 single <scenario_id> <config_id>"
                exit 1
            fi
            check_prerequisites && run_single_test "$1" "$2" "${RESULTS_DIR}/single_${TIMESTAMP}"
            ;;
        list)
            cmd_list
            ;;
        compare)
            if [[ $# -lt 2 ]]; then
                log_error "Usage: $0 compare <result_dir1> <result_dir2>"
                exit 1
            fi
            cmd_compare "$1" "$2"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
