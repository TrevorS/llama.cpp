#!/bin/bash
# 01_baseline_benchmark.sh - Establish baseline performance metrics
# Run this first to understand current performance before optimization

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/profile_config.sh"

log_info "Starting baseline benchmark for $MODEL_NAME"
log_info "Results will be saved to: $BASELINE_DIR"

# =============================================================================
# BASELINE: BASIC THROUGHPUT TEST
# =============================================================================
run_basic_benchmark() {
    log_info "Running basic throughput benchmark..."

    local output_file="$BASELINE_DIR/basic_throughput_${RUN_ID}.json"

    "$LLAMA_BENCH" \
        -m "$MODEL_PATH" \
        -fa 1 \
        -ngl "$NUM_GPU_LAYERS" \
        --no-mmap \
        -p 2048 \
        -n 32 \
        -r "$BENCH_REPS" \
        -o json \
        > "$output_file" 2>&1

    log_info "Basic benchmark saved to: $output_file"

    # Also save human-readable markdown
    "$LLAMA_BENCH" \
        -m "$MODEL_PATH" \
        -fa 1 \
        -ngl "$NUM_GPU_LAYERS" \
        --no-mmap \
        -p 2048 \
        -n 32 \
        -r "$BENCH_REPS" \
        -o md \
        > "$BASELINE_DIR/basic_throughput_${RUN_ID}.md" 2>&1
}

# =============================================================================
# BASELINE: VARYING PROMPT LENGTH (prefill performance)
# =============================================================================
run_prefill_benchmark() {
    log_info "Running prefill benchmark with varying prompt lengths..."

    local output_file="$BASELINE_DIR/prefill_scaling_${RUN_ID}.json"

    # Build prompt length string for llama-bench
    local pp_values=""
    IFS=',' read -ra PP_ARRAY <<< "$PROMPT_LENGTHS"
    for pp in "${PP_ARRAY[@]}"; do
        pp_values="$pp_values -p $pp"
    done

    "$LLAMA_BENCH" \
        -m "$MODEL_PATH" \
        -fa 1 \
        -ngl "$NUM_GPU_LAYERS" \
        --no-mmap \
        $pp_values \
        -n 0 \
        -r "$BENCH_REPS" \
        -o json \
        > "$output_file" 2>&1

    log_info "Prefill benchmark saved to: $output_file"
}

# =============================================================================
# BASELINE: VARYING CONTEXT DEPTH (attention scaling)
# =============================================================================
run_context_depth_benchmark() {
    log_info "Running context depth benchmark..."

    local output_file="$BASELINE_DIR/context_depth_${RUN_ID}.json"

    # Build context depth string
    local depth_values=""
    IFS=',' read -ra DEPTH_ARRAY <<< "$CONTEXT_DEPTHS"
    for d in "${DEPTH_ARRAY[@]}"; do
        depth_values="$depth_values -d $d"
    done

    "$LLAMA_BENCH" \
        -m "$MODEL_PATH" \
        -fa 1 \
        -ngl "$NUM_GPU_LAYERS" \
        --no-mmap \
        -p 2048 \
        -n 32 \
        $depth_values \
        -r "$BENCH_REPS" \
        -o json \
        > "$output_file" 2>&1

    log_info "Context depth benchmark saved to: $output_file"
}

# =============================================================================
# BASELINE: TOKEN GENERATION (decode performance)
# =============================================================================
run_decode_benchmark() {
    log_info "Running decode benchmark with varying generation lengths..."

    local output_file="$BASELINE_DIR/decode_scaling_${RUN_ID}.json"

    # Build generation length string
    local n_values=""
    IFS=',' read -ra N_ARRAY <<< "$GEN_LENGTHS"
    for n in "${N_ARRAY[@]}"; do
        n_values="$n_values -n $n"
    done

    "$LLAMA_BENCH" \
        -m "$MODEL_PATH" \
        -fa 1 \
        -ngl "$NUM_GPU_LAYERS" \
        --no-mmap \
        -p 128 \
        $n_values \
        -r "$BENCH_REPS" \
        -o json \
        > "$output_file" 2>&1

    log_info "Decode benchmark saved to: $output_file"
}

# =============================================================================
# BASELINE: COMPREHENSIVE TEST (pp + tg combined)
# =============================================================================
run_comprehensive_benchmark() {
    log_info "Running comprehensive benchmark..."

    local output_file="$BASELINE_DIR/comprehensive_${RUN_ID}.json"

    "$LLAMA_BENCH" \
        -m "$MODEL_PATH" \
        -fa 1 \
        -ngl "$NUM_GPU_LAYERS" \
        --no-mmap \
        -t pg \
        -p 128,512,2048 \
        -n 32,128 \
        -d 0,8192,32768 \
        -ub 2048 \
        -r "$BENCH_REPS" \
        -o json \
        > "$output_file" 2>&1

    log_info "Comprehensive benchmark saved to: $output_file"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================
main() {
    log_info "=========================================="
    log_info "BASELINE BENCHMARK SUITE"
    log_info "Model: $MODEL_NAME"
    log_info "Run ID: $RUN_ID"
    log_info "=========================================="

    if [[ ! -f "$MODEL_PATH" ]]; then
        log_error "Model file not found: $MODEL_PATH"
        log_error "Please set MODEL_PATH environment variable"
        exit 1
    fi

    # Run all baseline benchmarks
    run_basic_benchmark
    run_prefill_benchmark
    run_context_depth_benchmark
    run_decode_benchmark
    run_comprehensive_benchmark

    log_info "=========================================="
    log_info "BASELINE BENCHMARKS COMPLETE"
    log_info "Results in: $BASELINE_DIR"
    log_info "=========================================="

    # Generate summary
    generate_summary
}

generate_summary() {
    local summary_file="$BASELINE_DIR/summary_${RUN_ID}.txt"

    {
        echo "BASELINE PERFORMANCE SUMMARY"
        echo "============================"
        echo "Model: $MODEL_NAME"
        echo "Run ID: $RUN_ID"
        echo "Timestamp: $(date)"
        echo ""
        echo "Key Metrics (from basic throughput test):"
        echo "------------------------------------------"

        if [[ -f "$BASELINE_DIR/basic_throughput_${RUN_ID}.json" ]]; then
            # Extract key metrics using jq if available, otherwise use grep/sed
            if command -v jq &> /dev/null; then
                jq -r '.[] | "  \(.test_type): \(.avg_ts) t/s (±\(.stddev_ts))"' \
                    "$BASELINE_DIR/basic_throughput_${RUN_ID}.json" 2>/dev/null || \
                cat "$BASELINE_DIR/basic_throughput_${RUN_ID}.json"
            else
                cat "$BASELINE_DIR/basic_throughput_${RUN_ID}.json"
            fi
        fi

        echo ""
        echo "Files generated:"
        ls -la "$BASELINE_DIR/"*"${RUN_ID}"* 2>/dev/null || echo "  (none found)"

    } > "$summary_file"

    log_info "Summary saved to: $summary_file"
    cat "$summary_file"
}

main "$@"
