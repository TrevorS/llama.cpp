#!/bin/bash
# 05_optimization_workflow.sh - Iterative optimization workflow for Claude Code
# This script guides the optimization process and tracks changes

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/profile_config.sh"

# =============================================================================
# OPTIMIZATION TRACKING
# =============================================================================
OPTIMIZATION_LOG="$OPTIMIZATION_DIR/optimization_log.jsonl"

log_optimization() {
    local change_type="$1"
    local description="$2"
    local files_changed="$3"
    local run_id="$4"

    local entry=$(cat << EOF
{"timestamp": "$(date -Iseconds)", "run_id": "$run_id", "change_type": "$change_type", "description": "$description", "files_changed": "$files_changed"}
EOF
)
    echo "$entry" >> "$OPTIMIZATION_LOG"
}

# =============================================================================
# WORKFLOW COMMANDS
# =============================================================================

# Start a new optimization iteration
cmd_start() {
    local description="$1"

    if [[ -z "$description" ]]; then
        echo "Usage: $0 start <description>"
        echo "Example: $0 start 'Fuse SwiGLU with matmul'"
        exit 1
    fi

    export RUN_ID="opt_$(timestamp)"

    log_info "Starting optimization iteration: $RUN_ID"
    log_info "Description: $description"

    # Create iteration directory
    mkdir -p "$OPTIMIZATION_DIR/$RUN_ID"

    # Save description
    echo "$description" > "$OPTIMIZATION_DIR/$RUN_ID/description.txt"

    # Log start
    log_optimization "start" "$description" "" "$RUN_ID"

    echo ""
    echo "Optimization iteration started: $RUN_ID"
    echo "Now make your code changes, then run:"
    echo "  $0 benchmark $RUN_ID"
    echo ""
}

# Run benchmark for current optimization
cmd_benchmark() {
    local run_id="${1:-$RUN_ID}"

    if [[ -z "$run_id" ]]; then
        echo "Usage: $0 benchmark <run_id>"
        exit 1
    fi

    export RUN_ID="$run_id"

    log_info "Running benchmark for: $RUN_ID"

    # Run baseline benchmark
    "$SCRIPT_DIR/01_baseline_benchmark.sh"

    log_optimization "benchmark" "Benchmark completed" "" "$RUN_ID"

    echo ""
    echo "Benchmark complete for: $RUN_ID"
    echo "Results in: $BASELINE_DIR/*_${RUN_ID}.*"
    echo ""
}

# Run full profiling for current optimization
cmd_profile() {
    local run_id="${1:-$RUN_ID}"

    if [[ -z "$run_id" ]]; then
        echo "Usage: $0 profile <run_id>"
        exit 1
    fi

    export RUN_ID="$run_id"

    log_info "Running full profiling for: $RUN_ID"

    # Run nsight systems
    "$SCRIPT_DIR/02_nsight_systems_profile.sh" all

    # Run nsight compute (quick scan only for iterations)
    "$SCRIPT_DIR/03_nsight_compute_profile.sh" quick

    log_optimization "profile" "Profiling completed" "" "$RUN_ID"
}

# Compare with baseline
cmd_compare() {
    local baseline_id="$1"
    local optimized_id="$2"

    if [[ -z "$baseline_id" ]] || [[ -z "$optimized_id" ]]; then
        echo "Usage: $0 compare <baseline_run_id> <optimized_run_id>"
        exit 1
    fi

    "$SCRIPT_DIR/04_compare_runs.sh" "$baseline_id" "$optimized_id"
}

# Record the changes made
cmd_record() {
    local run_id="$1"
    local files="$2"

    if [[ -z "$run_id" ]] || [[ -z "$files" ]]; then
        echo "Usage: $0 record <run_id> <files_changed>"
        echo "Example: $0 record opt_20241218 'ggml/src/ggml-cuda/unary.cu'"
        exit 1
    fi

    log_optimization "changes" "Files modified" "$files" "$run_id"

    # Also save git diff if available
    if git rev-parse --git-dir > /dev/null 2>&1; then
        git diff > "$OPTIMIZATION_DIR/$run_id/changes.diff" 2>/dev/null || true
        git diff --stat > "$OPTIMIZATION_DIR/$run_id/changes_stat.txt" 2>/dev/null || true
    fi

    echo "Changes recorded for: $run_id"
}

# Show optimization history
cmd_history() {
    log_info "Optimization History"
    echo "===================="

    if [[ -f "$OPTIMIZATION_LOG" ]]; then
        cat "$OPTIMIZATION_LOG" | while read line; do
            echo "$line" | jq -r '"\(.timestamp) [\(.run_id)] \(.change_type): \(.description)"' 2>/dev/null || echo "$line"
        done
    else
        echo "(no optimization history)"
    fi
}

# Generate summary report
cmd_summary() {
    local output_file="$OPTIMIZATION_DIR/summary_$(timestamp).md"

    {
        echo "# Optimization Summary"
        echo ""
        echo "Generated: $(date)"
        echo "Model: $MODEL_NAME"
        echo ""

        echo "## Optimization Iterations"
        echo ""

        for dir in "$OPTIMIZATION_DIR"/opt_*/; do
            if [[ -d "$dir" ]]; then
                local run_id=$(basename "$dir")
                echo "### $run_id"
                echo ""

                if [[ -f "$dir/description.txt" ]]; then
                    echo "**Description:** $(cat "$dir/description.txt")"
                fi

                if [[ -f "$dir/changes_stat.txt" ]]; then
                    echo ""
                    echo "**Changes:**"
                    echo '```'
                    cat "$dir/changes_stat.txt"
                    echo '```'
                fi

                echo ""
            fi
        done

        echo "## Benchmark Results"
        echo ""
        echo "See individual benchmark files in: $BASELINE_DIR"

    } > "$output_file"

    log_info "Summary saved to: $output_file"
    cat "$output_file"
}

# =============================================================================
# MAIN
# =============================================================================
case "${1:-help}" in
    start)
        shift
        cmd_start "$*"
        ;;
    benchmark)
        cmd_benchmark "$2"
        ;;
    profile)
        cmd_profile "$2"
        ;;
    compare)
        cmd_compare "$2" "$3"
        ;;
    record)
        cmd_record "$2" "$3"
        ;;
    history)
        cmd_history
        ;;
    summary)
        cmd_summary
        ;;
    help|*)
        cat << EOF
Optimization Workflow for Claude Code

Commands:
  start <description>           Start new optimization iteration
  benchmark <run_id>            Run benchmarks for iteration
  profile <run_id>              Run full profiling for iteration
  compare <base_id> <opt_id>    Compare two iterations
  record <run_id> <files>       Record files changed
  history                       Show optimization history
  summary                       Generate summary report

Typical Workflow:
  1. $0 start "Fuse SwiGLU activation"
  2. [Make code changes]
  3. [Rebuild: cmake --build build-cuda -j]
  4. $0 benchmark opt_YYYYMMDD_HHMMSS
  5. $0 compare baseline_id opt_YYYYMMDD_HHMMSS
  6. $0 record opt_YYYYMMDD_HHMMSS "file1.cu,file2.cu"
  7. [Iterate or finalize]

Environment Variables:
  MODEL_PATH     - Path to gpt-oss-20b.gguf
  RUN_ID         - Override run ID
EOF
        ;;
esac
