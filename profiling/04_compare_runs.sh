#!/bin/bash
# 04_compare_runs.sh - Compare performance between optimization iterations
# Use this to validate that changes improve performance

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/profile_config.sh"

# =============================================================================
# USAGE
# =============================================================================
usage() {
    cat << EOF
Usage: $0 <baseline_run_id> <optimized_run_id> [output_file]

Compare performance between two benchmark runs.

Arguments:
  baseline_run_id   - Run ID of the baseline measurement
  optimized_run_id  - Run ID of the optimized measurement
  output_file       - Optional: output file for comparison (default: stdout)

Example:
  $0 20241218_100000 20241218_120000
  $0 baseline_v1 swiglu_fused_v1 comparison.txt
EOF
}

if [[ $# -lt 2 ]]; then
    usage
    exit 1
fi

BASELINE_ID="$1"
OPTIMIZED_ID="$2"
OUTPUT_FILE="${3:-/dev/stdout}"

# =============================================================================
# COMPARE JSON RESULTS
# =============================================================================
compare_json_results() {
    local baseline_file="$1"
    local optimized_file="$2"
    local test_name="$3"

    if [[ ! -f "$baseline_file" ]] || [[ ! -f "$optimized_file" ]]; then
        echo "  [SKIP] Missing files for $test_name"
        return
    fi

    echo "=== $test_name ===" >> "$OUTPUT_FILE"

    if command -v jq &> /dev/null; then
        # Use jq for proper JSON parsing
        local baseline_pp=$(jq -r 'map(select(.test_type == "pp")) | .[0].avg_ts // 0' "$baseline_file")
        local optimized_pp=$(jq -r 'map(select(.test_type == "pp")) | .[0].avg_ts // 0' "$optimized_file")

        local baseline_tg=$(jq -r 'map(select(.test_type == "tg")) | .[0].avg_ts // 0' "$baseline_file")
        local optimized_tg=$(jq -r 'map(select(.test_type == "tg")) | .[0].avg_ts // 0' "$optimized_file")

        if [[ "$baseline_pp" != "0" ]] && [[ "$optimized_pp" != "0" ]]; then
            local pp_speedup=$(echo "scale=2; $optimized_pp / $baseline_pp" | bc)
            local pp_pct=$(echo "scale=1; ($optimized_pp - $baseline_pp) / $baseline_pp * 100" | bc)
            echo "  Prefill: $baseline_pp -> $optimized_pp t/s (${pp_speedup}x, ${pp_pct}%)" >> "$OUTPUT_FILE"
        fi

        if [[ "$baseline_tg" != "0" ]] && [[ "$optimized_tg" != "0" ]]; then
            local tg_speedup=$(echo "scale=2; $optimized_tg / $baseline_tg" | bc)
            local tg_pct=$(echo "scale=1; ($optimized_tg - $baseline_tg) / $baseline_tg * 100" | bc)
            echo "  Decode:  $baseline_tg -> $optimized_tg t/s (${tg_speedup}x, ${tg_pct}%)" >> "$OUTPUT_FILE"
        fi
    else
        echo "  [NOTE] Install jq for detailed comparison" >> "$OUTPUT_FILE"
        echo "  Baseline: $(head -5 "$baseline_file")" >> "$OUTPUT_FILE"
        echo "  Optimized: $(head -5 "$optimized_file")" >> "$OUTPUT_FILE"
    fi

    echo "" >> "$OUTPUT_FILE"
}

# =============================================================================
# MAIN COMPARISON
# =============================================================================
main() {
    {
        echo "=============================================="
        echo "PERFORMANCE COMPARISON"
        echo "=============================================="
        echo "Baseline:  $BASELINE_ID"
        echo "Optimized: $OPTIMIZED_ID"
        echo "Timestamp: $(date)"
        echo ""

        # Compare basic throughput
        compare_json_results \
            "$BASELINE_DIR/basic_throughput_${BASELINE_ID}.json" \
            "$BASELINE_DIR/basic_throughput_${OPTIMIZED_ID}.json" \
            "Basic Throughput"

        # Compare prefill scaling
        compare_json_results \
            "$BASELINE_DIR/prefill_scaling_${BASELINE_ID}.json" \
            "$BASELINE_DIR/prefill_scaling_${OPTIMIZED_ID}.json" \
            "Prefill Scaling"

        # Compare decode scaling
        compare_json_results \
            "$BASELINE_DIR/decode_scaling_${BASELINE_ID}.json" \
            "$BASELINE_DIR/decode_scaling_${OPTIMIZED_ID}.json" \
            "Decode Scaling"

        # Compare context depth
        compare_json_results \
            "$BASELINE_DIR/context_depth_${BASELINE_ID}.json" \
            "$BASELINE_DIR/context_depth_${OPTIMIZED_ID}.json" \
            "Context Depth"

        echo "=============================================="
        echo "SUMMARY"
        echo "=============================================="

    } > "$OUTPUT_FILE" 2>&1

    if [[ "$OUTPUT_FILE" != "/dev/stdout" ]]; then
        log_info "Comparison saved to: $OUTPUT_FILE"
        cat "$OUTPUT_FILE"
    fi
}

main
