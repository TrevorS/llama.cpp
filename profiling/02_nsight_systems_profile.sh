#!/bin/bash
# 02_nsight_systems_profile.sh - Application-level GPU profiling with Nsight Systems
# This captures timeline, kernel launches, memory operations, CPU/GPU synchronization

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/profile_config.sh"

log_info "Starting Nsight Systems profiling for $MODEL_NAME"
log_info "Reports will be saved to: $NSIGHT_DIR"

# =============================================================================
# NSIGHT SYSTEMS: TIMELINE PROFILE
# =============================================================================
run_timeline_profile() {
    local profile_name="$1"
    local extra_args="$2"

    log_info "Running timeline profile: $profile_name"

    local report_file="$NSIGHT_DIR/nsys_${profile_name}_${RUN_ID}"

    # Run nsight systems profile
    "$NSYS" profile \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --sample=cpu \
        --cpuctxsw=process-tree \
        --output="$report_file" \
        --force-overwrite=true \
        -- "$LLAMA_BENCH" \
            -m "$MODEL_PATH" \
            -fa 1 \
            -ngl "$NUM_GPU_LAYERS" \
            --no-mmap \
            -r 1 \
            $extra_args

    log_info "Timeline profile saved to: ${report_file}.nsys-rep"

    # Export summary stats
    "$NSYS" stats \
        --report gputrace \
        --format csv \
        --output "${report_file}_gputrace" \
        "${report_file}.nsys-rep" 2>/dev/null || true

    "$NSYS" stats \
        --report gpukernsum \
        --format csv \
        --output "${report_file}_kernsum" \
        "${report_file}.nsys-rep" 2>/dev/null || true

    log_info "Stats exported to: ${report_file}_*.csv"
}

# =============================================================================
# PROFILE: PREFILL (compute-heavy, parallel)
# =============================================================================
profile_prefill() {
    log_info "Profiling PREFILL workload (prompt processing)..."

    run_timeline_profile "prefill_short" "-p 512 -n 0"
    run_timeline_profile "prefill_medium" "-p 2048 -n 0"
    run_timeline_profile "prefill_long" "-p 4096 -n 0"
}

# =============================================================================
# PROFILE: DECODE (memory-bound, sequential)
# =============================================================================
profile_decode() {
    log_info "Profiling DECODE workload (token generation)..."

    run_timeline_profile "decode_warmctx" "-p 128 -n 128"
    run_timeline_profile "decode_coldctx" "-p 2048 -n 64"
    run_timeline_profile "decode_deepctx" "-p 128 -n 64 -d 16384"
}

# =============================================================================
# PROFILE: MIXED WORKLOAD
# =============================================================================
profile_mixed() {
    log_info "Profiling MIXED workload..."

    run_timeline_profile "mixed_typical" "-t pg -p 1024 -n 128"
}

# =============================================================================
# ANALYSIS: KERNEL SUMMARY
# =============================================================================
analyze_kernel_summary() {
    log_info "Analyzing kernel summaries..."

    local analysis_file="$NSIGHT_DIR/kernel_analysis_${RUN_ID}.txt"

    {
        echo "KERNEL ANALYSIS SUMMARY"
        echo "======================="
        echo "Run ID: $RUN_ID"
        echo ""

        for csv_file in "$NSIGHT_DIR"/*_kernsum_*.csv 2>/dev/null; do
            if [[ -f "$csv_file" ]]; then
                echo "=== $(basename "$csv_file") ==="
                echo ""
                # Show top 20 kernels by time
                head -21 "$csv_file"
                echo ""
            fi
        done

        echo ""
        echo "Key Observations for MoE Model (gpt-oss-20b):"
        echo "----------------------------------------------"
        echo "Look for these kernel patterns:"
        echo "  - topk_moe*: MoE routing (should be fast, low % of time)"
        echo "  - mm*: Matrix multiply (bulk of compute)"
        echo "  - swiglu*: SwiGLU activation (fused vs unfused?)"
        echo "  - fattn*/flash_attn*: Attention (memory-bound)"
        echo "  - softmax*: Used in attention and routing"
        echo ""
        echo "Red flags:"
        echo "  - Many small kernel launches (CPU overhead)"
        echo "  - Large gaps between kernels (sync issues)"
        echo "  - Memory copy operations (should be minimal with unified memory)"

    } > "$analysis_file"

    log_info "Analysis saved to: $analysis_file"
    cat "$analysis_file"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================
main() {
    log_info "=========================================="
    log_info "NSIGHT SYSTEMS PROFILING"
    log_info "Model: $MODEL_NAME"
    log_info "Run ID: $RUN_ID"
    log_info "=========================================="

    if [[ ! -f "$MODEL_PATH" ]]; then
        log_error "Model file not found: $MODEL_PATH"
        exit 1
    fi

    if ! command -v "$NSYS" &> /dev/null; then
        log_error "nsys not found. Install NVIDIA Nsight Systems."
        log_error "On DGX systems: module load nsight-systems"
        exit 1
    fi

    # Run all profiles
    profile_prefill
    profile_decode
    profile_mixed

    # Analyze results
    analyze_kernel_summary

    log_info "=========================================="
    log_info "NSIGHT SYSTEMS PROFILING COMPLETE"
    log_info "Reports in: $NSIGHT_DIR"
    log_info ""
    log_info "To view reports interactively:"
    log_info "  nsys-ui $NSIGHT_DIR/nsys_*_${RUN_ID}.nsys-rep"
    log_info "=========================================="
}

# Allow running specific profiles
case "${1:-all}" in
    prefill)
        profile_prefill
        ;;
    decode)
        profile_decode
        ;;
    mixed)
        profile_mixed
        ;;
    analyze)
        analyze_kernel_summary
        ;;
    all)
        main
        ;;
    *)
        echo "Usage: $0 [prefill|decode|mixed|analyze|all]"
        exit 1
        ;;
esac
