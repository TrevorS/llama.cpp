#!/bin/bash
# 03_nsight_compute_profile.sh - Deep kernel-level analysis with Nsight Compute
# This profiles individual CUDA kernels for bottleneck identification

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/profile_config.sh"

log_info "Starting Nsight Compute profiling for $MODEL_NAME"
log_info "Reports will be saved to: $NSIGHT_DIR"

# =============================================================================
# KERNEL FILTER PATTERNS
# =============================================================================
# For gpt-oss-20b MoE model, focus on these critical kernels

# MoE routing kernels
MOE_KERNELS="topk_moe|mm_ids_helper|softmax_warp"

# Matrix multiplication kernels (MXFP4 quantized)
MATMUL_KERNELS="mmq|mmvq|mmvf|mul_mat"

# Activation kernels (SwiGLU)
ACTIVATION_KERNELS="swiglu|silu|gelu"

# Attention kernels
ATTENTION_KERNELS="flash_attn|fattn|softmax"

# Normalization kernels
NORM_KERNELS="rms_norm|layer_norm"

# RoPE kernels
ROPE_KERNELS="rope"

# Combine all for comprehensive profiling
ALL_KEY_KERNELS="${MOE_KERNELS}|${MATMUL_KERNELS}|${ACTIVATION_KERNELS}|${ATTENTION_KERNELS}|${NORM_KERNELS}|${ROPE_KERNELS}"

# =============================================================================
# DEEP KERNEL PROFILE
# =============================================================================
profile_kernels() {
    local profile_name="$1"
    local kernel_filter="$2"
    local workload_args="$3"

    log_info "Deep profiling: $profile_name (filter: $kernel_filter)"

    local report_file="$NSIGHT_DIR/ncu_${profile_name}_${RUN_ID}"

    # Run Nsight Compute with full metrics
    "$NCU" \
        --set full \
        --section LaunchStats \
        --section Occupancy \
        --section SpeedOfLight \
        --section SpeedOfLight_HierarchicalSingleRooflineChart \
        --section SpeedOfLight_RooflineChart \
        --section MemoryWorkloadAnalysis \
        --section MemoryWorkloadAnalysis_Chart \
        --section ComputeWorkloadAnalysis \
        --section SchedulerStats \
        --section WarpStateStats \
        --section SourceCounters \
        --kernel-name-base function \
        --kernel-name regex:"$kernel_filter" \
        --launch-skip 10 \
        --launch-count 20 \
        --output "$report_file" \
        --force-overwrite \
        -- "$LLAMA_BENCH" \
            -m "$MODEL_PATH" \
            -fa 1 \
            -ngl "$NUM_GPU_LAYERS" \
            --no-mmap \
            -r 1 \
            $workload_args

    log_info "Deep profile saved to: ${report_file}.ncu-rep"

    # Export summary CSV
    "$NCU" --import "$report_file.ncu-rep" \
        --csv \
        --page raw \
        > "${report_file}_raw.csv" 2>/dev/null || true

    log_info "Raw metrics exported to: ${report_file}_raw.csv"
}

# =============================================================================
# PROFILE: MOE ROUTING
# =============================================================================
profile_moe() {
    log_info "Profiling MoE routing kernels..."

    profile_kernels "moe_routing" "$MOE_KERNELS" "-p 512 -n 0"
}

# =============================================================================
# PROFILE: MATRIX MULTIPLICATION (MXFP4)
# =============================================================================
profile_matmul() {
    log_info "Profiling matrix multiplication kernels..."

    profile_kernels "matmul_prefill" "$MATMUL_KERNELS" "-p 1024 -n 0"
    profile_kernels "matmul_decode" "$MATMUL_KERNELS" "-p 128 -n 64"
}

# =============================================================================
# PROFILE: SWIGLU ACTIVATION
# =============================================================================
profile_swiglu() {
    log_info "Profiling SwiGLU activation kernels..."

    profile_kernels "swiglu" "$ACTIVATION_KERNELS" "-p 512 -n 0"
}

# =============================================================================
# PROFILE: ATTENTION
# =============================================================================
profile_attention() {
    log_info "Profiling attention kernels..."

    profile_kernels "attention_short" "$ATTENTION_KERNELS" "-p 512 -n 0"
    profile_kernels "attention_long" "$ATTENTION_KERNELS" "-p 2048 -n 0 -d 8192"
}

# =============================================================================
# PROFILE: ALL KEY KERNELS
# =============================================================================
profile_all_kernels() {
    log_info "Profiling all key kernels..."

    profile_kernels "all_key_prefill" "$ALL_KEY_KERNELS" "-p 1024 -n 0"
    profile_kernels "all_key_decode" "$ALL_KEY_KERNELS" "-p 128 -n 64"
}

# =============================================================================
# QUICK SCAN (fast overview)
# =============================================================================
quick_scan() {
    log_info "Running quick kernel scan..."

    local report_file="$NSIGHT_DIR/ncu_quickscan_${RUN_ID}"

    "$NCU" \
        --section LaunchStats \
        --section SpeedOfLight \
        --launch-skip 5 \
        --launch-count 50 \
        --output "$report_file" \
        --force-overwrite \
        -- "$LLAMA_BENCH" \
            -m "$MODEL_PATH" \
            -fa 1 \
            -ngl "$NUM_GPU_LAYERS" \
            --no-mmap \
            -r 1 \
            -p 512 -n 32

    log_info "Quick scan saved to: ${report_file}.ncu-rep"
}

# =============================================================================
# ANALYSIS: BOTTLENECK IDENTIFICATION
# =============================================================================
analyze_bottlenecks() {
    log_info "Analyzing kernel bottlenecks..."

    local analysis_file="$NSIGHT_DIR/bottleneck_analysis_${RUN_ID}.txt"

    {
        echo "KERNEL BOTTLENECK ANALYSIS"
        echo "=========================="
        echo "Run ID: $RUN_ID"
        echo "Model: $MODEL_NAME (MoE with SwiGLU)"
        echo ""

        echo "Analysis Guide for gpt-oss-20b:"
        echo "-------------------------------"
        echo ""
        echo "1. MoE ROUTING (topk_moe, mm_ids_helper):"
        echo "   - Should be <5% of total time"
        echo "   - Look for: Warp divergence due to load imbalance"
        echo "   - Optimization: Expert batching, better load balancing"
        echo ""
        echo "2. MATRIX MULTIPLICATION (mmq, mmvq for MXFP4):"
        echo "   - Should be 50-70% of total time"
        echo "   - Look for: Memory bandwidth saturation (check SOL%)"
        echo "   - Key metric: Compute throughput vs theoretical max"
        echo "   - For MXFP4: Check FP4 tensor core utilization"
        echo ""
        echo "3. SWIGLU ACTIVATION (swiglu_oai):"
        echo "   - Can be 10-20% of time if not fused"
        echo "   - Look for: Kernel fusion opportunities"
        echo "   - Check: Is gate and up projection fused with activation?"
        echo ""
        echo "4. ATTENTION (fattn*, flash_attn):"
        echo "   - Typically memory-bound"
        echo "   - Check: L2 cache hit rate"
        echo "   - Optimization: KV cache management"
        echo ""
        echo "5. RMS NORM:"
        echo "   - Should be very small (<2%)"
        echo "   - If high: Check for unnecessary syncs"
        echo ""

        echo "Roofline Analysis Key Points:"
        echo "-----------------------------"
        echo "- If kernel is below the roofline ceiling: room for optimization"
        echo "- Memory-bound (left of ridge): improve data reuse, caching"
        echo "- Compute-bound (right of ridge): improve instruction throughput"
        echo ""

        echo "DGX Spark Specific (GB10):"
        echo "--------------------------"
        echo "- Memory BW: 273 GB/s (unified LPDDR5x)"
        echo "- FP4 Peak: 1 PFLOPS (with sparsity)"
        echo "- Tensor Core Utilization target: >80%"
        echo ""

        # List generated reports
        echo "Generated Reports:"
        ls -la "$NSIGHT_DIR"/ncu_*_${RUN_ID}* 2>/dev/null || echo "  (none found)"

    } > "$analysis_file"

    log_info "Analysis saved to: $analysis_file"
    cat "$analysis_file"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================
main() {
    log_info "=========================================="
    log_info "NSIGHT COMPUTE PROFILING"
    log_info "Model: $MODEL_NAME"
    log_info "Run ID: $RUN_ID"
    log_info "=========================================="

    if [[ ! -f "$MODEL_PATH" ]]; then
        log_error "Model file not found: $MODEL_PATH"
        exit 1
    fi

    if ! command -v "$NCU" &> /dev/null; then
        log_error "ncu not found. Install NVIDIA Nsight Compute."
        log_error "Note: ncu requires elevated permissions or --target-processes all"
        exit 1
    fi

    # Run profiling (this can take a while)
    quick_scan
    profile_moe
    profile_matmul
    profile_swiglu
    profile_attention

    # Analyze results
    analyze_bottlenecks

    log_info "=========================================="
    log_info "NSIGHT COMPUTE PROFILING COMPLETE"
    log_info "Reports in: $NSIGHT_DIR"
    log_info ""
    log_info "To view reports interactively:"
    log_info "  ncu-ui $NSIGHT_DIR/ncu_*_${RUN_ID}.ncu-rep"
    log_info "=========================================="
}

# Allow running specific profiles
case "${1:-all}" in
    quick)
        quick_scan
        ;;
    moe)
        profile_moe
        ;;
    matmul)
        profile_matmul
        ;;
    swiglu)
        profile_swiglu
        ;;
    attention)
        profile_attention
        ;;
    analyze)
        analyze_bottlenecks
        ;;
    all)
        main
        ;;
    *)
        echo "Usage: $0 [quick|moe|matmul|swiglu|attention|analyze|all]"
        exit 1
        ;;
esac
