#!/bin/bash
# profile_config.sh - Configuration for gpt-oss-20b profiling on DGX Spark
# This file is sourced by other profiling scripts

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Model path (update this to your actual model location)
export MODEL_PATH="${MODEL_PATH:-/path/to/gpt-oss-20b.gguf}"
export MODEL_NAME="gpt-oss-20b"

# Model architecture details (for reference)
# - Total params: ~20B
# - Active params: ~3.6-4B (MoE sparse activation)
# - Experts: 64 total, top-k activation
# - Activation: SwiGLU (OpenAI variant)
# - Native quantization: MXFP4

# =============================================================================
# HARDWARE CONFIGURATION (DGX Spark / Grace Blackwell GB10)
# =============================================================================
export GPU_DEVICE="${GPU_DEVICE:-0}"
export NUM_GPU_LAYERS="${NUM_GPU_LAYERS:-999}"  # Offload all layers

# DGX Spark specs:
# - 128GB unified LPDDR5x memory (273 GB/s)
# - 1 PFLOP FP4 with sparsity
# - NVLink-C2C 900 GB/s CPU-GPU interconnect

# =============================================================================
# BENCHMARK PARAMETERS
# =============================================================================
# Prompt lengths to test (tokens)
export PROMPT_LENGTHS="${PROMPT_LENGTHS:-128,512,2048,4096}"

# Generation lengths to test (tokens)
export GEN_LENGTHS="${GEN_LENGTHS:-32,128,256}"

# Batch sizes for batched inference testing
export BATCH_SIZES="${BATCH_SIZES:-1,4,8,16,32}"

# Context depths for attention profiling
export CONTEXT_DEPTHS="${CONTEXT_DEPTHS:-0,4096,8192,16384,32768}"

# Number of repetitions for benchmarks
export BENCH_REPS="${BENCH_REPS:-5}"

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================
export PROFILE_ROOT="${PROFILE_ROOT:-$(dirname "$0")}"
export RESULTS_DIR="${PROFILE_ROOT}/results"
export NSIGHT_DIR="${PROFILE_ROOT}/nsight_reports"
export BASELINE_DIR="${RESULTS_DIR}/baseline"
export OPTIMIZATION_DIR="${RESULTS_DIR}/optimization"

# Create directories if they don't exist
mkdir -p "$RESULTS_DIR" "$NSIGHT_DIR" "$BASELINE_DIR" "$OPTIMIZATION_DIR"

# =============================================================================
# LLAMA.CPP PATHS
# =============================================================================
export LLAMA_ROOT="${LLAMA_ROOT:-$(dirname "$PROFILE_ROOT")}"
export LLAMA_BENCH="${LLAMA_ROOT}/build-cuda/bin/llama-bench"
export LLAMA_CLI="${LLAMA_ROOT}/build-cuda/bin/llama-cli"
export LLAMA_SERVER="${LLAMA_ROOT}/build-cuda/bin/llama-server"
export BATCHED_BENCH="${LLAMA_ROOT}/build-cuda/bin/batched-bench"
export PERPLEXITY="${LLAMA_ROOT}/build-cuda/bin/llama-perplexity"

# =============================================================================
# NSIGHT TOOLS
# =============================================================================
export NSYS="${NSYS:-nsys}"
export NCU="${NCU:-ncu}"

# Nsight Systems options
export NSYS_OPTIONS="--trace=cuda,nvtx,osrt --cuda-memory-usage=true"

# Nsight Compute options (deep kernel analysis)
export NCU_OPTIONS="--set full --section LaunchStats --section Occupancy --section SpeedOfLight --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis --section SourceCounters"

# =============================================================================
# KEY KERNELS TO PROFILE (for gpt-oss-20b MoE + SwiGLU)
# =============================================================================
# These are the critical kernels based on model architecture
export KEY_KERNELS=(
    "topk_moe"           # MoE top-k routing
    "mm_ids_helper"      # MoE expert dispatch
    "softmax"            # Attention + routing softmax
    "swiglu"             # SwiGLU activation (OpenAI variant)
    "flash_attn"         # Flash attention
    "fattn"              # llama.cpp flash attention variants
    "mmq"                # Quantized matrix multiply
    "mmvq"               # Matrix-vector quantized multiply
    "rope"               # Rotary position embedding
    "rms_norm"           # RMS normalization
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

check_prereqs() {
    local missing=0

    if [[ ! -f "$LLAMA_BENCH" ]]; then
        log_error "llama-bench not found at $LLAMA_BENCH"
        log_error "Build with: cmake -B build-cuda -DGGML_CUDA=ON && cmake --build build-cuda -j"
        missing=1
    fi

    if ! command -v "$NSYS" &> /dev/null; then
        log_error "nsys (Nsight Systems) not found. Install NVIDIA Nsight tools."
        missing=1
    fi

    if ! command -v "$NCU" &> /dev/null; then
        log_error "ncu (Nsight Compute) not found. Install NVIDIA Nsight tools."
        missing=1
    fi

    return $missing
}

timestamp() {
    date '+%Y%m%d_%H%M%S'
}

# Generate a unique run ID
RUN_ID="${RUN_ID:-$(timestamp)}"
export RUN_ID

echo "Profile configuration loaded. Run ID: $RUN_ID"
