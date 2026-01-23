# VL-Bench Docker image for DGX Spark / NGC
# Optimized for Qwen3-VL inference profiling on Blackwell GPUs
#
# Build: docker build -t vl-bench -f tools/vl-bench/Dockerfile .
# Run:   docker run --gpus all -v /path/to/models:/models vl-bench status

ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04
FROM ${BASE_IMAGE}

LABEL maintainer="llama.cpp VL-bench"
LABEL description="Vision-Language model benchmarking for llama.cpp"

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    jq \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for analysis
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    matplotlib

# Set working directory
WORKDIR /llama.cpp

# Copy source code
COPY . .

# Build llama.cpp with CUDA support
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="90;90a" \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j$(nproc)

# Create directories for results and models
RUN mkdir -p /models /results

# Environment variables
ENV MODEL_PATH=/models/model.gguf
ENV MMPROJ_PATH=/models/mmproj.gguf
ENV RESULTS_DIR=/results
ENV PATH="/llama.cpp/build/bin:${PATH}"

# Make scripts executable
RUN chmod +x tools/vl-bench/*.sh

# Default entrypoint
ENTRYPOINT ["tools/vl-bench/claude-driver.sh"]
CMD ["help"]
