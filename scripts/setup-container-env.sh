#!/bin/bash
# setup-container-env.sh
# Automated setup for NGC CUDA 13.0.1 container environment on DGX Spark
# Run this inside the container: bash /workspace/scripts/setup-container-env.sh

set -e  # Exit on error

echo "================================================================"
echo "  NGC CUDA 13.0.1 Container Environment Setup"
echo "  DGX Spark - GB10 Blackwell FP4 Implementation"
echo "================================================================"
echo ""

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# Step 1: Update system packages
# ============================================================================
echo -e "${BLUE}[1/6]${NC} Updating system packages..."
apt-get update > /dev/null 2>&1
echo -e "${GREEN}✓${NC} System packages updated"

# ============================================================================
# Step 2: Install build dependencies
# ============================================================================
echo -e "${BLUE}[2/6]${NC} Installing build dependencies..."
apt-get install -y \
  cmake \
  build-essential \
  git \
  python3 \
  python3-pip \
  vim \
  ninja-build \
  wget \
  curl \
  > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Build tools installed (cmake, ninja, git, python3)"

# ============================================================================
# Step 3: Verify CUDA environment
# ============================================================================
echo ""
echo -e "${BLUE}[3/6]${NC} Verifying CUDA 13.0.1 installation..."
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}')
echo -e "${GREEN}✓${NC} CUDA version: $CUDA_VERSION"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
echo -e "${GREEN}✓${NC} GPU: $GPU_NAME"
echo -e "${GREEN}✓${NC} Driver: $DRIVER_VERSION"

# ============================================================================
# Step 4: Verify workspace and models mounts
# ============================================================================
echo ""
echo -e "${BLUE}[4/6]${NC} Verifying volume mounts..."

if [ -d /workspace ]; then
    CODE_FILES=$(ls /workspace/*.cpp /workspace/*.md 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} /workspace mounted ($CODE_FILES files found)"
else
    echo -e "${YELLOW}✗${NC} /workspace not mounted!"
    exit 1
fi

if [ -d /models ]; then
    MODEL_FILES=$(ls /models/*.gguf 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} /models mounted ($MODEL_FILES model files found)"
else
    echo -e "${YELLOW}⚠${NC}  /models not mounted (optional for Week 1)"
fi

# ============================================================================
# Step 5: Verify git status
# ============================================================================
echo ""
echo -e "${BLUE}[5/6]${NC} Verifying git repository..."
cd /workspace
BRANCH=$(git branch --show-current)
STATUS=$(git status --short | wc -l)
echo -e "${GREEN}✓${NC} Branch: $BRANCH"
echo -e "${GREEN}✓${NC} Uncommitted changes: $STATUS files"

# ============================================================================
# Step 6: Create build directory
# ============================================================================
echo ""
echo -e "${BLUE}[6/6]${NC} Creating build directory..."
mkdir -p /workspace/build
echo -e "${GREEN}✓${NC} Build directory created"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================================"
echo -e "${GREEN}✅ Container environment ready!${NC}"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  1. cd /workspace"
echo "  2. bash scripts/day1-foundation.sh"
echo ""
echo "Environment Summary:"
echo "  - CUDA: $CUDA_VERSION"
echo "  - GPU: $GPU_NAME"
echo "  - Driver: $DRIVER_VERSION"
echo "  - Workspace: /workspace"
echo "  - Models: /models"
echo "  - Branch: $BRANCH"
echo ""
echo "Ready to implement FP4 tensor core support! 🚀"
echo "================================================================"
