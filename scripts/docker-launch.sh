#!/bin/bash
# docker-launch.sh
# Launches NGC CUDA 13.0.1 container with workspace and models mounts
# Run from HOST (not inside container):
#   bash /home/trevor/Projects/llama.cpp/scripts/docker-launch.sh [optional-command]

set -e

WORKSPACE="/home/trevor/Projects/llama.cpp"
MODELS="/home/trevor/models"
CONTAINER_NAME="llama-fp4-dev"
IMAGE="nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu22.04"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ============================================================================
# Check Prerequisites
# ============================================================================
echo -e "${BLUE}Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}✗${NC}  Docker not installed"
    exit 1
fi

if ! docker images | grep -q "cuda:13.0.1-devel"; then
    echo -e "${YELLOW}⚠${NC}  Image not found locally, pulling..."
    docker pull "$IMAGE"
fi

if [ ! -d "$WORKSPACE" ]; then
    echo -e "${YELLOW}✗${NC}  Workspace not found: $WORKSPACE"
    exit 1
fi

echo -e "${GREEN}✓${NC}  All prerequisites met"
echo ""

# ============================================================================
# Check if container is already running
# ============================================================================
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}⚠${NC}  Container '$CONTAINER_NAME' is already running"
    echo "    Attaching to existing container..."
    echo ""
    docker attach "$CONTAINER_NAME"
    exit 0
fi

# Check if container exists but is stopped
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}⚠${NC}  Container '$CONTAINER_NAME' exists but is stopped"
    read -p "    Restart and attach? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker start -ai "$CONTAINER_NAME"
        exit 0
    fi
fi

# ============================================================================
# Launch New Container
# ============================================================================
echo -e "${BLUE}Launching NGC CUDA 13.0.1 container...${NC}"
echo ""
echo "Container Details:"
echo "  Name:      $CONTAINER_NAME"
echo "  Image:     $IMAGE"
echo "  Workspace: $WORKSPACE → /workspace"
echo "  Models:    $MODELS → /models"
echo ""

# If argument provided, run that command instead of bash
if [ $# -gt 0 ]; then
    docker run --rm \
        --gpus all \
        --name "$CONTAINER_NAME" \
        -v "$WORKSPACE:/workspace" \
        -v "$MODELS:/models" \
        -w /workspace \
        "$IMAGE" \
        "$@"
else
    # Launch interactive bash
    docker run -it --rm \
        --gpus all \
        --name "$CONTAINER_NAME" \
        -v "$WORKSPACE:/workspace" \
        -v "$MODELS:/models" \
        -w /workspace \
        "$IMAGE" \
        bash
fi

echo ""
echo -e "${GREEN}✓${NC}  Container exited"
