#!/bin/bash
# run-full-day1.sh
# Complete Day 1 Setup and Implementation Script
# This script automates the entire Day 1 workflow:
# 1. Environment setup (CUDA, tools, mounts verification)
# 2. Create FP4 foundation files
# 3. Configure CMake
# 4. Build and test
#
# Usage (inside container):
#   bash /workspace/scripts/run-full-day1.sh

set -e  # Exit on error

WORKSPACE="/workspace"
BUILD_DIR="$WORKSPACE/build"
SCRIPTS_DIR="$WORKSPACE/scripts"

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# Main Menu
# ============================================================================
show_menu() {
    clear
    echo "================================================================"
    echo "  FP4 Tensor Core - Day 1 Automated Setup"
    echo "  NVIDIA NGC CUDA 13.0.1 + DGX Spark GB10"
    echo "================================================================"
    echo ""
    echo "Select operation:"
    echo ""
    echo "  1) Full Setup (environment + foundation + build)"
    echo "  2) Setup Environment Only"
    echo "  3) Create Foundation Files Only"
    echo "  4) Build Only"
    echo "  5) Run Tests"
    echo "  6) Clean Build"
    echo ""
    echo "  0) Exit"
    echo ""
    echo "================================================================"
}

# ============================================================================
# Function: Setup Environment
# ============================================================================
setup_environment() {
    echo ""
    echo "Running environment setup..."
    bash "$SCRIPTS_DIR/setup-container-env.sh"
    return $?
}

# ============================================================================
# Function: Create Foundation Files
# ============================================================================
create_foundation() {
    echo ""
    echo "Creating Day 1 foundation files..."
    bash "$SCRIPTS_DIR/day1-foundation.sh"
    return $?
}

# ============================================================================
# Function: Build
# ============================================================================
build_project() {
    echo ""
    echo "================================================================"
    echo "Building llama.cpp with FP4 support..."
    echo "================================================================"
    echo ""

    cd "$BUILD_DIR"

    # Show CMake configuration
    echo -e "${BLUE}Building with:${NC}"
    echo "  - GGML_CUDA: ON"
    echo "  - CUDA_ARCHITECTURES: 121 (Blackwell GB10)"
    echo "  - Generator: Ninja"
    echo "  - Build Type: Release"
    echo ""

    NPROC=$(nproc)
    echo "Starting build with $NPROC parallel jobs..."
    echo ""

    if ninja -j"$NPROC"; then
        echo ""
        echo -e "${GREEN}✓ Build successful!${NC}"
        return 0
    else
        echo ""
        echo -e "${RED}✗ Build failed!${NC}"
        return 1
    fi
}

# ============================================================================
# Function: Run Tests
# ============================================================================
run_tests() {
    echo ""
    echo "================================================================"
    echo "Running verification tests..."
    echo "================================================================"
    echo ""

    # Check if build outputs exist
    if [ ! -f "$BUILD_DIR/bin/llama-cli" ]; then
        echo -e "${YELLOW}⚠${NC}  llama-cli not found. Building first..."
        build_project || return 1
    fi

    echo -e "${BLUE}Test 1:${NC} Check FP4 compilation flags"
    if grep -r "BLACKWELL_FP4_AVAILABLE" "$WORKSPACE/ggml/src/ggml-cuda/" > /dev/null; then
        echo -e "${GREEN}✓${NC}  Blackwell FP4 flags found in source"
    else
        echo -e "${RED}✗${NC}  Blackwell FP4 flags NOT found!"
        return 1
    fi

    echo ""
    echo -e "${BLUE}Test 2:${NC} Check FP4 files created"
    for file in fp4-types.cuh mma-fp4.cuh convert-mxfp4-fp4.cuh; do
        if [ -f "$WORKSPACE/ggml/src/ggml-cuda/$file" ]; then
            echo -e "${GREEN}✓${NC}  $file exists"
        else
            echo -e "${RED}✗${NC}  $file missing!"
            return 1
        fi
    done

    echo ""
    echo -e "${BLUE}Test 3:${NC} Binary version check"
    "$BUILD_DIR/bin/llama-cli" --version
    echo -e "${GREEN}✓${NC}  llama-cli runs successfully"

    echo ""
    echo -e "${GREEN}✅ All tests passed!${NC}"
    return 0
}

# ============================================================================
# Function: Clean Build
# ============================================================================
clean_build() {
    echo ""
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    echo -e "${GREEN}✓${NC}  Build directory cleaned"
    return 0
}

# ============================================================================
# Function: Full Setup (All steps)
# ============================================================================
full_setup() {
    echo ""
    echo -e "${BLUE}Step 1/4: Environment Setup${NC}"
    setup_environment || return 1

    echo ""
    echo -e "${BLUE}Step 2/4: Create Foundation Files${NC}"
    create_foundation || return 1

    echo ""
    echo -e "${BLUE}Step 3/4: Build Project${NC}"
    build_project || return 1

    echo ""
    echo -e "${BLUE}Step 4/4: Run Tests${NC}"
    run_tests || return 1

    echo ""
    echo "================================================================"
    echo -e "${GREEN}✅ Day 1 Complete!${NC}"
    echo "================================================================"
    echo ""
    echo "Summary:"
    echo "  ✓ Environment verified (CUDA 13.0.1, GB10, tools)"
    echo "  ✓ Blackwell detection added"
    echo "  ✓ FP4 type definitions created"
    echo "  ✓ MMA stub functions created"
    echo "  ✓ Conversion stubs created"
    echo "  ✓ Project compiled successfully"
    echo "  ✓ All tests passed"
    echo ""
    echo "Next steps (Day 2):"
    echo "  1. Implement actual PTX inline assembly for FP4 MMA"
    echo "  2. Create identity matrix test"
    echo "  3. Debug and verify tensor core execution"
    echo ""
    echo "Location: /workspace/ggml/src/ggml-cuda/mma-fp4.cuh"
    echo "================================================================"

    return 0
}

# ============================================================================
# Main Loop
# ============================================================================
main() {
    while true; do
        show_menu
        read -p "Enter choice [0-6]: " choice
        echo ""

        case $choice in
            1) full_setup; break ;;
            2) setup_environment; break ;;
            3) create_foundation; break ;;
            4) build_project; break ;;
            5) run_tests; break ;;
            6) clean_build; break ;;
            0) echo "Exiting..."; exit 0 ;;
            *)
                echo -e "${RED}Invalid choice. Please try again.${NC}"
                sleep 1
                ;;
        esac

        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${GREEN}Operation completed successfully.${NC}"
        else
            echo ""
            echo -e "${RED}Operation failed!${NC}"
        fi

        echo ""
        read -p "Press Enter to continue..."
    done
}

# ============================================================================
# Entry Point
# ============================================================================

# If called with argument, skip menu and run that operation
if [ $# -eq 0 ]; then
    # Interactive mode
    main
else
    # Non-interactive mode (useful for automation)
    case "$1" in
        "full")      full_setup ;;
        "env")       setup_environment ;;
        "foundation") create_foundation ;;
        "build")     build_project ;;
        "test")      run_tests ;;
        "clean")     clean_build ;;
        *)
            echo "Usage: $0 [full|env|foundation|build|test|clean]"
            echo ""
            echo "  full       - Run complete Day 1 setup"
            echo "  env        - Setup environment only"
            echo "  foundation - Create FP4 files only"
            echo "  build      - Build project"
            echo "  test       - Run tests"
            echo "  clean      - Clean build directory"
            exit 1
            ;;
    esac
fi
