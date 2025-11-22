# 🚀 Day 1 Quick Start Guide

**FP4 Tensor Core Implementation for Blackwell GB10**

## Two-Minute Setup

### Option 1: Full Automated Setup (Recommended)

**From HOST machine:**
```bash
cd /home/trevor/Projects/llama.cpp
bash scripts/docker-launch.sh
```

**Inside Container:**
```bash
bash /workspace/scripts/run-full-day1.sh full
```

This runs the complete Day 1 workflow in sequence:
1. ✅ Environment setup (CUDA verification, tools, mounts)
2. ✅ Create FP4 foundation files
3. ✅ Configure CMake
4. ✅ Build project
5. ✅ Run tests

---

### Option 2: Step-by-Step with Menu

**From HOST:**
```bash
bash scripts/docker-launch.sh
```

**Inside Container:**
```bash
bash /workspace/scripts/run-full-day1.sh
```

Select operations from interactive menu:
- 1 = Full Setup (all steps)
- 2 = Environment only
- 3 = Create files only
- 4 = Build only
- 5 = Run tests
- 6 = Clean build

---

### Option 3: Manual Individual Scripts

**From HOST:**
```bash
bash scripts/docker-launch.sh
```

**Inside Container:**
```bash
# Step 1: Setup environment
bash /workspace/scripts/setup-container-env.sh

# Step 2: Create foundation files
bash /workspace/scripts/day1-foundation.sh

# Step 3: Build
cd /workspace/build
ninja -j$(nproc)

# Step 4: Test (optional)
bash /workspace/scripts/run-full-day1.sh test
```

---

## What Each Script Does

### `docker-launch.sh` (RUN FROM HOST)
- Checks Docker is installed
- Pulls latest NGC CUDA 13.0.1 image
- Mounts workspace and models directories
- Launches interactive container

**Usage:**
```bash
bash scripts/docker-launch.sh                  # Launch interactive shell
bash scripts/docker-launch.sh nvcc --version   # Run single command
```

---

### `setup-container-env.sh` (RUN INSIDE CONTAINER)
- Updates system packages
- Installs build tools (cmake, ninja, git, python3)
- Verifies CUDA 13.0.1 installation
- Checks GPU access (GB10 Blackwell)
- Verifies volume mounts
- Creates build directory

**Output:**
```
✓ CUDA 13.0.1 verified
✓ GPU: NVIDIA GB10
✓ /workspace mounted
✓ /models mounted
✓ Build directory created
```

---

### `day1-foundation.sh` (RUN INSIDE CONTAINER)
Creates four key components:

1. **Blackwell Detection** - Adds to `common.cuh`:
   - `#define GGML_CUDA_CC_BLACKWELL 1210`
   - `#define BLACKWELL_FP4_AVAILABLE`

2. **fp4-types.cuh** - FP4 tile type definitions
   - `tile<16,8,fp4_e2m1_packed>`
   - `tile<8,8,fp4_e2m1_packed>`
   - Register layout for packed FP4 values

3. **mma-fp4.cuh** - MMA stub function
   - `mma()` function with proper signature
   - Placeholder for PTX inline assembly
   - Proper preprocessor guards

4. **convert-mxfp4-fp4.cuh** - Conversion stubs
   - MXFP4 → E2M1 lookup table (ready-to-use)
   - `float_to_e2m1()` stub
   - `convert_mxfp4_to_nvfp4_block()` stub

---

### `run-full-day1.sh` (RUN INSIDE CONTAINER)

**Interactive Menu Mode:**
```bash
bash /workspace/scripts/run-full-day1.sh
```

**Direct Command Mode:**
```bash
bash /workspace/scripts/run-full-day1.sh full          # All steps
bash /workspace/scripts/run-full-day1.sh env           # Environment
bash /workspace/scripts/run-full-day1.sh foundation    # Files
bash /workspace/scripts/run-full-day1.sh build         # Compile
bash /workspace/scripts/run-full-day1.sh test          # Tests
bash /workspace/scripts/run-full-day1.sh clean         # Clean build
```

---

## Expected Output: Success

```
================================================================
✅ Day 1 Complete!
================================================================

Summary:
  ✓ Environment verified (CUDA 13.0.1, GB10, tools)
  ✓ Blackwell detection added
  ✓ FP4 type definitions created
  ✓ MMA stub functions created
  ✓ Conversion stubs created
  ✓ Project compiled successfully
  ✓ All tests passed

Next steps (Day 2):
  1. Implement actual PTX inline assembly for FP4 MMA
  2. Create identity matrix test
  3. Debug and verify tensor core execution

Location: /workspace/ggml/src/ggml-cuda/mma-fp4.cuh
================================================================
```

---

## Troubleshooting

### Issue: Docker image not found
```bash
docker pull nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu22.04
```

### Issue: GPU not detected in container
```bash
# Inside container - verify GPU access
nvidia-smi

# If fails, check host has nvidia-docker:
nvidia-docker --version
```

### Issue: Build fails with CUDA errors
```bash
# Clean and rebuild
rm -rf /workspace/build
mkdir -p /workspace/build
cd /workspace/build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -GNinja
ninja -j4  # Try with fewer jobs if still failing
```

### Issue: Files already exist
Scripts check for existing files and skip if already present. To force recreation:
```bash
# Remove old files
rm /workspace/ggml/src/ggml-cuda/{fp4-types,mma-fp4,convert-mxfp4-fp4}.cuh

# Re-run foundation script
bash /workspace/scripts/day1-foundation.sh
```

---

## Key Files Created

After Day 1 completion:

```
/workspace/
├── ggml/src/ggml-cuda/
│   ├── common.cuh                    [MODIFIED - Blackwell detection]
│   ├── fp4-types.cuh                 [NEW - Tile definitions]
│   ├── mma-fp4.cuh                   [NEW - MMA stubs]
│   └── convert-mxfp4-fp4.cuh         [NEW - Conversion stubs]
│
├── build/
│   ├── CMakeCache.txt
│   ├── CMakeFiles/
│   ├── bin/
│   │   └── llama-cli                 [Compiled binary]
│   └── ...
│
└── scripts/
    ├── docker-launch.sh              [Container launcher]
    ├── setup-container-env.sh        [Environment setup]
    ├── day1-foundation.sh            [File creation]
    ├── run-full-day1.sh              [Orchestration script]
    └── fp4_e2m1_analysis.py          [Already existed]
```

---

## Performance Expectations - Day 1

**Note:** Day 1 focuses on foundation - no performance improvements yet

- ✅ Build completes successfully
- ✅ Blackwell architecture detected
- ✅ FP4 data types compile
- ✅ MMA stubs callable (returns zeros)
- ❌ No actual tensor core execution yet (coming Day 2-3)

---

## Next Steps After Day 1

See `/workspace/docs/FP4_IMPLEMENTATION_PLAN.md` Days 4-7 for:

1. **PTX Assembly Implementation** - Replace `// TODO` in `mma-fp4.cuh`
   ```cuda
   asm volatile("mma.sync.aligned.kind::mxf4nvf4...")
   ```

2. **Identity Matrix Test** - Verify tensor core execution

3. **Conversion Functions** - Implement MXFP4 → NVFP4 conversion

4. **Integration** - Plug FP4 path into `vec_dot_mxfp4_q8_1`

---

## Useful Commands Inside Container

```bash
# Verify CUDA
nvcc --version
nvidia-smi

# Check code changes
cd /workspace && git diff

# Rebuild from clean
cd /workspace/build && ninja clean && ninja

# Monitor build progress
cd /workspace/build && ninja -v     # Verbose mode

# Test binary
./build/bin/llama-cli --version

# Run analysis tool
python3 /workspace/scripts/fp4_e2m1_analysis.py | head -30

# Get help on CMake config
cmake -LA /workspace/build
```

---

## Estimated Time

| Step | Time | Notes |
|------|------|-------|
| Setup environment | 5 min | Docker pull, tools install |
| Create foundation files | 10 min | Blackwell detection + 3 stubs |
| CMake configuration | 2 min | Arch 121 support |
| Build | 10-15 min | First build, full compilation |
| Tests | 2 min | Verification checks |
| **Total Day 1** | **30-40 min** | Fully automated |

---

## Support

- **Full Documentation:** `/workspace/docs/BLACKWELL_FP4_RESEARCH.md`
- **Implementation Plan:** `/workspace/docs/FP4_IMPLEMENTATION_PLAN.md`
- **Summary:** `/workspace/docs/FP4_RESEARCH_SUMMARY.md`

---

**Ready? Let's go! 🚀**

```bash
bash scripts/docker-launch.sh
```

Then inside container:
```bash
bash /workspace/scripts/run-full-day1.sh full
```
