# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

We develop inside Docker with CUDA support. This ensures consistent builds and access to GPU acceleration.

```bash
# Build the dev container
docker compose -f docker-compose.dev.yml build

# Start interactive dev shell with GPU
docker compose -f docker-compose.dev.yml run --rm dev

# Inside container:
rebuild                     # Full rebuild
rebuild -t llama-cli        # Build specific target
convert-model <args>        # Run convert_hf_to_gguf.py
llama-cli --help            # Run compiled tools (in PATH)
```

The container mounts:
- `.` → `/app/src` (your local source)
- `~/models` → `/models` (model files)
- `~/.cache/huggingface` → HF cache (faster downloads)

The initial build compiles from the cloned repo at `/app`. For development, work in `/app/src` and use `rebuild` to recompile.

## Build Commands (Native)

The Makefile is deprecated. Always use CMake:

```bash
# Basic CPU-only build
cmake -B build
cmake --build build --config Release -j $(nproc)

# Build with CUDA
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)

# Build with Metal (macOS)
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j $(nproc)

# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

Built binaries go to `build/bin/`.

## Testing

```bash
# Run all tests
ctest --test-dir build --output-on-failure -j $(nproc)

# Server tests (requires llama-server built)
cd tools/server/tests
pip install -r requirements.txt
./tests.sh

# Run specific server test
./tests.sh unit/test_chat_completion.py::test_invalid_chat_completion_req

# Debug server tests with verbose output
DEBUG=1 ./tests.sh -s -v -x
```

## Code Quality

```bash
# Format C++ code before committing
git clang-format

# Run pre-commit hooks
pre-commit run --all-files

# Local CI validation
mkdir tmp && bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

Add `ggml-ci` to commit message to trigger heavy CI on custom infrastructure.

## Project Validation Tools

**IMPORTANT: All validation must run inside Docker container.**

```bash
# Full validation inside Docker
docker compose -f docker-compose.dev.yml run --rm dev -c "
  # Install validation tools (ephemeral in each container)
  apt-get update -qq && apt-get install -y -qq clang-format >/dev/null 2>&1
  pip install --no-cache-dir --quiet flake8 flake8-no-print pre-commit

  # Python validation
  python3 -m py_compile convert_hf_to_gguf.py
  python3 -m flake8 convert_hf_to_gguf.py --max-line-length=120

  # C++ validation (requires clang-format 16+ for .clang-format compatibility)
  # Note: Ubuntu 22.04 has clang-format 14, which doesn't support all .clang-format options

  # Run tests
  ctest --test-dir build --output-on-failure -j \$(nproc)
"

# Quick test subset
docker compose -f docker-compose.dev.yml run --rm dev -c "
  cd build && ctest -j 8 -R 'test-log|test-arg-parser|test-opt'
"
```

**Validation Results (Last Run: 2025-12-16)**
- **Python Syntax**: ✓ All files pass
- **Python Linting (flake8)**: ✓ 0 issues in modified files
- **CMake Configuration**: ✓ Valid
- **CTest Suite**: 39/43 tests pass (91%)
  - Failed tests are environment/network-dependent, not code-related
  - test-tokenizers-ggml-vocabs, test-thread-safety, test-state-restore-fragmented (model download issues)
  - test-backend-ops (timeout)
- **C++ Formatting**: Pending (requires clang-format 16+, container has v14)

**Known Issues**:
- `.clang-format` uses `AlignTrailingComments: Kind` syntax (requires clang-format 16+)
- Dev container has clang-format 14, causing format validation to fail
- This is a project infrastructure issue, not related to code changes

## Project Permissions

- **Project Type**: personal
- **Direct Commits Allowed**: yes (feature branch)
- **Branch Protection**: none on feature branches
- **Last Validated**: 2025-12-16

## Architecture

### Core Libraries
- **`ggml/`**: Tensor computation library with backend implementations (CPU, CUDA, Metal, Vulkan, SYCL, etc.)
- **`src/`**: llama library implementation
- **`include/llama.h`**: Main C API header

### Key Source Files in `src/`
- `llama-model.cpp`: Model loading and inference graph building (~450k lines, largest file)
- `llama-arch.cpp`: Architecture definitions and tensor name mappings
- `llama-vocab.cpp`: Tokenizer implementations
- `llama-sampling.cpp`: Sampling strategies
- `llama-context.cpp`: Inference context management
- `llama-kv-cache.cpp`: Key-value cache for attention
- `llama-grammar.cpp`: GBNF grammar constraint parsing

### Backend Structure (`ggml/src/`)
Each backend has its own directory: `ggml-cuda/`, `ggml-metal/`, `ggml-vulkan/`, `ggml-sycl/`, etc.

### Main Tools (`tools/`)
- `cli/`: Main inference CLI (`llama-cli`)
- `server/`: OpenAI-compatible HTTP server (`llama-server`)
- `quantize/`: Model quantization
- `perplexity/`: Model evaluation
- `llama-bench/`: Performance benchmarking

### Python Conversion Scripts (root)
- `convert_hf_to_gguf.py`: Convert HuggingFace models to GGUF format
- `gguf-py/`: Python GGUF library with tensor mappings in `gguf/constants.py` and `gguf/tensor_mapping.py`

## Adding New Model Support

1. **Python conversion** (`convert_hf_to_gguf.py`):
   - Add `@ModelBase.register("YourModelForCausalLM")` class
   - Define tensor layout in `gguf-py/gguf/constants.py`
   - Map tensor names in `gguf-py/gguf/tensor_mapping.py`

2. **C++ architecture** (`src/`):
   - Add enum in `llama-arch.h`
   - Add name mapping in `llama-arch.cpp`
   - Add graph builder in `llama-model.cpp` (inherit from `llm_graph_context`)

## Coding Style

- 4-space indentation, 120 column limit
- Pointer style: `void * ptr`, reference style: `int & ref`
- Use `snake_case` for functions/variables
- Sized integers (`int32_t`) in public API
- Enum values: `UPPERCASE_WITH_PREFIX` (e.g., `LLAMA_VOCAB_TYPE_SPM`)
- Avoid new dependencies; keep cross-platform compatibility
- Tensor dimensions are reversed from PyTorch order
- Matrix multiply: `C = ggml_mul_mat(ctx, A, B)` means C^T = A * B^T

## Naming Convention

Optimize for longest common prefix:
```cpp
// Good
int number_small;
int number_big;

// Avoid
int small_number;
int big_number;
```

Function pattern: `<class>_<action>_<noun>` (e.g., `llama_model_init`, `llama_sampler_get_seed`)

## Development Notes (Learned from Experience)

### Docker Environment

- **ALWAYS run tests in Docker** - The dev container has all dependencies pre-installed
- **Architecture matters**: NVIDIA CUDA containers for arm64 (DGX Spark) need to be built correctly
- If you see "cannot execute binary file", the image was built for the wrong architecture - rebuild it
- Container is configured for Blackwell GB10 (CUDA arch 121) with unified memory support

### Running Python Tests

```bash
# Run tests in Docker container (preferred method)
docker compose -f docker-compose.dev.yml run --rm dev python3 tests/your_test.py

# Or with the built image directly
docker run --rm -v "$(pwd)":/app/src llama-cpp-dev:latest python3 tests/your_test.py
```

### GGUF Python Library

- Constants are in `gguf-py/gguf/constants.py`
- `MODEL_ARCH` is an `IntEnum` - add new architectures with `= auto()`
- `MODEL_ARCH_NAMES` dict maps enum to string names (like "qwen3omnimoe")
- `MODEL_TENSOR_NAMES` dict defines which tensors each architecture expects
- Metadata keys use `{arch}` placeholder that gets replaced at runtime
- M-RoPE support: `Keys.Rope.DIMENSION_SECTIONS` already exists for multimodal position encoding

### Adding New Model Architecture

1. Add enum to `MODEL_ARCH` class
2. Add string mapping to `MODEL_ARCH_NAMES` dict
3. Add tensor list to `MODEL_TENSOR_NAMES` dict
4. For MoE models, include `FFN_*_EXP` tensors for experts
5. For models with shared experts, include `FFN_*_SHEXP` tensors

### Current Project: Qwen3-Omni

Implementation docs:
- `docs/qwen3-omni-tasks.md` - Task breakdown
- `docs/qwen3-omni-requirements.md` - Requirements
- `docs/qwen3-omni-implementation-plan.md` - Architecture details
- `docs/qwen3-notes.md` - Implementation progress notes
