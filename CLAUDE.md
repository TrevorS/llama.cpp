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
