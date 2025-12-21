# Qwen3-Omni TTS Debugging Tools

Tools for debugging the Qwen3-Omni text-to-speech pipeline implementation in llama.cpp.

## Overview

This directory contains tools to extract, compare, and debug tensor outputs at each stage of the Qwen3-Omni TTS pipeline. These tools help identify where the llama.cpp implementation diverges from the HuggingFace reference.

## Files

### Main Tools

| File | Purpose |
|------|---------|
| **debug_hf_pipeline.py** | Extract reference tensors from HuggingFace at all pipeline stages |
| **load_tensor.py** | Load and inspect binary tensor files, convert to .npy format |
| **compare_tensors.py** | Compare HuggingFace vs llama.cpp tensors to find divergence |
| **example_workflow.sh** | Example end-to-end debugging workflow |

### Legacy Tools

| File | Purpose |
|------|---------|
| extract_embeddings.py | Simple Thinker embedding extraction (superseded by debug_hf_pipeline.py) |
| debug_hf_code2wav.py | Code2Wav-specific debugging |
| debug_dwconv.py | Depthwise convolution debugging |
| test_conv_transpose.py | Transpose convolution tests |
| test_dwconv.py | Depthwise convolution tests |
| quick_compare.py | Quick tensor comparison utility |

### Documentation

| File | Purpose |
|------|---------|
| **DEBUG_PIPELINE.md** | Comprehensive guide to the TTS pipeline and debugging strategy |
| **requirements.txt** | Python dependencies |
| **README.md** | This file |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Requirements:
- torch >= 2.0.0
- transformers >= 4.40.0
- numpy >= 1.24.0
- scipy >= 1.10.0 (optional, for WAV export)

### 2. Extract HuggingFace Reference Tensors

```bash
# Basic usage
python debug_hf_pipeline.py \
    --text "Hello world" \
    --output /tmp/hf_tensors/

# With custom model path
python debug_hf_pipeline.py \
    --text "Hello world" \
    --output /tmp/hf_tensors/ \
    --model /models/Qwen3-Omni-30B-A3B-Instruct
```

This creates 22+ numbered tensor files capturing each pipeline stage:
- Input tokenization
- Thinker embeddings and generation
- Text projection (Thinker → Talker)
- Talker prefill construction
- Codec token generation
- Code predictor output
- Code2Wav processing
- Final waveform

Plus an `output.wav` file you can listen to.

### 3. Inspect Tensors

```bash
# View tensor info
python load_tensor.py info /tmp/hf_tensors/15_talker_prefill_embeds.bin

# Convert all to .npy format
python load_tensor.py convert /tmp/hf_tensors/
```

### 4. Compare with llama.cpp

After running your llama.cpp implementation and saving tensors:

```bash
# Direct comparison (supports both .bin and .npy)
python compare_tensors.py /tmp/hf_tensors/ /tmp/cpp_tensors/

# Or convert first if needed
python load_tensor.py convert /tmp/hf_tensors/
python compare_tensors.py /tmp/hf_tensors/ /tmp/cpp_tensors/
```

The comparison tool will:
- Show shape, MSE, max difference, correlation for each tensor
- Identify the **first divergence point** (most likely bug location)
- Report which tensors match vs diverge

## Binary Tensor Format

Tensors are saved in a simple binary format with dimension headers:

```
[num_dims: uint32]
[dim0: uint32]
[dim1: uint32]
...
[data: float32 array in row-major order]
```

This format is:
- Simple to read from C++ or Python
- Self-describing (includes shape)
- Compact (no JSON overhead)
- Compatible with numpy via `load_tensor.py`

## Pipeline Stages

See [DEBUG_PIPELINE.md](DEBUG_PIPELINE.md) for detailed documentation of each stage.

Key stages to check:

1. **Tokenization** - Ensure token IDs match exactly
2. **Thinker embeddings** - Verify embedding layer weights
3. **Text projection** - 7168→3584 MLP (Thinker→Talker space)
4. **Prefill construction** - Exactly 9 positions, sum of text+codec embeddings
5. **Talker generation** - First divergence often occurs here
6. **Code predictor** - 16 codebooks per frame
7. **Code2Wav** - Embedding lookup and convolution

## Common Issues

### Special Token IDs

**Critical:** The TTS_BOS token ID in documentation is wrong.

- ❌ Docs say: `TTS_BOS = 151669`
- ✅ Actual value: `TTS_BOS = 151672`

Always verify against model config, not documentation.

### Dimension Ordering

- **PyTorch**: `[batch, seq, hidden]`
- **GGML**: `[hidden, seq, batch]` (reversed)

### Matrix Multiply Semantics

- **PyTorch**: `C = A @ B` means `C = A * B`
- **GGML**: `C = ggml_mul_mat(A, B)` means `C^T = A * B^T`

### Prefill Structure

Must be exactly **9 positions**:

```
pos | text_emb              | codec_emb
----|----------------------|------------------
0-2 | assistant_hidden[0-2]| zeros
3   | tts_pad_embed        | codec_nothink
4   | tts_pad_embed        | codec_think_bos
5   | tts_pad_embed        | codec_think_eos
6   | tts_pad_embed        | speaker_id
7   | tts_bos_embed        | codec_pad
8   | assistant_hidden[3]  | codec_bos
```

Final: `prefill[i] = text_emb[i] + codec_emb[i]`

## Example Workflow

```bash
# 1. Extract reference
python debug_hf_pipeline.py \
    --text "Hello" \
    --output /tmp/debug/ \
    --max-codec-tokens 10

# 2. Listen to reference audio
aplay /tmp/debug/output.wav

# 3. Run your implementation
./build/bin/llama-tts \
    --model model.gguf \
    --text "Hello" \
    --dump /tmp/cpp/

# 4. Compare
python compare_tensors.py /tmp/debug/ /tmp/cpp/

# Output shows first divergence:
# FIRST DIVERGENCE: 15_talker_prefill_embeds
# This is likely where the bug is!
```

## Debugging Strategy

1. **Start from the beginning** - Don't assume early stages are correct
2. **Compare layer by layer** - Find where divergence starts
3. **Check special tokens first** - Often the source of bugs
4. **Verify dimensions** - PyTorch vs GGML ordering
5. **Inspect intermediate values** - Not just final output
6. **Use reference audio** - Does HF implementation sound correct?

## Performance Tips

- Use `--max-codec-tokens 10` for quick tests (~0.4s audio)
- Use `--device cuda` if available (much faster)
- Start with short text like "Hello" for rapid iteration

## Additional Resources

- **Main docs**: [DEBUG_PIPELINE.md](DEBUG_PIPELINE.md)
- **HF implementation**: `/transformers/src/transformers/models/qwen3_omni_moe/modular_qwen3_omni_moe.py`
- **llama.cpp implementation**: `/llama.cpp/src/llama-model.cpp` (search for "qwen3omni")

## Support

For questions about the tools or TTS debugging, see:
- Project docs: `/llama.cpp/docs/qwen3-omni-*.md`
- Issue tracker: https://github.com/ggml-org/llama.cpp/issues/16186
