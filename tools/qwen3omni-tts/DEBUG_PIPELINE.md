# Qwen3-Omni TTS Pipeline Debugging Guide

This guide explains how to use the `debug_hf_pipeline.py` script to extract reference tensors from the HuggingFace implementation for debugging the llama.cpp TTS implementation.

## Overview

The TTS pipeline has multiple stages where errors can accumulate:

1. **Tokenization** - Input text → token IDs
2. **Thinker Token Embeddings** - Token IDs → embeddings
3. **TTS Special Token Embeddings** - Special tokens for TTS control
4. **Thinker Generation** - Text generation with hidden states
5. **Text Projection** - Project Thinker hidden states to Talker space
6. **Talker Prefill** - Build initial 9-position prefill embeddings
7. **Talker Codec Generation** - Generate codec tokens
8. **Code Predictor** - Predict 16 codebooks per frame
9. **Code2Wav** - Convert codes to audio waveform

The debug script captures tensors at each stage for comparison with llama.cpp.

## Quick Start

### 1. Extract Reference Tensors from HuggingFace

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

# Generate more codec tokens for longer audio
python debug_hf_pipeline.py \
    --text "This is a longer test" \
    --output /tmp/hf_tensors/ \
    --max-codec-tokens 50
```

### 2. Inspect Extracted Tensors

```bash
# View info about a specific tensor
python load_tensor.py info /tmp/hf_tensors/02_thinker_tok_embd.bin

# Convert all tensors to .npy format (for existing tools)
python load_tensor.py convert /tmp/hf_tensors/
```

### 3. Compare with llama.cpp Implementation

After running your llama.cpp implementation and saving tensors:

```bash
# First convert HF tensors to .npy
python load_tensor.py convert /tmp/hf_tensors/

# Compare with existing tool
python compare_tensors.py /tmp/hf_tensors/ /tmp/cpp_tensors/
```

## Output Files

The script creates numbered tensor files in order of pipeline execution:

| File | Description | Shape (typical) |
|------|-------------|----------------|
| `01_input_ids.bin` | Tokenized input | `[1, seq_len]` |
| `02_thinker_tok_embd.bin` | Thinker token embeddings | `[1, seq_len, 7168]` |
| `03_tts_special_embeds_raw.bin` | TTS special tokens (before projection) | `[1, 3, 7168]` |
| `04_thinker_output_ids.bin` | Thinker generated tokens | `[1, total_len]` |
| `05_thinker_embed_l0.bin` | Thinker embeddings layer 0 | `[1, total_len, 7168]` |
| `06_thinker_hidden_l6.bin` | Thinker hidden states layer 6 | `[1, total_len, 7168]` |
| `07_tts_bos_projected.bin` | TTS_BOS after text_projection | `[1, 1, 3584]` |
| `08_tts_eos_projected.bin` | TTS_EOS after text_projection | `[1, 1, 3584]` |
| `09_tts_pad_projected.bin` | TTS_PAD after text_projection | `[1, 1, 3584]` |
| `10_assistant_hidden_projected.bin` | Assistant tokens after projection | `[1, seg_len, 3584]` |
| `11_assistant_text_hidden.bin` | Text part of prefill (9 positions) | `[1, 9, 3584]` |
| `12_codec_special_ids.bin` | Codec special token IDs | `[1, 6]` |
| `13_codec_special_embeds.bin` | Codec special embeddings | `[1, 6, 3584]` |
| `14_assistant_codec_hidden.bin` | Codec part of prefill (9 positions) | `[1, 9, 3584]` |
| `15_talker_prefill_embeds.bin` | Final prefill (text + codec sum) | `[1, 9, 3584]` |
| `16_trailing_text_hidden.bin` | Trailing text for continuation | `[1, trail_len, 3584]` |
| `17_talker_codec_tokens.bin` | Generated codec token IDs | `[1, n_tokens]` |
| `18_talker_codes.bin` | Code predictor output (16 codebooks) | `[1, 16, n_frames]` |
| `19_codes_with_offset.bin` | Codes with codebook offsets | `[1, 16, n_frames]` |
| `20_code_embeds.bin` | Individual codebook embeddings | `[1, 16, n_frames, 896]` |
| `21_code2wav_input.bin` | Sum of codebook embeddings | `[1, n_frames, 896]` |
| `22_waveform.bin` | Final audio waveform | `[1, 1, n_samples]` |
| `output.wav` | Playable audio file (24kHz) | - |

## Special Token IDs

The script prints these token IDs at runtime. Reference values:

- **TTS_PAD**: 151671
- **TTS_BOS**: 151672  (Note: docs incorrectly listed as 151669)
- **TTS_EOS**: 151673
- **CODEC_PAD**: 4196
- **CODEC_BOS**: 4197
- **CODEC_EOS**: 4198
- **CODEC_NOTHINK**: 4203
- **CODEC_THINK_BOS**: 4204
- **CODEC_THINK_EOS**: 4205

## Prefill Structure

The Talker prefill has exactly **9 positions**:

```
Position  | Text Embedding                  | Codec Embedding
----------|----------------------------------|------------------
0         | assistant_hidden[:, 0]          | zeros
1         | assistant_hidden[:, 1]          | zeros
2         | assistant_hidden[:, 2]          | zeros
3         | tts_pad_embed                   | codec_nothink_embed
4         | tts_pad_embed                   | codec_think_bos_embed
5         | tts_pad_embed                   | codec_think_eos_embed
6         | tts_pad_embed                   | speaker_embed
7         | tts_bos_embed                   | codec_pad_embed
8         | assistant_hidden[:, 3] (1st txt)| codec_bos_embed
```

Final embedding at each position = text_embedding + codec_embedding

## Text Projection MLP

The `text_projection` module transforms Thinker hidden states (7168-dim) to Talker space (3584-dim):

```python
# Architecture
linear_fc1: Linear(7168 -> 18432, bias=True)
act_fn: SiLU
linear_fc2: Linear(18432 -> 3584, bias=True)

# Forward pass
output = linear_fc2(silu(linear_fc1(hidden_state)))
```

## Code2Wav Embedding

Code2Wav uses 16 codebooks, each with 4096 entries:

```python
# Each codebook has an offset
code_offset = [0, 4096, 8192, ..., 61440]  # 16 offsets

# Embedding lookup
embedded[i] = code_embedding[codes[i] + code_offset[i]]

# Final input: mean across codebooks
code2wav_input = embedded.mean(dim=1)  # [batch, frames, 896]
```

## Debugging Strategy

1. **Start from input** - Verify tokenization matches
2. **Check embeddings** - Compare token embeddings layer by layer
3. **Verify special tokens** - TTS and codec special token embeddings
4. **Text projection** - Ensure MLP weights are correct
5. **Prefill construction** - Verify the 9-position structure exactly
6. **Codec generation** - First divergence often happens here
7. **Code predictor** - Check 16-codebook output
8. **Code2Wav** - Verify embedding lookup and convolution

## Common Issues

### Incorrect Special Token IDs
- TTS_BOS was documented as 151669 but is actually 151672
- Always verify IDs against model config

### Dimension Ordering
- PyTorch: `[batch, seq, hidden]`
- GGML: `[hidden, seq, batch]` (reversed)

### Matrix Multiply Semantics
- PyTorch: `C = A @ B` means `C = A * B`
- GGML: `C = ggml_mul_mat(A, B)` means `C^T = A * B^T`

### Prefill Position Mismatch
- Must be exactly 9 positions
- Text and codec embeddings must be summed, not concatenated

### Code Offset Calculation
- Each codebook has offset = codebook_index * codebook_size
- Offsets: [0, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 36864, 40960, 45056, 49152, 53248, 57344, 61440]

## Requirements

```bash
pip install torch transformers numpy scipy
```

Optional (for WAV export):
```bash
pip install scipy
```

## Performance Tips

- Use `--max-codec-tokens 10` for quick tests (generates ~0.4s audio)
- Use `--device cuda` if you have GPU available
- Default layer 6 is recommended for accept_hidden_layer (as per HF implementation)

## Example Workflow

```bash
# 1. Extract reference tensors
python debug_hf_pipeline.py \
    --text "Hello" \
    --output /tmp/debug/ \
    --max-codec-tokens 10

# 2. Check a specific tensor
python load_tensor.py info /tmp/debug/15_talker_prefill_embeds.bin

# 3. Convert to .npy for comparison
python load_tensor.py convert /tmp/debug/

# 4. Run your llama.cpp implementation (save tensors to /tmp/cpp/)
./llama-tts --model model.gguf --text "Hello" --dump /tmp/cpp/

# 5. Compare
python compare_tensors.py /tmp/debug/ /tmp/cpp/
```

## Additional Resources

- HuggingFace implementation: `transformers/src/transformers/models/qwen3_omni_moe/modular_qwen3_omni_moe.py`
- Key methods:
  - `_get_talker_assistant_parts()` (lines 2456-2511)
  - `generate()` (lines 2513-2696)
  - Text projection MLP (lines 1591-1599)
  - Code2Wav forward (lines 2384-2396)
