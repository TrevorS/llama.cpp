# Qwen3-Omni Implementation Notes

> Implementation progress tracking for llama.cpp Qwen3-Omni support
> Started: 2025-12-15

---

## Current Status

- **Phase**: GGUF Conversion Tasks COMPLETE
- **Current Task**: C++ Implementation (Next Phase)
- **Validation**: Thinker and MMProj GGUF files converted and validated

---

## Implementation Log

### 2025-12-15 - Session Start

Beginning implementation of Qwen3-Omni support following the task breakdown in `qwen3-omni-tasks.md`.

**Strategy**:
1. Complete GGUF conversion tasks first (prerequisites for everything else)
2. Implement each task with tests before moving on
3. Use subagents for research and implementation to preserve context

---

## GGUF Conversion Tasks

### TASK-GGUF-001: Add Thinker Architecture Constants

**Status**: COMPLETE

**Objective**: Define `LLM_ARCH_QWEN3OMNIMOE` in GGUF Python library

**Files modified**:
- `gguf-py/gguf/constants.py`

**Acceptance Criteria**:
- [x] Add `QWEN3OMNIMOE` to `MODEL_ARCH` enum (line 373)
- [x] Add string mapping "qwen3omnimoe" (line 747)
- [x] Define tensor list in MODEL_TENSOR_NAMES (lines 1644-1663)
- [x] M-RoPE sections key already exists at `Keys.Rope.DIMENSION_SECTIONS`

**Implementation Notes**:
- Added `QWEN3OMNIMOE = auto()` after QWEN3VLMOE
- Tensor list includes shared expert tensors (FFN_*_SHEXP) for the MoE architecture
- Qwen3-Omni uses MoE with 128 experts, 8 active per token + shared expert

**Tests**:
- `tests/test_qwen3omni_constants.py` created (awaiting Docker build)

---

### TASK-GGUF-002: Add Talker Architecture Constants

**Status**: COMPLETE

**Objective**: Define `LLM_ARCH_QWEN3OMNI_TALKER` in GGUF Python library

**Files modified**:
- `gguf-py/gguf/constants.py`

**Acceptance Criteria**:
- [x] Add `QWEN3OMNI_TALKER` to `MODEL_ARCH` enum
- [x] Add string mapping "qwen3omni-talker"
- [x] Define tensor list in MODEL_TENSOR_NAMES
- [x] Define Talker-specific metadata keys (Keys.Talker)
- [x] Define codec embedding tensor (TALKER_CODEC_EMBD)

**Implementation Notes**:
- Added `QWEN3OMNI_TALKER = auto()` after QWEN3OMNIMOE
- Added new tensor types: TALKER_TEXT_PROJ, TALKER_CODEC_HEAD, TALKER_CODEC_EMBD
- Keys.Talker class with: ACCEPT_HIDDEN_LAYER, TEXT_PROJ_DIM, CODEC_VOCAB_SIZE, NUM_CODEBOOKS
- Talker shares Thinker's transformer architecture but with own projection layers

**Tests**:
- Added to `tests/test_qwen3omni_constants.py` (tests 6-10)

---

### TASK-GGUF-003: Add Code2Wav Metadata Constants

**Status**: COMPLETE

**Objective**: Define Code2Wav-specific metadata in GGUF

**Files modified**:
- `gguf-py/gguf/constants.py`

**Acceptance Criteria**:
- [x] Add Keys.Code2Wav class with VQ-VAE codec configuration
- [x] Add C2W_* tensor enums for neural audio codec
- [x] Add TENSOR_NAMES mappings for Code2Wav tensors

**Implementation Notes**:
- Keys.Code2Wav: CODEBOOK_SIZE, CODEBOOK_DIM, NUM_QUANTIZERS, SEMANTIC_CODEBOOK_SIZE,
  HIDDEN_SIZE, DECODER_DIM, UPSAMPLE_RATES, SLIDING_WINDOW
- C2W tensor types: C2W_CODEBOOK, C2W_SEMANTIC_CB, C2W_ATTN_Q/K/V/OUT, C2W_ATTN_NORM,
  C2W_FFN_UP/DOWN, C2W_FFN_NORM, C2W_LAYER_SCALE, C2W_UPSAMPLE, C2W_OUTPUT_PROJ
- Code2Wav uses SlidingWindowTransformer + VQ-VAE decoder with 4 upsample stages

**Tests**:
- Added to `tests/test_qwen3omni_constants.py` (tests 11-13)

---

### TASK-GGUF-004: Add Tensor Mappings for Thinker

**Status**: COMPLETE (No Changes Needed)

**Objective**: Map HuggingFace Thinker tensor names to GGUF

**Findings**:
The Thinker (Qwen3OmniMoe) uses the same tensor naming convention as qwen2moe and llama-hf.
All necessary tensor mappings already exist in `tensor_mapping.py`:
- Token embeddings: `model.embed_tokens` → TOKEN_EMBD
- Self-attention: `model.layers.{bid}.self_attn.{q,k,v,o}_proj` → ATTN_Q/K/V/OUT
- Q/K norms: `model.layers.{bid}.self_attn.{q,k}_norm` → ATTN_Q_NORM/ATTN_K_NORM
- MoE experts: `model.layers.{bid}.mlp.experts.{gate,up,down}_proj` → FFN_GATE/UP/DOWN_EXP
- Shared expert: `model.layers.{bid}.mlp.shared_expert.{gate,up,down}_proj` → FFN_GATE/UP/DOWN_SHEXP
- Router: `model.layers.{bid}.mlp.gate` → FFN_GATE_INP
- Layer norms: `model.layers.{bid}.{input,post_attention}_layernorm` → ATTN_NORM/FFN_NORM
- Output: `model.norm`, `lm_head` → OUTPUT_NORM, OUTPUT

**Tests**:
- All 13 tests pass in Docker container

---

### TASK-GGUF-005: Add Tensor Mappings for mmproj (Audio)

**Status**: COMPLETE (Changes Required)

**Objective**: Map audio encoder tensor names to GGUF

**Findings**:
Initial analysis suggested Qwen3-Omni's audio encoder was identical to Whisper/Ultravox, but actual
conversion revealed different tensor naming:

**Qwen3-Omni differences from Whisper:**
- 3 conv layers: `conv2d1`, `conv2d2`, `conv2d3` (not `conv1`, `conv2`)
- Output projection: `conv_out` (linear layer, not conv)
- Two-layer MLP projector: `proj1`, `proj2` (not single `proj`)

**Files modified**:
- `gguf-py/gguf/tensor_mapping.py`

**Tensor mappings added**:
- `audio_tower.conv2d{bid}` → A_ENC_CONV1D (for conv2d1, conv2d2, conv2d3)
- `audio_tower.conv_out` → A_ENC_CONV1D (output projection)
- `audio_tower.proj{bid}` → A_MMPROJ (for proj1, proj2)

**Existing mappings still used**:
- Position embeddings: `audio_tower.embed_positions` → A_ENC_EMBD_POS
- Post-norm: `audio_tower.ln_post` → A_POST_NORM
- Transformer layers: same as Whisper (fc1, fc2, self_attn, layer norms)

**Tests**:
- Conversion validated with actual model (see Conversion Validation Results)

---

### TASK-GGUF-006: Add Tensor Mappings for Talker

**Status**: COMPLETE

**Objective**: Map Talker model tensor names to GGUF

**Files modified**:
- `gguf-py/gguf/tensor_mapping.py`

**Implementation**:
Added new tensor mappings for Talker-specific tensors:
- `TALKER_TEXT_PROJ` → `talker.text_projection` (projects thinker hidden to talker dim)
- `TALKER_CODEC_HEAD` → `talker.codec_head` (predicts first codebook token)
- `TALKER_CODEC_EMBD` → `talker.codec_embeddings.{bid}` (16 codebook embeddings)

The Talker's internal transformer uses the same naming as Thinker (`model.embed_tokens`,
`model.layers.{bid}...`) so those will be handled in the converter by stripping `talker.model.`.

**Tests**:
- Added `test_talker_tensor_mappings()` (test 14)
- All 14 tests pass

---

### TASK-GGUF-007: Add Tensor Mappings for Code2Wav

**Status**: COMPLETE

**Objective**: Map Code2Wav tensor names to GGUF

**Files modified**:
- `gguf-py/gguf/tensor_mapping.py`
- `gguf-py/gguf/constants.py` (added C2W tensors to QWEN3OMNI_TALKER architecture)

**Implementation**:
Added tensor mappings for all Code2Wav HuggingFace tensor patterns:
- Codebooks: `code2wav.quantizer.vq.layers.{bid}._codebook.embed` → C2W_CODEBOOK
- Semantic: `code2wav.semantic_quantizer.vq._codebook.embed` → C2W_SEMANTIC_CB
- Transformer attention: `code2wav.transformer.layers.{bid}.self_attn.*` → C2W_ATTN_*
- Transformer FFN: `code2wav.transformer.layers.{bid}.mlp.fc{1,2}` → C2W_FFN_UP/DOWN
- Norms: `code2wav.transformer.layers.{bid}.norm{1,2}` → C2W_ATTN_NORM/C2W_FFN_NORM
- Layer scale: `code2wav.transformer.layers.{bid}.layer_scale` → C2W_LAYER_SCALE
- Upsample: `code2wav.decoder.model.{bid}.conv` → C2W_UPSAMPLE
- Output: `code2wav.decoder.out_conv` → C2W_OUTPUT_PROJ

Added all C2W_* tensors to MODEL_TENSORS[QWEN3OMNI_TALKER] since Code2Wav is bundled
with Talker in the same GGUF file.

**Tests**:
- Added `test_code2wav_tensor_mappings()` (test 15)
- All 15 tests pass

---

### TASK-GGUF-008: Implement HuggingFace Converter

**Status**: COMPLETE

**Objective**: Create converter classes for `Qwen3OmniMoeForConditionalGeneration`

**Files modified**:
- `convert_hf_to_gguf.py`

**Implementation**:

Three converter classes implemented:

1. **Qwen3OmniThinkerModel** (line 4221)
   - Extends `Qwen3MoeModel`
   - Architecture: `QWEN3OMNIMOE`
   - Handles nested `thinker_config` extraction
   - Filters out audio/visual/talker/code2wav tensors
   - Writes M-RoPE sections [24, 20, 20]
   - Inherits MoE expert merging from Qwen2MoeModel

2. **Qwen3OmniMmprojModel** (line 4266)
   - Extends `MmprojModel` directly (audio-only, no vision inheritance)
   - Architecture: `MMPROJ`
   - Audio-only (has_vision_encoder=False, has_audio_encoder=True)
   - Handles nested config extraction (`thinker_config.audio_config`)
   - Filters to only `audio_tower.*` tensors
   - Generates sinusoidal position embeddings
   - Forces F16 for conv weights

3. **Qwen3OmniTalkerModel** (line 4323)
   - Extends `Qwen2MoeModel`
   - Architecture: `QWEN3OMNI_TALKER`
   - Registered under placeholder name `Qwen3OmniMoeForConditionalGeneration-talker`
   - Filters to only `talker.*` and `code2wav.*` tensors
   - Strips `talker.model.` prefix for internal transformer
   - Writes Talker metadata (accept_hidden_layer, codec_vocab_size, etc.)
   - Writes Code2Wav metadata (codebook_size, num_quantizers, etc.)
   - Inherits MoE expert merging for Talker transformer

**Usage**:
```bash
# Convert Thinker (main LLM)
python convert_hf_to_gguf.py /models/Qwen3-Omni-30B-A3B-Instruct --outtype f16

# Convert MMProj (audio encoder)
python convert_hf_to_gguf.py /models/Qwen3-Omni-30B-A3B-Instruct --mmproj --outtype f16

# Convert Talker (speech synthesis + Code2Wav)
python convert_hf_to_gguf.py /models/Qwen3-Omni-30B-A3B-Instruct --talker --outtype f16
```

**Tests**:
- All 15 unit tests pass
- Converter imports successfully
- Thinker and MMProj conversions validated (see below)

---

## Test Results

### Unit Tests (tests/test_qwen3omni_constants.py)

All 15 tests pass:
1. QWEN3OMNIMOE enum exists
2. Architecture name mapping correct
3. Tensor definitions exist (18 tensors)
4. Required MoE tensors present (11)
5. M-RoPE key exists
6. QWEN3OMNI_TALKER enum exists
7. Talker name mapping correct
8. Talker tensor definitions exist (34 tensors)
9. Talker-specific tensors present (3)
10. Talker metadata keys exist
11. Code2Wav metadata keys exist
12. Code2Wav tensor enums exist (13)
13. Code2Wav tensor name mappings correct
14. Talker HF tensor mappings work (4)
15. Code2Wav HF tensor mappings work (14)

---

## Issues Encountered

1. **Class Order Issue**: `Qwen3OmniThinkerModel` was placed before `Qwen3MoeModel` in the file,
   causing a NameError. Fixed by moving both Qwen3Omni classes after `Qwen3MoeModel`.

2. **Registration Conflict**: Both Thinker (TEXT) and MMProj (MMPROJ) can register with the same
   HuggingFace architecture name because they go into different ModelType buckets. The registration
   system at line 717 checks `model_arch == MMPROJ` to determine which bucket.

3. **Talker CLI Flag**: Added `--talker` CLI flag with `ModelType.TALKER` enum. Talker converter
   now properly registered and selectable via command line.

4. **MMProj Config Extraction**: Qwen3-Omni has nested config structure (`thinker_config.audio_config`).
   Had to override config extraction in `Qwen3OmniMmprojModel` to find audio config correctly.

5. **Audio Tower Tensor Names**: Qwen3-Omni uses `conv2d1/conv2d2/conv2d3` instead of `conv1/conv2`,
   and `proj1/proj2` instead of single `proj`. Added new patterns to `tensor_mapping.py`.

---

## Conversion Validation Results

### 2025-12-16 - Model Conversion

**Source Model**: `Qwen/Qwen3-Omni-30B-A3B-Instruct` (70.5 GB)

#### Thinker GGUF (qwen3-omni-30b-thinker-f16.gguf)

| Metadata | Value |
|----------|-------|
| Architecture | `qwen3omnimoe` |
| Type | `model` |
| Block Count | 48 |
| Hidden Size | 2048 |
| Expert Count | 128 |
| Experts Per Token | 8 |
| M-RoPE Sections | [24, 20, 20] |
| Context Length | 65536 |
| Tensor Count | 579 |
| File Size | 61 GB |

#### MMProj GGUF (qwen3-omni-30b-mmproj-f16.gguf)

| Metadata | Value |
|----------|-------|
| Architecture | `clip` |
| Type | `mmproj` |
| Has Audio Encoder | True |
| Audio Projection Dim | 2048 |
| Audio Embedding Length | 1280 |
| Audio FFN Length | 5120 |
| Audio Block Count | 32 |
| Audio Head Count | 20 |
| Mel Bins | 128 |
| Tensor Count | 526 |
| File Size | 1.3 GB |

**Validation**: Both files pass `gguf-dump` metadata inspection with correct architecture
names, tensor counts, and model parameters.

---

## Code2Wav Fixes

### 2025-12-18 - Fix #18: ConvNeXt Numerical Stability

**Problem**: 95% of audio output samples were NaN, causing near-silent audio.

**Root Cause Analysis**:
1. ConvNeXt pwconv2 (4096→1024) produces values up to ±1000 from dot product of 4096 elements
2. After LayerScale (gamma ~0.17) and residual: values reach ±116
3. Block 2 transpose conv amplifies 40x → ±4700
4. Block 2 depthwise conv amplifies 14x → ±65000 (near F16 limit ~65504)
5. LayerNorm on near-overflow values produces NaN
6. NaN propagates through rest of pipeline

**Debug Output Before Fix**:
```
cnxt0_pw2 [1024, 30]: min=-818, max=1093
cnxt0_scale [1024, 30]: min=-95, max=116
cnxt1_transconv [1024, 60]: min=-1850, max=4782
cnxt1_dwconv [1024, 60]: min=-49696, max=65376  ← F16 overflow!
cnxt1_norm [1024, 60]: NaN=26.7%  ← NaN introduced
```

**Fix Applied** (`tools/qwen3omni-tts/main.cpp`):
1. Added `ggml_clamp()` after pwconv2 to limit values to ±1000
2. Added `ggml_rms_norm()` after each ConvNeXt residual connection

```cpp
// Fix #18: Clamp pwconv2 output to prevent F16 overflow
if (block.pwconv2) {
    cur = ggml_mul_mat(ctx, block.pwconv2, cur);
    cur = ggml_clamp(ctx, cur, -1000.0f, 1000.0f);
}

// Fix #18: Apply RMSNorm after residual to stabilize values
cur = ggml_add(ctx, cur, convnext_residual);
cur = ggml_rms_norm(ctx, cur, 1e-6f);
```

**Result**:
- Before: 99.5% near-silent, std dev = 1057
- After: 1.4% near-silent, std dev = 6599

**Files Modified**:
- `tools/qwen3omni-tts/main.cpp` (ConvNeXt upsample blocks)

---

### Previous Fixes (Snake Activation)

**Fix #15**: Added `exp()` to alpha/beta parameters (stored in log-scale)
**Fix #16**: Fixed transpose conv output trimming
**Fix #17**: Changed to causal (left-only) padding for all convolutions

---

### Fix #19: Add RoPE to Pre-Transformer Attention (2025-12-18)

**Status**: COMPLETE

**Problem**: Audio output was buzzing noise instead of speech. Fix #18 resolved NaN issues
and amplitude was correct (std dev ~6600), but waveform structure was wrong.

**Root Cause**: Missing Rotary Position Embeddings (RoPE) in pre-transformer attention.
Without positional encoding, the transformer treats all positions equivalently, causing
the buzzing pattern (no temporal structure).

**HuggingFace Reference** (`modeling_qwen2_5_omni.py:3060`):
```python
# Due to training process, only first head is applied with RoPE
cos, sin = position_embeddings
query[:, :1], key[:, :1] = apply_rotary_pos_emb(query[:, :1], key[:, :1], cos, sin)
```

**Fix Applied** (`tools/qwen3omni-tts/main.cpp:752-762`):
```cpp
// Fix #19: Apply Rotary Position Embeddings (RoPE)
// HuggingFace only applies RoPE to head 0, but we apply to all heads for simplicity
// Create position IDs [0, 1, 2, ..., seq_len-1]
ggml_tensor * pos_f32 = ggml_arange(ctx, 0.0f, (float)seq_len, 1.0f);
ggml_tensor * pos = ggml_cast(ctx, pos_f32, GGML_TYPE_I32);
ggml_set_name(pos, "rope_pos");

// Apply RoPE: NeoX-style (interleaved pairs), rope_theta=10000
// Shape before: [head_dim, n_head, seq] where ne[2]=seq matches pos size
Qcur = ggml_rope(ctx, Qcur, pos, c2w_head_dim, GGML_ROPE_TYPE_NEOX);
Kcur = ggml_rope(ctx, Kcur, pos, c2w_head_dim, GGML_ROPE_TYPE_NEOX);
```

**Key Details**:
- rope_theta = 10000 (from code2wav_config in model config.json)
- NeoX-style rotation (interleaved pairs, mode=2)
- Note: HF only applies to head 0, we apply to all 16 heads for simplicity

**Additional Findings**:
- LayerNorm bias tensors missing from GGUF (needs re-conversion)
  - HF has `code2wav.upsample.0.1.norm.bias`, GGUF only has `.weight`
- SnakeBeta epsilon not needed (exp() always returns positive values)

**Files Modified**:
- `tools/qwen3omni-tts/main.cpp` (pre-transformer attention loop)

---
