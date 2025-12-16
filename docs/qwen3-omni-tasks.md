# Qwen3-Omni-30B-A3B Implementation Tasks

> Implementable task breakdown for llama.cpp Qwen3-Omni support
> **Source**: [qwen3-omni-requirements.md](qwen3-omni-requirements.md)
> **Spec**: [qwen3-omni-implementation-plan.md](qwen3-omni-implementation-plan.md)

---

## Task Overview

| Category | Task Count | Complexity |
|----------|------------|------------|
| GGUF Conversion | 8 | Medium |
| Thinker (Main LLM) | 6 | Medium |
| Audio Encoder | 4 | Low |
| Vision Encoder | 3 | Low |
| Talker Model | 7 | High |
| Code Predictor | 4 | Medium |
| Code2Wav | 8 | Very High |
| Server API | 6 | Medium |
| CLI Tool | 4 | Medium |
| Testing & Validation | 8 | Medium |

**Total**: ~58 tasks

---

## GGUF Conversion Tasks

### TASK-GGUF-001: Add Thinker Architecture Constants

**Description**: Define `LLM_ARCH_QWEN3OMNIMOE` in GGUF Python library

**Acceptance Criteria**:
- [ ] Add `QWEN3OMNIMOE` to `MODEL_ARCH` enum in `gguf-py/gguf/constants.py`
- [ ] Define all Thinker tensor keys matching existing Qwen3MOE pattern
- [ ] Add M-RoPE section metadata key: `qwen3omni.mrope_sections`

**Implementation Approach**: Extend existing Qwen3MOE definitions

**Files**:
- `gguf-py/gguf/constants.py`

**Tests**:
- Unit test: Verify enum values are unique and accessible
- Integration: Constants load correctly in convert script

**Prerequisites**: None

---

### TASK-GGUF-002: Add Talker Architecture Constants

**Description**: Define `LLM_ARCH_QWEN3OMNI_TALKER` in GGUF Python library

**Acceptance Criteria**:
- [ ] Add `QWEN3OMNI_TALKER` to `MODEL_ARCH` enum
- [ ] Define Talker-specific metadata keys:
  - `talker.hidden_size` (1024)
  - `talker.block_count` (20)
  - `talker.expert_count` (128)
  - `talker.experts_used_count` (6)
  - `talker.accept_hidden_layer` (18)
  - `talker.vocab_size` (3072)
- [ ] Define codec embedding keys for 16 codebooks

**Implementation Approach**: New architecture following existing patterns

**Files**:
- `gguf-py/gguf/constants.py`

**Prerequisites**: TASK-GGUF-001

---

### TASK-GGUF-003: Add Code2Wav Metadata Constants

**Description**: Define Code2Wav-specific metadata in GGUF

**Acceptance Criteria**:
- [ ] Add metadata keys:
  - `code2wav.codebook_size` (2048)
  - `code2wav.codebook_dim` (512)
  - `code2wav.num_quantizers` (16)
  - `code2wav.semantic_codebook_size` (4096)
  - `code2wav.hidden_size` (1024)
  - `code2wav.decoder_dim` (1536)
  - `code2wav.upsample_rates` ([8, 5, 4, 3])
  - `code2wav.sliding_window` (72)

**Implementation Approach**: New subsystem metadata

**Files**:
- `gguf-py/gguf/constants.py`

**Prerequisites**: None (parallel with TASK-GGUF-001)

---

### TASK-GGUF-004: Add Tensor Mappings for Thinker

**Description**: Map HuggingFace Thinker tensor names to GGUF

**Acceptance Criteria**:
- [ ] Map `model.layers.{i}.self_attn.*` tensors
- [ ] Map `model.layers.{i}.self_attn.{q,k}_norm.weight` (Qwen3-style)
- [ ] Map `model.layers.{i}.mlp.experts.{j}.*` tensors
- [ ] Map `model.layers.{i}.mlp.shared_expert.*` tensors
- [ ] Map `model.layers.{i}.mlp.gate.weight` (MoE router)

**Implementation Approach**: Extend Qwen3MOE mappings with M-RoPE awareness

**Files**:
- `gguf-py/gguf/tensor_mapping.py`

**Tests**:
- Verify all HF tensor names map correctly

**Prerequisites**: TASK-GGUF-001

---

### TASK-GGUF-005: Add Tensor Mappings for mmproj (Audio)

**Description**: Map audio encoder tensor names to GGUF

**Acceptance Criteria**:
- [ ] Map `audio_tower.conv1.weight`, `audio_tower.conv2.weight`
- [ ] Map `audio_tower.embed_positions.weight`
- [ ] Map `audio_tower.layers.{bid}.self_attn.*` tensors
- [ ] Map `audio_tower.layers.{bid}.fc{1,2}.weight`
- [ ] Map `audio.multi_modal_projector.linear_{bid}.weight`

**Implementation Approach**: Similar to existing Whisper encoder mappings

**Files**:
- `gguf-py/gguf/tensor_mapping.py`

**Prerequisites**: None

---

### TASK-GGUF-006: Add Tensor Mappings for Talker

**Description**: Map Talker model tensor names to GGUF

**Acceptance Criteria**:
- [ ] Map `talker.text_projection.{weight,bias}`
- [ ] Map `talker.model.embed_tokens.weight`
- [ ] Map `talker.model.layers.{i}.*` (MoE layers)
- [ ] Map `talker.codec_head.weight`
- [ ] Map `talker.codec_embeddings.{0-15}.weight` (16 codebook embeddings)

**Implementation Approach**: New mappings for Talker architecture

**Files**:
- `gguf-py/gguf/tensor_mapping.py`

**Prerequisites**: TASK-GGUF-002

---

### TASK-GGUF-007: Add Tensor Mappings for Code2Wav

**Description**: Map Code2Wav tensor names to GGUF

**Acceptance Criteria**:
- [ ] Map `code2wav.codebook.{0-15}.weight` (16 quantizer codebooks)
- [ ] Map `code2wav.semantic_codebook.weight`
- [ ] Map `code2wav.transformer.layers.{i}.*`
- [ ] Map `code2wav.transformer.layers.{i}.layer_scale`
- [ ] Map `code2wav.upsample.{0-3}.conv.{weight,bias}`
- [ ] Map `code2wav.convnext.{i}.*` (depthwise, pointwise convs)
- [ ] Map `code2wav.output_proj.weight`

**Implementation Approach**: New mappings for VQ-VAE decoder

**Files**:
- `gguf-py/gguf/tensor_mapping.py`

**Prerequisites**: TASK-GGUF-003

---

### TASK-GGUF-008: Implement HuggingFace Converter

**Description**: Create converter class for `Qwen3OmniMoeForConditionalGeneration`

**Acceptance Criteria**:
- [ ] Register `@ModelBase.register("Qwen3OmniMoeForConditionalGeneration")`
- [ ] Implement `set_gguf_parameters()` with all metadata
- [ ] Implement `modify_tensors()` for tensor preprocessing
- [ ] Generate three GGUF files:
  - `qwen3-omni-*.gguf` (Thinker)
  - `mmproj-qwen3-omni-*.gguf` (Vision + Audio encoders)
  - `talker-qwen3-omni-*.gguf` (Talker + Code Predictor + Code2Wav)
- [ ] Implement quantization rules:
  - Keep `mlp.gate` at F16
  - Keep `codec_embeddings` at F16
  - Keep `code2wav.codebook.*` at F16
  - Keep `code2wav.upsample.*.conv.*` at F16

**Implementation Approach**: Follow existing multimodal converter patterns (Qwen2VL, LLaVA)

**Files**:
- `convert_hf_to_gguf.py`

**Tests**:
- Convert F16 reference model successfully
- Verify tensor counts match expectations
- Verify metadata is correct

**Prerequisites**: TASK-GGUF-001 through TASK-GGUF-007

---

## Thinker Model Tasks

### TASK-THINKER-001: Add Architecture Enum

**Description**: Register `LLM_ARCH_QWEN3OMNIMOE` in C++ llama library

**Acceptance Criteria**:
- [ ] Add `LLM_ARCH_QWEN3OMNIMOE` to `llm_arch` enum in `src/llama-arch.h`
- [ ] Add architecture name mapping in `src/llama-arch.cpp`
- [ ] Wire up architecture detection in model loading

**Implementation Approach**: Follow existing architecture registration pattern

**Files**:
- `src/llama-arch.h`
- `src/llama-arch.cpp`

**Tests**:
- Architecture enum is recognized
- Model loading detects Qwen3-Omni type

**Prerequisites**: TASK-GGUF-008

---

### TASK-THINKER-002: Create Graph Builder

**Description**: Implement Thinker inference graph in `src/models/qwen3omni.cpp`

**Acceptance Criteria**:
- [ ] Create `llm_build_qwen3omni` graph builder
- [ ] Support 48 layers, hidden size 2048
- [ ] Implement Q/K normalization (Qwen3 style)
- [ ] Support GQA: 32 attention heads, 4 KV heads
- [ ] Implement MoE routing: 128 experts, 8 active per token
- [ ] Support shared expert (768 FFN) + sparse experts (768 FFN each)

**Implementation Approach**: Extend `llm_build_qwen3moe` with M-RoPE awareness

**Files**:
- `src/models/qwen3omni.cpp` (new)
- `src/llama-model.cpp` (registration)

**Tests**:
- Forward pass produces correct output shape
- Compare against HuggingFace reference

**Prerequisites**: TASK-THINKER-001

---

### TASK-THINKER-003: Implement TMRoPE Position Encoding

**Description**: Implement M-RoPE with sections [24, 20, 20] for temporal/spatial encoding

**Acceptance Criteria**:
- [ ] Section 1 (24 dims): Temporal position shared across modalities
- [ ] Section 2 (20 dims): Spatial Y/Height for vision, 0 for audio/text
- [ ] Section 3 (20 dims): Spatial X/Width for vision, 0 for audio/text
- [ ] Audio/video frames synchronized at 25 tokens/second temporal rate
- [ ] Position calculation: `position_id_per_seconds = 25`

**Implementation Approach**: Extend existing `ggml_rope_multi()` implementation from Qwen2VL

**Files**:
- `src/models/qwen3omni.cpp`

**Tests**:
- Verify position IDs align across modalities
- Test: video frame 15 at 30fps and audio at 0.5s both get temporal_pos = 12

**Prerequisites**: TASK-THINKER-002

---

### TASK-THINKER-004: Add Layer 18 Hidden State Extraction

**Description**: Enable extraction of hidden states from Thinker layer 18 for Talker

**Acceptance Criteria**:
- [ ] Store layer 18 hidden states during forward pass
- [ ] Provide API to access extracted hidden states
- [ ] Handle batch processing correctly
- [ ] Memory-efficient: only store when Talker is loaded

**Implementation Approach**: Add extraction hook in graph builder

**Files**:
- `src/models/qwen3omni.cpp`
- `src/llama-context.cpp`

**Tests**:
- Hidden states have correct shape [batch, seq, 2048]
- Values match HuggingFace layer 18 output

**Prerequisites**: TASK-THINKER-002

---

### TASK-THINKER-005: Add Special Token Handling

**Description**: Handle Qwen3-Omni special tokens in vocabulary

**Acceptance Criteria**:
- [ ] Recognize audio tokens:
  - `<audio_start>` (151669)
  - `<audio_end>` (151670)
  - `<audio>` (151675)
  - `<tts_bos>` (151672)
  - `<tts_eos>` (151673)
  - `<tts_pad>` (151671)
- [ ] Recognize vision tokens:
  - `<vision_start>` (151652)
  - `<vision_end>` (151653)
  - `<image>` (151655)
  - `<video>` (151656)

**Implementation Approach**: Extend vocabulary handling in tokenizer

**Files**:
- `src/llama-vocab.cpp`

**Tests**:
- Token IDs correctly identified
- Tokenization/detokenization round-trips

**Prerequisites**: TASK-THINKER-001

---

### TASK-THINKER-006: Register in mtmd Multimodal System

**Description**: Wire Thinker into multimodal embedding merge system

**Acceptance Criteria**:
- [ ] Add `MTMD_MODEL_TYPE_QWEN3OMNI` model type
- [ ] Register model in mtmd loader
- [ ] Support multimodal input embedding merge

**Implementation Approach**: Follow existing Qwen2VL/Qwen3VL patterns

**Files**:
- `tools/mtmd/mtmd.cpp`
- `tools/mtmd/mtmd.h`

**Tests**:
- Model loads correctly via mtmd
- Multimodal inputs are processed

**Prerequisites**: TASK-THINKER-002, TASK-AUDIO-003, TASK-VISION-002

---

## Audio Encoder Tasks

### TASK-AUDIO-001: Add 128 Mel-Bin Support

**Description**: Extend audio preprocessing for 128 mel bins (vs Whisper's 80)

**Acceptance Criteria**:
- [ ] Configurable mel bin count
- [ ] Generate 128 mel-bin spectrograms for Qwen3-Omni
- [ ] Maintain backward compatibility with Whisper (80 bins)

**Implementation Approach**: Parameterize existing mel spectrogram code

**Files**:
- `tools/mtmd/mtmd-audio.cpp`

**Tests**:
- 128-bin spectrogram matches HuggingFace output
- Whisper models still work with 80 bins

**Prerequisites**: None

---

### TASK-AUDIO-002: Create Audio Encoder Graph Builder

**Description**: Implement Whisper-style audio encoder with Qwen3-Omni params

**Acceptance Criteria**:
- [ ] 32 transformer layers
- [ ] d_model = 1280, 20 attention heads
- [ ] FFN dim = 5120
- [ ] Conv chunk size 500
- [ ] Max source positions 1500
- [ ] Project output to 2048 dimensions

**Implementation Approach**: Extend `tools/mtmd/models/whisper-enc.cpp`

**Files**:
- `tools/mtmd/models/qwen3omni-audio.cpp` (new)

**Tests**:
- Output shape matches expected [batch, seq, 2048]
- Embedding values match HuggingFace

**Prerequisites**: TASK-AUDIO-001

---

### TASK-AUDIO-003: Implement Audio Projector

**Description**: Project audio encoder output to Thinker embedding space

**Acceptance Criteria**:
- [ ] Multi-layer projector: 1280 → 2048 dimensions
- [ ] Match `audio.multi_modal_projector` weights

**Implementation Approach**: Linear projection layers

**Files**:
- `tools/mtmd/models/qwen3omni-audio.cpp`

**Tests**:
- Projected embeddings match HuggingFace

**Prerequisites**: TASK-AUDIO-002

---

### TASK-AUDIO-004: Add Audio Temporal Position Calculation

**Description**: Calculate TMRoPE temporal positions for audio tokens

**Acceptance Criteria**:
- [ ] Position = (frame_idx / sample_rate) × 25 tokens/second
- [ ] Spatial positions (Y, X) = 0 for audio
- [ ] Positions align with video frames at same timestamp

**Implementation Approach**: Extend position encoding in mtmd

**Files**:
- `tools/mtmd/mtmd-audio.cpp`

**Tests**:
- Audio at 0.5s → temporal_pos = 12
- Positions align with synchronized video

**Prerequisites**: TASK-THINKER-003

---

## Vision Encoder Tasks

### TASK-VISION-001: Create Vision Encoder Graph Builder

**Description**: Implement ViT-27L with deepstack for Qwen3-Omni

**Acceptance Criteria**:
- [ ] 27 transformer layers
- [ ] Hidden size 1152, 16 attention heads
- [ ] FFN size 4304
- [ ] 768×768 images, 16×16 patches
- [ ] Deepstack feature extraction at layers [8, 16, 24]
- [ ] Temporal patch size 2 for video
- [ ] Project output to 2048 dimensions
- [ ] Spatial merge factor 2

**Implementation Approach**: Extend `tools/mtmd/models/qwen3vl.cpp` (95% reuse)

**Files**:
- `tools/mtmd/models/qwen3omni-vision.cpp` (new)

**Tests**:
- Output shape matches expected
- Deepstack features extracted correctly

**Prerequisites**: None

---

### TASK-VISION-002: Add Vision Projector

**Description**: Project vision encoder output to Thinker embedding space

**Acceptance Criteria**:
- [ ] Multi-layer projector: 1152 → 2048 dimensions
- [ ] Handle deepstack concatenation

**Implementation Approach**: Follow Qwen3VL pattern

**Files**:
- `tools/mtmd/models/qwen3omni-vision.cpp`

**Tests**:
- Projected embeddings match HuggingFace

**Prerequisites**: TASK-VISION-001

---

### TASK-VISION-003: Add Vision Temporal Position Calculation

**Description**: Calculate TMRoPE 3D positions for vision tokens

**Acceptance Criteria**:
- [ ] Temporal position = frame_idx × position_id_per_seconds
- [ ] Height position = patch_row
- [ ] Width position = patch_col
- [ ] Example: video 30fps, frame 15 → temporal = (15/30) × 25 = 12

**Implementation Approach**: Extend position encoding

**Files**:
- `tools/mtmd/models/qwen3omni-vision.cpp`

**Tests**:
- 3D positions match HuggingFace
- Temporal aligns with audio

**Prerequisites**: TASK-THINKER-003

---

## Talker Model Tasks

### TASK-TALKER-001: Add Talker Architecture Enum

**Description**: Register `LLM_ARCH_QWEN3OMNI_TALKER` in C++ llama library

**Acceptance Criteria**:
- [ ] Add `LLM_ARCH_QWEN3OMNI_TALKER` to `llm_arch` enum
- [ ] Add architecture name mapping
- [ ] Wire up architecture detection

**Implementation Approach**: Follow existing architecture registration pattern

**Files**:
- `src/llama-arch.h`
- `src/llama-arch.cpp`

**Prerequisites**: TASK-GGUF-002

---

### TASK-TALKER-002: Create Talker Graph Builder

**Description**: Implement Talker transformer graph

**Acceptance Criteria**:
- [ ] 20 transformer layers
- [ ] Hidden size 1024, 16 attention heads, 2 KV heads
- [ ] MoE: 128 experts, 6 active per token
- [ ] Shared expert (768 FFN) + sparse experts (384 FFN each)
- [ ] M-RoPE sections [24, 20, 20]
- [ ] Max positions 65,536

**Implementation Approach**: New model file extending MoE base

**Files**:
- `src/models/qwen3omni-talker.cpp` (new)

**Tests**:
- Forward pass produces correct output shape
- Compare against HuggingFace

**Prerequisites**: TASK-TALKER-001

---

### TASK-TALKER-003: Implement Text Projection MLP

**Description**: Project Thinker hidden states to Talker dimension

**Acceptance Criteria**:
- [ ] Linear projection: 2048 → 1024 dimensions
- [ ] Include bias term
- [ ] Accept hidden states from Thinker layer 18

**Implementation Approach**: Simple linear layer

**Files**:
- `src/models/qwen3omni-talker.cpp`

**Tests**:
- Projection output matches HuggingFace

**Prerequisites**: TASK-THINKER-004, TASK-TALKER-002

---

### TASK-TALKER-004: Implement Codec Head

**Description**: Predict first codebook token from Talker hidden states

**Acceptance Criteria**:
- [ ] Linear projection: 1024 → 3072 (Talker vocab size)
- [ ] Output logits for first codebook token
- [ ] Support autoregressive generation

**Implementation Approach**: Add output layer to Talker graph

**Files**:
- `src/models/qwen3omni-talker.cpp`

**Tests**:
- Logits match HuggingFace
- Sampling produces valid tokens

**Prerequisites**: TASK-TALKER-002

---

### TASK-TALKER-005: Implement Codec Embeddings

**Description**: Embed codec tokens for autoregressive generation

**Acceptance Criteria**:
- [ ] 16 separate embedding tables (one per codebook)
- [ ] Handle codec tokens:
  - `<codec_bos>` (2149)
  - `<codec_eos>` (2150)
  - `<codec_pad>` (2148)
- [ ] Support thinking mode tokens:
  - `<codec_think_bos>` (2156)
  - `<codec_think_eos>` (2157)
  - `<codec_nothink>` (2155)

**Implementation Approach**: Multiple embedding lookup tables

**Files**:
- `src/models/qwen3omni-talker.cpp`

**Tests**:
- Embeddings load correctly
- Token lookup works for all 16 codebooks

**Prerequisites**: TASK-TALKER-002

---

### TASK-TALKER-006: Implement Voice Selection

**Description**: Support speaker ID conditioning for voice selection

**Acceptance Criteria**:
- [ ] Support speaker IDs:
  - Chelsie (2301)
  - Ethan (2302)
  - Aiden (2303)
- [ ] Pass speaker ID to Talker for conditioning
- [ ] Default to Chelsie if not specified

**Implementation Approach**: Embed speaker ID into generation context

**Files**:
- `src/models/qwen3omni-talker.cpp`

**Tests**:
- Different speaker IDs produce different outputs
- Voice characteristics match descriptions

**Prerequisites**: TASK-TALKER-004

---

### TASK-TALKER-007: Implement Dual-Model Inference Pipeline

**Description**: Connect Thinker output to Talker input

**Acceptance Criteria**:
- [ ] Extract hidden states from Thinker layer 18
- [ ] Pass `trailing_text_hidden` for coherence
- [ ] Talker runs after/in parallel with Thinker
- [ ] Handle KV cache for both models

**Implementation Approach**: New inference pipeline in context

**Files**:
- `src/llama-context.cpp`
- `tools/mtmd/mtmd.cpp`

**Tests**:
- End-to-end Thinker → Talker pipeline works
- Hidden state transfer is correct

**Prerequisites**: TASK-THINKER-004, TASK-TALKER-003

---

## Code Predictor Tasks

### TASK-CODEPR-001: Create Code Predictor Graph Builder

**Description**: Implement 5-layer transformer for codebook 2-16 prediction

**Acceptance Criteria**:
- [ ] 5 transformer layers
- [ ] Hidden size 1024, 16 attention heads, 8 KV heads
- [ ] FFN size 3072
- [ ] Vocab size 2048 per codebook

**Implementation Approach**: Small transformer, can be part of Talker file

**Files**:
- `src/models/qwen3omni-talker.cpp`

**Tests**:
- Forward pass produces correct output
- Compare against HuggingFace

**Prerequisites**: TASK-TALKER-002

---

### TASK-CODEPR-002: Implement Codebook Embedding Tables

**Description**: Create 15 embedding tables for codebooks 2-16 input

**Acceptance Criteria**:
- [ ] 15 separate embedding tables (2-16)
- [ ] Each: 2048 entries × 1024 dimensions
- [ ] Previous codebook embedding used as input

**Implementation Approach**: Multiple embedding lookups

**Files**:
- `src/models/qwen3omni-talker.cpp`

**Tests**:
- Embeddings load correctly
- Sequential prediction works

**Prerequisites**: TASK-CODEPR-001

---

### TASK-CODEPR-003: Implement Sequential Prediction Loop

**Description**: Generate codebooks 2-16 using previous predictions

**Acceptance Criteria**:
- [ ] For each codebook i (2-16):
  - Embed previous codebook (i-1)
  - Run through 5-layer transformer
  - Predict codebook i
- [ ] Output 16 complete codebook tokens per audio frame

**Implementation Approach**: Autoregressive loop within Code Predictor

**Files**:
- `src/models/qwen3omni-talker.cpp`

**Tests**:
- All 16 codebooks generated correctly
- Matches HuggingFace output

**Prerequisites**: TASK-CODEPR-002

---

### TASK-CODEPR-004: Implement Code Predictor LM Head

**Description**: Output projection for codebook token prediction

**Acceptance Criteria**:
- [ ] Linear projection: 1024 → 2048 (codebook vocab size)
- [ ] Shared across all codebook predictions

**Implementation Approach**: Single linear layer

**Files**:
- `src/models/qwen3omni-talker.cpp`

**Tests**:
- Logits match HuggingFace

**Prerequisites**: TASK-CODEPR-001

---

## Code2Wav Tasks

### TASK-C2W-001: Create Code2Wav Module Structure

**Description**: Set up Code2Wav as new module in mtmd

**Acceptance Criteria**:
- [ ] Create `tools/mtmd/code2wav.h` header
- [ ] Create `tools/mtmd/code2wav.cpp` implementation
- [ ] Add to CMakeLists.txt build

**Implementation Approach**: New module with clean API

**Files**:
- `tools/mtmd/code2wav.h` (new)
- `tools/mtmd/code2wav.cpp` (new)
- `tools/mtmd/CMakeLists.txt`

**Prerequisites**: None

---

### TASK-C2W-002: Implement Codebook Lookup

**Description**: Convert audio tokens to embeddings via codebook lookup

**Acceptance Criteria**:
- [ ] 16 codebooks × 2048 entries × 512 dimensions
- [ ] 1 semantic codebook × 4096 entries × 512 dimensions
- [ ] Sum embeddings across all 17 codebooks
- [ ] Handle `<codec_pad>` tokens

**Implementation Approach**: Embedding lookup tables with sum reduction

**Files**:
- `tools/mtmd/code2wav.cpp`

**Tests**:
- Lookup produces correct embeddings
- Sum matches HuggingFace

**Prerequisites**: TASK-C2W-001

---

### TASK-C2W-003: Implement Transformer Backbone

**Description**: 8-layer transformer with sliding window attention

**Acceptance Criteria**:
- [ ] 8 transformer layers
- [ ] Hidden size 1024, 16 attention heads
- [ ] Sliding window attention with window size 72
- [ ] LayerScale for stability

**Implementation Approach**: Standard transformer with window attention

**Files**:
- `tools/mtmd/code2wav.cpp`

**Tests**:
- Output matches HuggingFace
- Sliding window works correctly

**Prerequisites**: TASK-C2W-002

---

### TASK-C2W-004: Implement ConvTranspose1D Upsampling Stack

**Description**: Upsample transformer output by 480×

**Acceptance Criteria**:
- [ ] 4 ConvTranspose1D layers with strides [8, 5, 4, 3]
- [ ] Total upsampling: 8 × 5 × 4 × 3 = 480×
- [ ] Decoder dimension 1536
- [ ] At 50 tokens/sec → 24,000 samples/sec

**Implementation Approach**: Use `ggml_conv_transpose_1d`

**Files**:
- `tools/mtmd/code2wav.cpp`

**Tests**:
- Output length = input_length × 480
- Weights load correctly

**Prerequisites**: TASK-C2W-003

---

### TASK-C2W-005: Implement ConvNeXt Blocks

**Description**: Add ConvNeXt blocks with causal convolutions after each upsample

**Acceptance Criteria**:
- [ ] Depthwise convolution (causal, no future leakage)
- [ ] Pointwise convolution layers
- [ ] GELU activation
- [ ] Residual connections

**Implementation Approach**: Use `ggml_conv_1d_dw` with causal padding

**Files**:
- `tools/mtmd/code2wav.cpp`

**Tests**:
- Causality: no future leakage
- Output matches HuggingFace

**Prerequisites**: TASK-C2W-004

---

### TASK-C2W-006: Implement Output Projection

**Description**: Final projection to audio waveform

**Acceptance Criteria**:
- [ ] Project to single channel (mono)
- [ ] Output 24kHz sample rate
- [ ] Handle proper scaling

**Implementation Approach**: Linear projection to 1 dimension

**Files**:
- `tools/mtmd/code2wav.cpp`

**Tests**:
- Output is valid audio samples
- Sample rate is 24kHz

**Prerequisites**: TASK-C2W-005

---

### TASK-C2W-007: Add WAV/PCM Output Support

**Description**: Generate standard audio file formats

**Acceptance Criteria**:
- [ ] Output WAV format (24kHz mono PCM)
- [ ] Output raw PCM format
- [ ] Proper WAV header generation
- [ ] Handle chunked output for streaming

**Implementation Approach**: WAV header + PCM data

**Files**:
- `tools/mtmd/code2wav.cpp`

**Tests**:
- WAV files play correctly
- Headers are valid

**Prerequisites**: TASK-C2W-006

---

### TASK-C2W-008: Implement Streaming Chunked Processing

**Description**: Process audio tokens in chunks for streaming output

**Acceptance Criteria**:
- [ ] Start processing after first 50 audio tokens
- [ ] 100 audio codes per chunk
- [ ] First audio chunk latency < 100ms
- [ ] Sliding window handles chunk boundaries

**Implementation Approach**: Chunked processing with state management

**Files**:
- `tools/mtmd/code2wav.cpp`

**Tests**:
- Streaming output matches batch output
- Latency target met

**Prerequisites**: TASK-C2W-003, TASK-C2W-006

---

## Server API Tasks

### TASK-SERVER-001: Add Binary Streaming Support

**Description**: Enable HTTP binary streaming for audio output

**Acceptance Criteria**:
- [ ] Add `next_binary` callback to `server_http_res`
- [ ] Support `Transfer-Encoding: chunked` for binary
- [ ] Support `Content-Type: audio/wav`

**Implementation Approach**: Extend existing streaming infrastructure

**Files**:
- `tools/server/server-http.h`
- `tools/server/server-http.cpp`

**Tests**:
- Binary chunks stream correctly
- HTTP headers are correct

**Prerequisites**: None

---

### TASK-SERVER-002: Implement Audio Speech Endpoint

**Description**: Create `/v1/audio/speech` endpoint for TTS

**Acceptance Criteria**:
- [ ] Accept POST with JSON body:
  - `model`: model name
  - `input`: text to synthesize
  - `voice`: chelsie/ethan/aiden
  - `response_format`: wav/pcm
  - `stream`: boolean
- [ ] Return chunked audio in specified format
- [ ] Support streaming mode

**Implementation Approach**: New endpoint handler

**Files**:
- `tools/server/server-context.cpp`

**Tests**:
- Endpoint responds correctly
- Audio output is valid

**Prerequisites**: TASK-SERVER-001, TASK-TALKER-007, TASK-C2W-007

---

### TASK-SERVER-003: Add Dual Output Mode to Chat Completions

**Description**: Support text + audio output in chat completions

**Acceptance Criteria**:
- [ ] Accept `modalities: ["text", "audio"]` in request
- [ ] Accept `audio: { voice, format }` options
- [ ] Stream text deltas via SSE
- [ ] Stream base64 audio chunks via SSE

**Implementation Approach**: Extend existing chat completions handler

**Files**:
- `tools/server/server-context.cpp`

**Tests**:
- Both modalities stream correctly
- SSE format is valid

**Prerequisites**: TASK-SERVER-002

---

### TASK-SERVER-004: Add Talker Model Loading

**Description**: Load Talker GGUF file in server

**Acceptance Criteria**:
- [ ] Accept `--talker` command line option
- [ ] Load Talker model alongside Thinker
- [ ] Initialize dual-model inference pipeline

**Implementation Approach**: Extend model loading

**Files**:
- `tools/server/server.cpp`

**Tests**:
- Talker loads correctly
- Memory usage is as expected

**Prerequisites**: TASK-TALKER-007

---

### TASK-SERVER-005: Add Voice Selection Parameter

**Description**: Support voice selection in API requests

**Acceptance Criteria**:
- [ ] Accept `voice` parameter in speech endpoint
- [ ] Accept `audio.voice` in chat completions
- [ ] Validate voice names (chelsie, ethan, aiden)
- [ ] Default to chelsie if not specified

**Implementation Approach**: Parameter parsing and validation

**Files**:
- `tools/server/server-context.cpp`

**Tests**:
- All voice options work
- Invalid voices return error

**Prerequisites**: TASK-TALKER-006

---

### TASK-SERVER-006: Add Audio Response Format Handling

**Description**: Support multiple audio output formats

**Acceptance Criteria**:
- [ ] WAV format (default)
- [ ] Raw PCM format
- [ ] Set appropriate Content-Type headers

**Implementation Approach**: Format conversion at output

**Files**:
- `tools/server/server-context.cpp`

**Tests**:
- All formats produce valid output
- Headers are correct

**Prerequisites**: TASK-C2W-007

---

## CLI Tool Tasks

### TASK-CLI-001: Create Qwen3-Omni CLI Tool

**Description**: Command-line tool for omni inference

**Acceptance Criteria**:
- [ ] Accept `--model` for Thinker
- [ ] Accept `--mmproj` for vision/audio encoders
- [ ] Accept `--talker` for Talker + Code2Wav
- [ ] Accept `--voice` for speaker selection
- [ ] Accept `-o output.wav` for audio output
- [ ] Accept `-p` for text prompt
- [ ] Accept `--image`, `--audio`, `--video` for multimodal input

**Implementation Approach**: New CLI tool following existing patterns

**Files**:
- `tools/qwen3-omni/main.cpp` (new)
- `tools/qwen3-omni/CMakeLists.txt` (new)

**Tests**:
- CLI parses all arguments
- Help text is complete

**Prerequisites**: TASK-THINKER-006, TASK-TALKER-007, TASK-C2W-007

---

### TASK-CLI-002: Implement Multimodal Input Handling

**Description**: Load and process image/audio/video inputs

**Acceptance Criteria**:
- [ ] Load image files (JPEG, PNG)
- [ ] Load audio files (WAV, MP3)
- [ ] Load video files (MP4, WebM)
- [ ] Pass to appropriate encoder

**Implementation Approach**: Use existing mtmd input handling

**Files**:
- `tools/qwen3-omni/main.cpp`

**Tests**:
- All input formats load correctly
- Errors on invalid files

**Prerequisites**: TASK-CLI-001

---

### TASK-CLI-003: Implement Audio Output

**Description**: Write generated audio to file

**Acceptance Criteria**:
- [ ] Output WAV file at 24kHz mono
- [ ] Support `-o` output path
- [ ] Print progress during generation
- [ ] Handle Ctrl+C gracefully

**Implementation Approach**: Use Code2Wav output

**Files**:
- `tools/qwen3-omni/main.cpp`

**Tests**:
- Output file is valid WAV
- Progress reporting works

**Prerequisites**: TASK-C2W-007, TASK-CLI-001

---

### TASK-CLI-004: Add Interactive Mode

**Description**: Support interactive conversation with audio output

**Acceptance Criteria**:
- [ ] Enter interactive mode without `-p`
- [ ] Accept text input, output text + audio
- [ ] Support conversation history
- [ ] Exit on `/quit` or Ctrl+D

**Implementation Approach**: REPL loop with audio output

**Files**:
- `tools/qwen3-omni/main.cpp`

**Tests**:
- Interactive mode works
- History is maintained

**Prerequisites**: TASK-CLI-003

---

## Testing & Validation Tasks

### TASK-TEST-001: Create HuggingFace Comparison Test (Thinker)

**Description**: Validate Thinker output matches HuggingFace

**Acceptance Criteria**:
- [ ] Compare text logits within 1e-4 tolerance
- [ ] Test with text-only input
- [ ] Test with image input
- [ ] Test with audio input
- [ ] Test with video input

**Implementation Approach**: Python test script with HF reference

**Files**:
- `tests/compare_hf.py` (new)

**Prerequisites**: TASK-THINKER-006

---

### TASK-TEST-002: Create HuggingFace Comparison Test (Talker)

**Description**: Validate Talker output tokens match HuggingFace

**Acceptance Criteria**:
- [ ] Compare first codebook tokens
- [ ] Compare all 16 codebook outputs
- [ ] Test with multiple text inputs
- [ ] Test with different voices

**Implementation Approach**: Python test script with HF reference

**Files**:
- `tests/compare_talker.py` (new)

**Prerequisites**: TASK-TALKER-007

---

### TASK-TEST-003: Create Audio Quality Metrics Test

**Description**: Measure audio quality against reference

**Acceptance Criteria**:
- [ ] PESQ score > 3.5
- [ ] STOI score > 0.9
- [ ] SI-SDR > 15 dB
- [ ] Compare against HuggingFace audio output

**Implementation Approach**: Python test with audio metrics libraries

**Files**:
- `tests/audio_quality.py` (new)

**Prerequisites**: TASK-C2W-007

---

### TASK-TEST-004: Create End-to-End Integration Test

**Description**: Full pipeline test from input to audio output

**Acceptance Criteria**:
- [ ] Text → Audio works
- [ ] Image + Text → Audio works
- [ ] Audio + Text → Audio works
- [ ] Video + Text → Audio works

**Implementation Approach**: Shell script or pytest

**Files**:
- `tests/test_qwen3_omni_e2e.sh` (new)

**Prerequisites**: TASK-CLI-001

---

### TASK-TEST-005: Create Performance Benchmark

**Description**: Measure token generation and audio decode rates

**Acceptance Criteria**:
- [ ] Thinker: ~20ms/token
- [ ] Talker: ~10ms/16 codes
- [ ] Code2Wav: ~50ms/chunk (real-time factor > 1.0)
- [ ] First audio chunk: < 100ms

**Implementation Approach**: Benchmark script with timing

**Files**:
- `tests/bench_qwen3_omni.py` (new)

**Prerequisites**: TASK-CLI-001

---

### TASK-TEST-006: Create Quantization Quality Test

**Description**: Verify quantized models meet quality targets

**Acceptance Criteria**:
- [ ] Q4_K_M Thinker: Text quality acceptable
- [ ] Q8_0 Talker: Audio quality PESQ > 3.5
- [ ] F16 Code2Wav codebooks: Exact lookups

**Implementation Approach**: Quality metrics at each quant level

**Files**:
- `tests/test_quantization.py` (new)

**Prerequisites**: TASK-GGUF-008, TASK-TEST-003

---

### TASK-TEST-007: Create Server API Tests

**Description**: Test server endpoints

**Acceptance Criteria**:
- [ ] `/v1/audio/speech` returns valid audio
- [ ] `/v1/chat/completions` with audio modality works
- [ ] Streaming mode works correctly
- [ ] Error handling is correct

**Implementation Approach**: pytest with requests library

**Files**:
- `tools/server/tests/test_qwen3_omni.py` (new)

**Prerequisites**: TASK-SERVER-003

---

### TASK-TEST-008: Create Memory Usage Test

**Description**: Verify model fits in target memory budget

**Acceptance Criteria**:
- [ ] Quantized model ≤ 26GB VRAM
- [ ] Peak memory during inference acceptable
- [ ] Memory properly freed after inference

**Implementation Approach**: Memory profiling test

**Files**:
- `tests/test_memory.py` (new)

**Prerequisites**: TASK-GGUF-008, TASK-CLI-001

---

## Dependency Graph

```
GGUF Conversion (can start immediately):
  TASK-GGUF-001 ──┬── TASK-GGUF-002 ── TASK-GGUF-006
                  │
                  └── TASK-GGUF-004

  TASK-GGUF-003 ───── TASK-GGUF-007

  TASK-GGUF-005 (parallel)

  All GGUF tasks ──── TASK-GGUF-008

Thinker (after GGUF-008):
  TASK-THINKER-001 ── TASK-THINKER-002 ── TASK-THINKER-003
                                       ── TASK-THINKER-004
                                       ── TASK-THINKER-005

  TASK-THINKER-002 + AUDIO + VISION ── TASK-THINKER-006

Audio Encoder (can start immediately):
  TASK-AUDIO-001 ── TASK-AUDIO-002 ── TASK-AUDIO-003

  TASK-THINKER-003 ── TASK-AUDIO-004

Vision Encoder (can start immediately):
  TASK-VISION-001 ── TASK-VISION-002

  TASK-THINKER-003 ── TASK-VISION-003

Talker (after GGUF-002):
  TASK-TALKER-001 ── TASK-TALKER-002 ── TASK-TALKER-003 ── TASK-TALKER-007
                                     ── TASK-TALKER-004 ── TASK-TALKER-006
                                     ── TASK-TALKER-005

  TASK-TALKER-002 ── TASK-CODEPR-001 ── TASK-CODEPR-002 ── TASK-CODEPR-003
                                     ── TASK-CODEPR-004

Code2Wav (can start immediately):
  TASK-C2W-001 ── TASK-C2W-002 ── TASK-C2W-003 ── TASK-C2W-004 ── TASK-C2W-005 ── TASK-C2W-006 ── TASK-C2W-007
                               ── TASK-C2W-008

Server (after core models):
  TASK-SERVER-001 ── TASK-SERVER-002 ── TASK-SERVER-003
                  ── TASK-SERVER-004
                  ── TASK-SERVER-005
                  ── TASK-SERVER-006

CLI (after all core):
  TASK-CLI-001 ── TASK-CLI-002
              ── TASK-CLI-003 ── TASK-CLI-004

Testing (after respective components):
  TASK-THINKER-006 ── TASK-TEST-001
  TASK-TALKER-007 ── TASK-TEST-002
  TASK-C2W-007 ── TASK-TEST-003
  TASK-CLI-001 ── TASK-TEST-004, TASK-TEST-005, TASK-TEST-008
  TASK-SERVER-003 ── TASK-TEST-007
  TASK-GGUF-008 + TASK-TEST-003 ── TASK-TEST-006
```

---

## Critical Path

The following tasks are on the critical path to a minimal working implementation:

1. **GGUF Conversion**: TASK-GGUF-001 → TASK-GGUF-004 → TASK-GGUF-008
2. **Thinker Core**: TASK-THINKER-001 → TASK-THINKER-002 → TASK-THINKER-004
3. **Talker Core**: TASK-TALKER-001 → TASK-TALKER-002 → TASK-TALKER-004 → TASK-TALKER-007
4. **Code Predictor**: TASK-CODEPR-001 → TASK-CODEPR-003
5. **Code2Wav Core**: TASK-C2W-001 → TASK-C2W-002 → TASK-C2W-006 → TASK-C2W-007

Parallel work streams:
- Audio encoder (TASK-AUDIO-*)
- Vision encoder (TASK-VISION-*)
- Server API (TASK-SERVER-*)
- CLI tool (TASK-CLI-*)
