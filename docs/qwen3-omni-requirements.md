# Qwen3-Omni-30B-A3B Requirements Document

> Extracted from implementation specification for llama.cpp support
> **Source**: [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
> **Issue**: [ggml-org/llama.cpp#16186](https://github.com/ggml-org/llama.cpp/issues/16186)
> **Spec**: [qwen3-omni-implementation-plan.md](qwen3-omni-implementation-plan.md)

---

## Functional Requirements

### FR-1: Thinker Model (Main LLM)

#### FR-1.1: Architecture Support
- **Description**: Implement `LLM_ARCH_QWEN3OMNIMOE` architecture in llama.cpp
- **Acceptance Criteria**:
  - Load Qwen3-Omni Thinker weights from GGUF
  - Support 48-layer transformer with 128 experts (8 active per token)
  - Implement Q/K normalization (Qwen3 style)
  - Support GQA with 32 attention heads, 4 KV heads
- **Dependencies**: Extends existing `LLM_ARCH_QWEN3MOE`
- **Files**: `src/llama-arch.h`, `src/llama-arch.cpp`, `src/models/qwen3omni.cpp`

#### FR-1.2: TMRoPE (Time-aligned Multimodal RoPE)
- **Description**: Implement M-RoPE with sections [24, 20, 20] for temporal/spatial encoding
- **Acceptance Criteria**:
  - Section 1 (24 dims): Temporal position shared across modalities
  - Section 2 (20 dims): Spatial Y/Height for vision, 0 for text/audio
  - Section 3 (20 dims): Spatial X/Width for vision, 0 for text/audio
  - Audio/video frames synchronized at 25 tokens/second temporal rate
- **Dependencies**: Extends `ggml_rope_multi()` from Qwen2VL
- **Files**: `src/models/qwen3omni.cpp`

#### FR-1.3: Text Output Generation
- **Description**: Generate text tokens from multimodal input
- **Acceptance Criteria**:
  - Accept text, audio, image, and video inputs
  - Generate text responses via standard sampling
  - Support max 65,536 context positions
  - Match HuggingFace reference output within 1e-4 tolerance
- **Dependencies**: FR-1.1, FR-1.2, FR-2, FR-3

---

### FR-2: Audio Encoder

#### FR-2.1: Whisper-Style Encoder
- **Description**: Process audio input through 32-layer Whisper-style encoder
- **Acceptance Criteria**:
  - Accept 16kHz audio input
  - Generate 128 mel-bin spectrograms (not Whisper's 80)
  - Process through 32 transformer layers (d_model=1280, 20 heads)
  - Project output to 2048 dimensions for Thinker
- **Dependencies**: Extends `tools/mtmd/models/whisper-enc.cpp`
- **Files**: `tools/mtmd/mtmd-audio.cpp`, `tools/mtmd/models/qwen3omni-audio.cpp`

#### FR-2.2: Audio Token Handling
- **Description**: Handle special audio tokens in vocabulary
- **Acceptance Criteria**:
  - Recognize `<audio_start>` (151669), `<audio_end>` (151670)
  - Recognize `<audio>` (151675) placeholder token
  - Support TTS tokens: `<tts_bos>` (151672), `<tts_eos>` (151673), `<tts_pad>` (151671)
- **Dependencies**: FR-1.1
- **Files**: `src/llama-vocab.cpp`

---

### FR-3: Vision Encoder

#### FR-3.1: ViT with Deepstack
- **Description**: Process images through 27-layer ViT with deepstack features
- **Acceptance Criteria**:
  - Accept 768×768 images, 16×16 patches
  - Process through 27 transformer layers (hidden=1152, 16 heads)
  - Extract deepstack features at layers [8, 16, 24]
  - Support temporal patch size 2 for video frames
  - Project output to 2048 dimensions
- **Dependencies**: Extends `tools/mtmd/models/qwen3vl.cpp`
- **Files**: `tools/mtmd/models/qwen3omni-vision.cpp`

#### FR-3.2: Vision Token Handling
- **Description**: Handle special vision tokens
- **Acceptance Criteria**:
  - Recognize `<vision_start>` (151652), `<vision_end>` (151653)
  - Recognize `<image>` (151655), `<video>` (151656) tokens
  - Apply spatial merge factor of 2
- **Dependencies**: FR-1.1
- **Files**: `src/llama-vocab.cpp`

---

### FR-4: Talker Model (Audio Token Generation)

#### FR-4.1: Talker Architecture
- **Description**: Implement second MoE decoder for audio token generation
- **Acceptance Criteria**:
  - 20-layer transformer with 128 experts (6 active per token)
  - Hidden size 1024, 16 attention heads, 2 KV heads
  - Shared expert (768 FFN) + sparse experts (384 FFN each)
  - Accept hidden states from Thinker layer 18
  - Text projection MLP: 2048 → 1024 dimensions
- **Dependencies**: FR-1.1
- **Files**: `src/llama-arch.h`, `src/models/qwen3omni-talker.cpp`

#### FR-4.2: Codec Head
- **Description**: Predict first codebook token
- **Acceptance Criteria**:
  - Linear projection from 1024 to 3072 (Talker vocab size)
  - Output first codebook token per generation step
- **Dependencies**: FR-4.1
- **Files**: `src/models/qwen3omni-talker.cpp`

#### FR-4.3: Codec Embeddings
- **Description**: Embed codec tokens for autoregressive generation
- **Acceptance Criteria**:
  - 16 separate embedding tables for each codebook
  - Handle codec tokens: `<codec_bos>` (2149), `<codec_eos>` (2150), `<codec_pad>` (2148)
  - Support thinking mode: `<codec_think_bos>` (2156), `<codec_think_eos>` (2157)
- **Dependencies**: FR-4.1
- **Files**: `src/models/qwen3omni-talker.cpp`

#### FR-4.4: Dual-Model Inference Pipeline
- **Description**: Connect Thinker output to Talker input
- **Acceptance Criteria**:
  - Extract hidden states from Thinker layer 18 (not final output)
  - Pass `trailing_text_hidden` for coherence
  - Run Talker in parallel/after Thinker
- **Dependencies**: FR-1.1, FR-4.1
- **Files**: `src/llama-context.cpp`, `tools/mtmd/mtmd.cpp`

---

### FR-5: Code Predictor

#### FR-5.1: Multi-Codebook Prediction
- **Description**: Predict codebooks 2-16 sequentially
- **Acceptance Criteria**:
  - 5-layer transformer (hidden=1024, 16 heads, 8 KV heads)
  - Predict each codebook using previous predictions as input
  - Output 16 complete codebook tokens per audio frame
  - Vocab size 2048 per codebook
- **Dependencies**: FR-4.1
- **Files**: `src/models/qwen3omni-talker.cpp`

---

### FR-6: Code2Wav (Neural Audio Codec)

#### FR-6.1: Codebook Lookup
- **Description**: Convert audio tokens to embeddings
- **Acceptance Criteria**:
  - 16 codebooks × 2048 entries × 512 dimensions
  - 1 semantic codebook × 4096 entries
  - Sum embeddings across codebooks
- **Dependencies**: None
- **Files**: `tools/mtmd/code2wav.cpp`

#### FR-6.2: Transformer Backbone
- **Description**: Process codebook embeddings through transformer
- **Acceptance Criteria**:
  - 8-layer transformer, 1024 hidden, 16 attention heads
  - Sliding window attention with window size 72
  - LayerScale for stability
- **Dependencies**: FR-6.1
- **Files**: `tools/mtmd/code2wav.cpp`

#### FR-6.3: Upsampling Stack
- **Description**: Convert transformer output to audio waveform
- **Acceptance Criteria**:
  - ConvTranspose1D with strides [8, 5, 4, 3] = 480× total upsampling
  - ConvNeXt blocks with causal convolutions after each upsample
  - Decoder dimension 1536
  - Output 24kHz mono audio
- **Dependencies**: FR-6.2
- **Files**: `tools/mtmd/code2wav.cpp`

#### FR-6.4: Waveform Output
- **Description**: Generate final audio waveform
- **Acceptance Criteria**:
  - Output 24kHz mono PCM audio
  - At 50 tokens/sec input → 24,000 samples/sec output
  - Support WAV, PCM, MP3 output formats
- **Dependencies**: FR-6.3
- **Files**: `tools/mtmd/code2wav.cpp`

---

### FR-7: Voice Selection

#### FR-7.1: Speaker ID Support
- **Description**: Support multiple voice options
- **Acceptance Criteria**:
  - Chelsie (ID 2301): Female, honeyed, velvety, gentle
  - Ethan (ID 2302): Male, bright, upbeat, warm
  - Aiden (ID 2303): Male, warm, laid-back American
  - Pass speaker ID to Talker for voice conditioning
- **Dependencies**: FR-4.1
- **Files**: `src/models/qwen3omni-talker.cpp`, CLI and server interfaces

---

### FR-8: GGUF Conversion

#### FR-8.1: Three-File Architecture
- **Description**: Convert HuggingFace model to three GGUF files
- **Acceptance Criteria**:
  - `qwen3-omni-*.gguf`: Thinker (main LLM)
  - `mmproj-qwen3-omni-*.gguf`: Vision + Audio encoders
  - `talker-qwen3-omni-*.gguf`: Talker + Code Predictor + Code2Wav
- **Dependencies**: All FR-1 through FR-6
- **Files**: `convert_hf_to_gguf.py`, `gguf-py/gguf/constants.py`, `gguf-py/gguf/tensor_mapping.py`

#### FR-8.2: Tensor Naming Convention
- **Description**: Map HuggingFace tensor names to GGUF
- **Acceptance Criteria**:
  - Thinker: `model.layers.{i}.self_attn.*`, `model.layers.{i}.mlp.experts.{j}.*`
  - Talker: `talker.model.layers.{i}.*`, `talker.codec_head.*`, `talker.codec_embeddings.{0-15}.*`
  - Code2Wav: `code2wav.codebook.{0-15}.*`, `code2wav.transformer.*`, `code2wav.upsample.*`
- **Dependencies**: FR-8.1
- **Files**: `gguf-py/gguf/tensor_mapping.py`

#### FR-8.3: New Metadata Keys
- **Description**: Add GGUF metadata for Talker and Code2Wav
- **Acceptance Criteria**:
  - Talker: `talker.hidden_size`, `talker.block_count`, `talker.expert_count`, `talker.accept_hidden_layer`
  - Code2Wav: `code2wav.codebook_size`, `code2wav.codebook_dim`, `code2wav.num_quantizers`, `code2wav.upsample_rates`
- **Dependencies**: FR-8.1
- **Files**: `gguf-py/gguf/constants.py`

---

### FR-9: Server API

#### FR-9.1: Audio Speech Endpoint
- **Description**: New `/v1/audio/speech` endpoint for TTS
- **Acceptance Criteria**:
  - Accept: `model`, `input`, `voice`, `response_format`, `stream`
  - Return chunked WAV/MP3/PCM audio
  - Support streaming mode with chunked transfer
- **Dependencies**: FR-4, FR-5, FR-6
- **Files**: `tools/server/server-context.cpp`, `tools/server/server-http.cpp`

#### FR-9.2: Dual Output Mode
- **Description**: Chat completions with text + audio output
- **Acceptance Criteria**:
  - Accept `modalities: ["text", "audio"]` in request
  - Accept `audio: { voice, format }` options
  - Stream both text deltas and base64 audio chunks via SSE
- **Dependencies**: FR-9.1
- **Files**: `tools/server/server-context.cpp`

#### FR-9.3: Binary Streaming
- **Description**: HTTP binary streaming support
- **Acceptance Criteria**:
  - Add `next_binary` callback to `server_http_res`
  - Stream audio chunks via chunked transfer encoding
  - Support `Content-Type: audio/wav`
- **Dependencies**: None
- **Files**: `tools/server/server-http.h`, `tools/server/server-http.cpp`

---

### FR-10: CLI Tool

#### FR-10.1: Qwen3-Omni CLI
- **Description**: Command-line tool for omni inference
- **Acceptance Criteria**:
  - Accept `--mmproj` for vision/audio encoders
  - Accept `--talker` for Talker + Code2Wav
  - Accept `--voice` for speaker selection
  - Accept `-o output.wav` for audio output
  - Support multimodal input (text, image, audio, video)
- **Dependencies**: All FR-1 through FR-7
- **Files**: `tools/qwen3-omni/main.cpp`

---

## Non-Functional Requirements

### NFR-1: Performance

#### NFR-1.1: First Audio Chunk Latency
- **Description**: Time to first audio output chunk
- **Target**: < 100ms from generation start
- **Strategy**: Start Code2Wav after first 50 audio tokens
- **Measurement**: Time from Talker first token to first audio chunk

#### NFR-1.2: Token Generation Rate
- **Description**: Audio token generation speed
- **Target**: ~20ms/token for Thinker, ~10ms/16 codes for Talker
- **Measurement**: Tokens per second benchmark

#### NFR-1.3: Code2Wav Decode Rate
- **Description**: Audio decoding throughput
- **Target**: ~50ms per chunk (480 samples = 20ms audio)
- **Measurement**: Real-time factor (should be > 1.0)

#### NFR-1.4: Streaming Chunk Size
- **Description**: Audio vocalization interval
- **Target**: 100 audio codes per chunk
- **Rationale**: Balance between latency and decode efficiency

---

### NFR-2: Memory

#### NFR-2.1: Quantized Model Size
- **Description**: Total VRAM required for inference
- **Target**: ≤ 26GB for full quantized model
- **Breakdown**:
  - Thinker: ~18GB (Q4_K_M)
  - mmproj: ~3GB (F16)
  - Talker: ~4GB (Q8_0)
  - Code2Wav: ~0.8GB (mostly F16)

#### NFR-2.2: F16 Reference Size
- **Description**: Unquantized model size
- **Value**: ~82GB total
- **Breakdown**: Thinker ~70GB, mmproj ~3GB, Talker ~8GB, Code2Wav ~1GB

---

### NFR-3: Quantization Constraints

#### NFR-3.1: Thinker Quantization
- **Supported**: Q4_K_M (default), Q5_K_M, Q8_0, F16
- **Notes**: Standard llama.cpp quantization works well

#### NFR-3.2: Talker Quantization
- **Supported**: F16 (best), Q8_0 (acceptable), Q6_K (slight quality loss)
- **Not Recommended**: Q4_K_M (noticeable quality degradation)
- **Constraints**:
  - Keep `mlp.gate` tensors at F16
  - Keep `codec_embeddings` tensors at F16
- **Reason**: Quantization noise compounds through 16 codebook predictions

#### NFR-3.3: Code2Wav Quantization
- **Constraints**:
  - Codebook embeddings: **Must be F16** (lookup table, must be exact)
  - ConvTranspose weights: F16 (sensitive to quantization)
  - Transformer layers: Q8_0 OK
  - ConvNeXt blocks: Q8_0 OK

---

### NFR-4: Audio Quality

#### NFR-4.1: PESQ Score
- **Description**: Perceptual speech quality
- **Target**: > 3.5 for Q8_0 quantization
- **Minimum**: > 2.5 (reject if below)

#### NFR-4.2: STOI Score
- **Description**: Speech intelligibility
- **Target**: > 0.9

#### NFR-4.3: SI-SDR
- **Description**: Scale-invariant signal-to-distortion ratio
- **Target**: > 15 dB

#### NFR-4.4: Sample Rate
- **Output**: 24kHz mono PCM
- **Constraint**: Must match original model output

---

### NFR-5: Accuracy

#### NFR-5.1: HuggingFace Parity (Text)
- **Description**: Text output matches reference implementation
- **Target**: ≤ 1e-4 numerical tolerance
- **Verification**: `tests/compare_hf.py`

#### NFR-5.2: HuggingFace Parity (Audio Tokens)
- **Description**: Talker output tokens match reference
- **Verification**: `tests/compare_talker.py`

#### NFR-5.3: Audio Waveform Parity
- **Description**: Code2Wav output matches reference
- **Verification**: `tests/audio_quality.py` with PESQ/STOI/SI-SDR

---

### NFR-6: Backend Support

#### NFR-6.1: Required ggml Operations
- All required ops exist: `ggml_conv_transpose_1d`, `ggml_conv_1d_dw`, `ggml_pad`, `ggml_pool_1d`, `ggml_win_part`, `ggml_win_unpart`, `ggml_flash_attn_ext`

#### NFR-6.2: Backend Compatibility
| Operation | CPU | CUDA | Metal | Vulkan |
|-----------|:---:|:----:|:-----:|:------:|
| conv_transpose_1d | ✅ | ✅ | ✅ | ✅ |
| flash_attn_ext | ✅ | ✅ | ✅ | ⚠️ partial |

---

### NFR-7: Maintainability

#### NFR-7.1: Code Reuse
- **Target**: Maximize reuse of existing components
- **Thinker**: 80% reuse from Qwen3MOE
- **Audio Encoder**: 90% reuse from Whisper
- **Vision Encoder**: 95% reuse from Qwen3VL
- **TMRoPE**: 70% reuse from M-RoPE

#### NFR-7.2: Estimated New Code
- **Total**: ~6,000-8,000 lines
- **Breakdown**:
  - Thinker: ~500 LOC
  - Audio Encoder: ~200 LOC
  - Vision Encoder: ~200 LOC
  - TMRoPE: ~100 LOC
  - Talker: ~1500 LOC
  - Code Predictor: ~500 LOC
  - Code2Wav: ~2000 LOC
  - GGUF Converter: ~800 LOC
  - Dual-model API: ~500 LOC

---

## Requirement Dependencies

### Prerequisites Graph

```
FR-1.1 (Thinker Arch)
   ├── FR-1.2 (TMRoPE)
   ├── FR-2.1 (Audio Encoder) ──┐
   ├── FR-3.1 (Vision Encoder) ─┤
   │                            ├── FR-1.3 (Text Output)
   └── FR-4.1 (Talker Arch) ────┘
          │
          ├── FR-4.2 (Codec Head)
          ├── FR-4.3 (Codec Embeddings)
          ├── FR-4.4 (Dual-Model Pipeline)
          │
          └── FR-5.1 (Code Predictor)
                 │
                 └── FR-6.1 (Codebook Lookup)
                        │
                        ├── FR-6.2 (Transformer)
                        │      │
                        │      └── FR-6.3 (Upsampling)
                        │             │
                        │             └── FR-6.4 (Waveform Output)
                        │
                        └── FR-7.1 (Voice Selection)

FR-8.1 (GGUF 3-file) ← All FR-1 through FR-6
   ├── FR-8.2 (Tensor Naming)
   └── FR-8.3 (Metadata Keys)

FR-9.1 (Audio API) ← FR-4, FR-5, FR-6
   ├── FR-9.2 (Dual Output)
   └── FR-9.3 (Binary Streaming)

FR-10.1 (CLI) ← All FR-1 through FR-7
```

### Implementation Phases

#### Phase 1: Multimodal Input (Text Output Only)
**Requirements**: FR-1.1, FR-1.2, FR-1.3, FR-2.1, FR-2.2, FR-3.1, FR-3.2, FR-8.1 (partial)
**Outcome**: Qwen3-Omni works as text-only output model

#### Phase 2: Talker (Audio Token Generation)
**Requirements**: FR-4.1, FR-4.2, FR-4.3, FR-4.4, FR-5.1, FR-7.1
**Prerequisites**: Phase 1
**Outcome**: Generate 16-codebook audio token sequences

#### Phase 3: Code2Wav (Waveform Generation)
**Requirements**: FR-6.1, FR-6.2, FR-6.3, FR-6.4
**Prerequisites**: Phase 2
**Outcome**: Convert audio tokens to 24kHz waveforms

#### Phase 4: Integration & Streaming
**Requirements**: FR-8.1, FR-8.2, FR-8.3, FR-9.1, FR-9.2, FR-9.3, FR-10.1
**Prerequisites**: Phase 3
**Outcome**: Full production-ready implementation

---

## Technical Constraints

### Architecture Constraints

1. **Three-GGUF Architecture**: Model split into Thinker, mmproj, Talker files for memory efficiency
2. **Layer 18 Extraction**: Talker must receive hidden states from Thinker layer 18, not final output
3. **Sequential Codebook Prediction**: Code Predictor generates codebooks 2-16 in order, each using previous predictions
4. **Causal Convolutions**: Code2Wav ConvNeXt blocks must be causal (no future leakage)
5. **Sliding Window Size**: Code2Wav transformer uses window size 72 for efficient long-sequence processing

### Compatibility Constraints

1. **Existing APIs**: Must work with existing llama.cpp context and sampling APIs
2. **GGUF Format**: Follow established tensor naming and metadata conventions
3. **Backend Neutral**: Core implementation must work across CPU/CUDA/Metal/Vulkan

### Quality Constraints

1. **Codebook Precision**: 16 quantizer codebooks must remain at F16 for audio quality
2. **Audio Output Format**: Must output 24kHz mono PCM (matching original model)
3. **Token Vocabulary**: Must use exact special token IDs from HuggingFace model

---

## References

- [HuggingFace Model](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
- [GitHub Issue #16186](https://github.com/ggml-org/llama.cpp/issues/16186)
- [Transformers PR #41025](https://github.com/huggingface/transformers/pull/41025)
- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215)
- [SNAC: Multi-Scale Neural Audio Codec](https://arxiv.org/abs/2410.14411)
