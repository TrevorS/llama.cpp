# Qwen3-Omni-30B-A3B Implementation Plan

> Implementation specification for adding Qwen3-Omni-30B-A3B support to llama.cpp.
>
> **Model**: [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
> **Issue**: [ggml-org/llama.cpp#16186](https://github.com/ggml-org/llama.cpp/issues/16186)

## Executive Summary

Qwen3-Omni is a **Thinker-Talker architecture** multimodal model:
- **Input**: Text, Audio, Image, Video
- **Output**: Text AND Speech (24kHz audio waveforms)

**Current llama.cpp status**: Partial support exists. The Thinker's text backbone (Qwen3MOE) and audio/vision encoders are close to existing implementations. The major gaps are the **Talker** (speech synthesis) and **Code2Wav** (neural audio codec).

## Model Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Qwen3-Omni-30B-A3B                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INPUTS                           OUTPUTS                           │
│  ├─ Text                          ├─ Text tokens                    │
│  ├─ Audio (speech)                └─ Audio waveform (24kHz)         │
│  ├─ Images                                                          │
│  └─ Video                                                           │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │Audio Encoder│  │Vision Encoder│ │    Thinker (Qwen3MOE)       │  │
│  │ (Whisper)   │  │ (ViT-27L)   │  │    48 layers, 128 experts   │  │
│  │ 32 layers   │  │ Deepstack   │  │    Text reasoning + output  │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────────┬──────────────┘  │
│         │                │                        │                  │
│         └────────────────┴────────────────────────┤                  │
│                    Embeddings                     │ Hidden states    │
│                    (to Thinker)                   ▼                  │
│                                          ┌───────────────┐           │
│                                          │    Talker     │           │
│                                          │   20 layers   │           │
│                                          │  128 experts  │           │
│                                          └───────┬───────┘           │
│                                                  │ Audio tokens      │
│                                                  ▼                   │
│                                          ┌───────────────┐           │
│                                          │   Code2Wav    │           │
│                                          │ Neural Codec  │           │
│                                          │ 16 quantizers │           │
│                                          └───────┬───────┘           │
│                                                  │                   │
│                                                  ▼                   │
│                                          Audio Waveform (24kHz)      │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Specifications

### 1. Thinker (Main LLM) - `Qwen3OmniMoeThinker`

| Parameter | Value |
|-----------|-------|
| Architecture | Qwen3MOE with M-RoPE |
| Hidden size | 2048 |
| Layers | 48 |
| Attention heads | 32 |
| KV heads | 4 (GQA) |
| Head dim | 128 |
| Experts | 128 total, 8 active per token |
| MoE FFN size | 768 |
| Vocab size | 152,064 |
| Max positions | 65,536 |
| RoPE theta | 1,000,000 |
| M-RoPE sections | [24, 20, 20] |
| Q/K norm | Yes |
| RMS norm eps | 1e-6 |

**llama.cpp status**: `LLM_ARCH_QWEN3MOE` exists in `src/models/qwen3moe.cpp`. Needs M-RoPE sections `[24,20,20]` for multimodal.

### 2. Audio Encoder - `Qwen3OmniMoeAudioEncoder`

| Parameter | Value |
|-----------|-------|
| Type | Whisper-style transformer |
| d_model | 1280 |
| Layers | 32 |
| Attention heads | 20 |
| FFN dim | 5120 |
| Mel bins | 128 |
| Output dim | 2048 |
| Conv chunk size | 500 |
| Max source positions | 1500 |
| Sample rate | 16,000 Hz |

**llama.cpp status**: Whisper encoder exists in `tools/mtmd/models/whisper-enc.cpp`. Needs adaptation for 128 mel bins (vs Whisper's 80) and projection to 2048 dims.

### 3. Vision Encoder - `Qwen3OmniMoeVisionEncoder`

| Parameter | Value |
|-----------|-------|
| Image size | 768 |
| Patch size | 16×16 |
| Temporal patch | 2 (for video) |
| Hidden size | 1152 |
| Depth | 27 layers |
| Attention heads | 16 |
| FFN size | 4304 |
| Output dim | 2048 |
| Deepstack layers | [8, 16, 24] |
| Spatial merge | 2 |
| Activation | GELU |

**llama.cpp status**: Similar to Qwen3VL in `tools/mtmd/models/qwen3vl.cpp`. Has deepstack support.

### 4. Talker - `Qwen3OmniMoeTalker`

| Parameter | Value |
|-----------|-------|
| Hidden size | 1024 |
| Layers | 20 |
| Attention heads | 16 |
| KV heads | 2 |
| Head dim | 128 |
| Experts | 128 total, 6 active per token |
| MoE FFN size | 384 |
| Shared expert FFN | 768 |
| Vocab size | 3,072 (audio tokens) |
| Max positions | 65,536 |
| M-RoPE sections | [24, 20, 20] |
| Accept hidden layer | 18 |

**Purpose**: Takes Thinker's layer-18 hidden states and generates audio token sequences.

**llama.cpp status**: Does not exist. New model type required.

### 5. Code2Wav (Neural Audio Codec)

| Parameter | Value |
|-----------|-------|
| Hidden size | 1024 |
| Transformer layers | 8 |
| Attention heads | 16 |
| Decoder dim | 1536 |
| Codebook size | 2048 |
| Codebook dim | 512 |
| Num quantizers | 16 |
| Semantic quantizers | 1 |
| Semantic codebook | 4096 |
| Upsample rates | [8, 5, 4, 3] = 480× |
| Sliding window | 72 |
| Output sample rate | 24,000 Hz |

**Purpose**: Converts audio tokens to waveform using VQ-VAE style codec (similar to [SNAC](https://arxiv.org/abs/2410.14411)).

**llama.cpp status**: Does not exist. Completely new subsystem.

### 6. Code Predictor

| Parameter | Value |
|-----------|-------|
| Hidden size | 1024 |
| Layers | 5 |
| Attention heads | 16 |
| KV heads | 8 |
| FFN size | 3,072 |
| Vocab size | 2,048 |
| Code groups | 16 |

**Purpose**: Predicts codebooks 2-16 sequentially from the first codebook.

**llama.cpp status**: Does not exist.

## Special Token IDs

### Audio Tokens (Thinker vocabulary)
| Token | ID | Purpose |
|-------|-----|---------|
| `<audio_start>` | 151669 | Start of audio input |
| `<audio_end>` | 151670 | End of audio input |
| `<audio>` | 151675 | Audio placeholder |
| `<tts_bos>` | 151672 | TTS begin of sequence |
| `<tts_eos>` | 151673 | TTS end of sequence |
| `<tts_pad>` | 151671 | TTS padding |

### Codec Tokens (Talker vocabulary)
| Token | ID | Purpose |
|-------|-----|---------|
| `<codec_bos>` | 2149 | Codec begin of sequence |
| `<codec_eos>` | 2150 | Codec end of sequence |
| `<codec_pad>` | 2148 | Codec padding |
| `<codec_nothink>` | 2155 | No thinking mode |
| `<codec_think_bos>` | 2156 | Thinking begin |
| `<codec_think_eos>` | 2157 | Thinking end |

### Vision Tokens
| Token | ID |
|-------|-----|
| `<vision_start>` | 151652 |
| `<vision_end>` | 151653 |
| `<image>` | 151655 |
| `<video>` | 151656 |

### Speaker IDs (for voice selection)
| Speaker | ID | Voice Description |
|---------|-----|-------------------|
| Chelsie | 2301 | Female, honeyed, velvety, gentle |
| Ethan | 2302 | Male, bright, upbeat, warm |
| Aiden | 2303 | Male, warm, laid-back American |

## TMRoPE (Time-aligned Multimodal RoPE)

The model uses M-RoPE with 3 sections `[24, 20, 20]` to encode:

```
Dimension allocation: [24, 20, 20] = 64 total (head_dim / 2)

Section 1 (24 dims): Temporal position
  - Shared across all modalities
  - Audio/video frames aligned by timestamp
  - Text uses sequential positions

Section 2 (20 dims): Spatial Y / Height
  - Vision: patch row position
  - Audio: 0 (unused)
  - Text: 0 (unused)

Section 3 (20 dims): Spatial X / Width
  - Vision: patch column position
  - Audio: 0 (unused)
  - Text: 0 (unused)
```

**Position calculation**:
```python
position_id_per_seconds = 25  # Tokens per second for temporal alignment

# Audio tokens
audio_positions = (frame_idx / fps) * position_id_per_seconds

# Vision tokens (3D)
temporal = frame_idx * position_id_per_seconds
height = patch_row
width = patch_col

# Example: Video 30fps, frame 15 → temporal_pos = (15/30) * 25 = 12
#          Audio at 0.5s mark   → temporal_pos = 0.5 * 25 = 12
# Both share temporal_pos = 12, enabling cross-modal attention
```

## Gap Analysis

### What Exists in llama.cpp

| Component | Status | Location |
|-----------|--------|----------|
| Qwen3MOE text model | ✅ Ready | `src/models/qwen3moe.cpp` |
| Whisper audio encoder | ✅ Ready | `tools/mtmd/models/whisper-enc.cpp` |
| Mel spectrogram preprocessing | ✅ Ready | `tools/mtmd/mtmd-audio.cpp` |
| Qwen3VL vision encoder | ✅ Ready | `tools/mtmd/models/qwen3vl.cpp` |
| Deepstack vision features | ✅ Ready | `tools/mtmd/models/qwen3vl.cpp` |
| M-RoPE implementation | ✅ Ready | `src/models/qwen2vl.cpp` |
| MoE expert routing | ✅ Ready | `src/llama-graph.cpp` |
| Multimodal embedding merge | ✅ Ready | `tools/mtmd/mtmd.cpp` |

### What's Missing

| Component | Effort | Notes |
|-----------|--------|-------|
| **Talker decoder** | Large | New model type, runs after Thinker |
| **Code2Wav codec** | Large | VQ-VAE decoder, upsampling, waveform generation |
| **Code Predictor** | Medium | Multi-codebook prediction head |
| **GGUF conversion** | Medium | New tensor mappings for all components |
| **Dual-model inference** | Medium | Thinker → Talker pipeline |
| **Audio output API** | Medium | Return waveforms from inference |
| **TMRoPE alignment** | Small | Extend M-RoPE for audio timestamps |

## Implementation Phases

### Phase 1: Text + Audio/Vision Input (Thinker Only)

**Goal**: Run Qwen3-Omni as a text-only output model (like "Thinking" variant)

1. Add `LLM_ARCH_QWEN3OMNIMOE` architecture to `src/llama-arch.h`
2. Create graph builder in `src/models/qwen3omni.cpp`
3. Extend audio encoder for 128 mel bins in `tools/mtmd/mtmd-audio.cpp`
4. Add TMRoPE position embedding alignment
5. Create GGUF converter for thinker + encoders
6. Wire up in mtmd for multimodal input

**Output**: Text responses to audio/vision/text input

### Phase 2: Talker (Audio Output)

**Goal**: Generate audio tokens from Thinker hidden states

1. Add `LLM_ARCH_QWEN3OMNI_TALKER` architecture
2. Create graph builder in `src/models/qwen3omni-talker.cpp`
3. Implement dual-model inference pipeline (Thinker layer-18 → Talker)
4. Add audio token vocabulary handling
5. Create Code Predictor for codebooks 2-16

**Output**: 16-codebook audio token sequences

### Phase 3: Code2Wav (Waveform Generation)

**Goal**: Convert audio tokens to actual waveforms

1. Implement VQ-VAE decoder in `tools/mtmd/code2wav.cpp`
2. Add upsampling layers (ConvTranspose1D stack)
3. Implement codebook lookup
4. Add ConvNeXt blocks with causal convolution
5. Add waveform output API

**Output**: 24kHz mono audio waveforms

### Phase 4: Streaming & Optimization

**Goal**: Real-time voice interaction

1. Implement sliding window attention for Code2Wav
2. Add chunked audio output streaming
3. Optimize for low latency (<100ms first audio chunk)
4. Add voice selection API (Chelsie/Ethan/Aiden)

## GGUF File Structure

For Qwen3-Omni, we need **three GGUF files**:

```
qwen3-omni-30b-a3b-Q4_K_M.gguf          # Thinker (main LLM)
mmproj-qwen3-omni-30b-a3b-F16.gguf       # Vision + Audio encoders
talker-qwen3-omni-30b-a3b-Q8_0.gguf      # Talker + Code Predictor + Code2Wav
```

### Tensor Naming Convention

**Thinker (Text Model)**:
```
model.embed_tokens.weight
model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
model.layers.{i}.self_attn.{q,k}_norm.weight          # Qwen3 Q/K norm
model.layers.{i}.mlp.experts.{j}.{gate,up,down}_proj.weight
model.layers.{i}.mlp.shared_expert.{gate,up,down}_proj.weight
model.layers.{i}.mlp.gate.weight
model.layers.{i}.{input,post_attention}_layernorm.weight
model.norm.weight
lm_head.weight
```

**Vision Encoder (mmproj)**:
```
v.patch_embd.weight
v.patch_embd.weight.1                    # Temporal patch (for video)
v.position_embd.weight
v.blk.{bid}.attn_{q,k,v}.weight
v.blk.{bid}.attn_out.weight
v.blk.{bid}.ln1.weight
v.blk.{bid}.ffn_{up,down}.weight
v.deepstack.{bid}.{norm,fc1,fc2}.weight  # Qwen3VL deepstack
mm.{bid}.weight                          # Projector layers
```

**Audio Encoder (mmproj)**:
```
audio_tower.conv1.weight
audio_tower.conv2.weight
audio_tower.embed_positions.weight
audio_tower.layers.{bid}.self_attn.{q,k,v,out}_proj.weight
audio_tower.layers.{bid}.{self_attn,final}_layer_norm.weight
audio_tower.layers.{bid}.fc{1,2}.weight
audio.multi_modal_projector.linear_{bid}.weight
```

**Talker**:
```
talker.text_projection.weight
talker.text_projection.bias
talker.model.embed_tokens.weight
talker.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
talker.model.layers.{i}.mlp.experts.{j}.{gate,up,down}_proj.weight
talker.model.layers.{i}.mlp.shared_expert.{gate,up,down}_proj.weight
talker.model.layers.{i}.{input,post_attention}_layernorm.weight
talker.codec_head.weight
talker.codec_embeddings.{0-15}.weight
```

**Code Predictor**:
```
code_predictor.model.embed_tokens.{0-14}.weight
code_predictor.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
code_predictor.model.layers.{i}.mlp.{gate,up,down}_proj.weight
code_predictor.lm_head.weight
```

**Code2Wav**:
```
code2wav.codebook.{0-15}.weight           # 16 quantizer codebooks (2048×512)
code2wav.semantic_codebook.weight          # Semantic codebook (4096×512)
code2wav.transformer.layers.{i}.self_attn.{q,k,v,o}_proj.weight
code2wav.transformer.layers.{i}.mlp.{fc1,fc2}.weight
code2wav.transformer.layers.{i}.layer_scale
code2wav.upsample.{0-3}.conv.weight        # ConvTranspose1d
code2wav.upsample.{0-3}.conv.bias
code2wav.convnext.{i}.depthwise_conv.weight
code2wav.convnext.{i}.pointwise_conv.{1,2}.weight
code2wav.output_proj.weight
```

## Talker Architecture Details

The Talker converts Thinker hidden states into audio tokens:

```
Thinker Layer 18 Hidden States (2048-dim)
         │
         ▼
┌─────────────────────────────┐
│    Text Projection MLP      │  2048 → 1024 dims
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│    Talker Transformer       │  20 layers, MoE (128 experts, 6 active)
│    + Codec Embeddings       │  Hidden size: 1024
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│    Codec Head               │  Linear(1024 → 3072) for first codebook
└─────────────────────────────┘
         │
         ▼
   First Codebook Token
         │
         ▼
┌─────────────────────────────┐
│    Code Predictor           │  5 layers, predicts remaining 15 codebooks
│    Sequential generation    │  Uses previous codebook as input
└─────────────────────────────┘
         │
         ▼
   16 Codebook Tokens (complete)
```

**Key details**:
- Talker extracts from Thinker layer 18 (not final output)
- `trailing_text_hidden` provides context for coherence with text
- MoE has shared expert (768 FFN) always active + 6 sparse experts (384 FFN each)

## Code2Wav Architecture Details

Code2Wav converts discrete audio tokens to waveforms:

```
16 Codebook Tokens
         │
         ▼
┌─────────────────────────────┐
│   Codebook Embeddings       │  16 codebooks × 2048 entries × 512 dims
│   + Semantic Codebook       │  1 semantic × 4096 entries
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   Transformer Backbone      │  8 layers, 1024 hidden
│   Sliding Window Attention  │  Window size: 72
│   + LayerScale              │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   Upsampling Decoder        │  ConvTranspose1D stack
│   Rates: [8, 5, 4, 3]       │  Total: 480× upsampling
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│   ConvNeXt Blocks           │  Causal convolutions
│   Decoder dim: 1536         │
└─────────────────────────────┘
         │
         ▼
   24kHz Audio Waveform
```

**Upsampling math**: 8 × 5 × 4 × 3 = 480× upsampling. At 50 tokens/sec input → 24,000 samples/sec output.

## ggml Operations Analysis

All primitives needed for Code2Wav exist in ggml:

| Operation | ggml Function | Status |
|-----------|---------------|--------|
| Transposed Conv1D | `ggml_conv_transpose_1d` | ✅ Available |
| Upsampling | `ggml_upscale`, `ggml_interpolate` | ✅ Available |
| Depthwise Conv1D | `ggml_conv_1d_dw` | ✅ Available |
| Padding | `ggml_pad`, `ggml_pad_reflect_1d` | ✅ Available |
| Pooling | `ggml_pool_1d` (max/avg) | ✅ Available |
| Window Attention | `ggml_win_part`, `ggml_win_unpart` | ✅ Available |
| Flash Attention | `ggml_flash_attn_ext` | ✅ Available |

**Backend support**: All operations work on CPU, CUDA, Metal. Vulkan has partial flash attention support.

## Streaming Audio Output

### Server Changes Required

**1. Binary streaming support** in `tools/server/server-http.h`:
```cpp
struct server_http_res {
    // Existing...
    std::function<bool(std::string &)> next = nullptr;

    // NEW: Binary streaming
    std::function<bool(std::vector<uint8_t> &)> next_binary = nullptr;
};
```

**2. New API endpoint** `/v1/audio/speech`:
```
POST /v1/audio/speech
{
  "model": "qwen3-omni",
  "input": "Hello, how can I help you today?",
  "voice": "ethan",           // chelsie, ethan, aiden
  "response_format": "wav",   // wav, mp3, pcm
  "stream": true
}

Response:
Content-Type: audio/wav
Transfer-Encoding: chunked
[WAV header + PCM chunks...]
```

**3. Dual output mode** for chat completions with audio:
```
POST /v1/chat/completions
{
  "model": "qwen3-omni",
  "messages": [...],
  "modalities": ["text", "audio"],
  "audio": { "voice": "chelsie", "format": "wav" },
  "stream": true
}
```

### Latency Budget

```
Token Generation: ~20ms/token
Talker Processing: ~10ms/16 codes
Code2Wav Decode:   ~50ms/chunk (480 samples = 20ms audio)

Target: <100ms first audio chunk
Strategy: Start Code2Wav after first 50 audio tokens
```

## Quantization Strategy

### Thinker (Main LLM)

Standard llama.cpp quantization works well:

| Quantization | Size | Quality | Recommendation |
|--------------|------|---------|----------------|
| Q4_K_M | ~18GB | Good | Default choice |
| Q5_K_M | ~22GB | Better | Quality-focused |
| Q8_0 | ~35GB | Best | Maximum quality |
| F16 | ~70GB | Reference | Development only |

### Talker (Audio Token Generation)

**Caution**: More sensitive to quantization than text LLMs.

| Quantization | Quality Impact | Recommendation |
|--------------|----------------|----------------|
| F16 | None | Best for audio quality |
| Q8_0 | Minimal | Good balance |
| Q6_K | Slight | Acceptable |
| Q4_K_M | Noticeable | Not recommended |

**Reason**: Quantization noise compounds through 16 codebook predictions.

**Tensor-specific rules**:
```python
def tensor_force_quant(name):
    if "mlp.gate" in name: return F16        # Keep MoE gate precise
    if "codec_embeddings" in name: return F16 # Keep embeddings precise
    return None  # Use default
```

### Code2Wav (Neural Codec)

**Critical**: Codebooks should NOT be quantized.

| Component | Quantization | Reason |
|-----------|--------------|--------|
| Codebook embeddings | F16 | Lookup table, must be exact |
| Transformer layers | Q8_0 | Moderate quantization OK |
| ConvTranspose weights | F16 | Sensitive to quantization |
| ConvNeXt blocks | Q8_0 | Less sensitive |

### Memory Budget (Quantized)

| Component | F16 Size | Quantized Size |
|-----------|----------|----------------|
| Thinker | ~70GB | ~18GB (Q4_K_M) |
| mmproj | ~3GB | ~3GB (F16) |
| Talker | ~8GB | ~4GB (Q8_0) |
| Code2Wav | ~1GB | ~0.8GB (mostly F16) |
| **Total** | ~82GB | ~26GB |

Quantized model fits in 32GB VRAM.

## Test Strategy

### Phase 1: Thinker Tests

```bash
# Compare against HuggingFace reference
python tests/compare_hf.py \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --gguf qwen3-omni.gguf \
    --input "Describe this image" \
    --image test.jpg \
    --tolerance 1e-4

# Audio transcription WER
./mtmd-cli -m qwen3-omni.gguf --mmproj mmproj.gguf \
    --audio librispeech_test.wav -p "Transcribe this audio"
```

### Phase 2: Talker Tests

```bash
# Compare Talker output tokens against HuggingFace
python tests/compare_talker.py \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --gguf talker.gguf \
    --text "Hello world" \
    --compare-tokens
```

### Phase 3: Audio Quality Metrics

```bash
# PESQ/STOI/SI-SDR comparison
python tests/audio_quality.py \
    --reference audio_hf.wav \
    --generated audio_gguf.wav \
    --metrics "pesq,stoi,si-sdr"

# Expected baselines:
# PESQ: > 3.5 (good quality)
# STOI: > 0.9 (intelligibility)
# SI-SDR: > 15 dB (signal quality)
```

### Phase 4: End-to-End

```bash
./qwen3-omni-cli -m qwen3-omni.gguf \
    --mmproj mmproj.gguf \
    --talker talker.gguf \
    -p "Say hello in a friendly voice" \
    --voice chelsie \
    -o output.wav

ffplay output.wav
```

## Files to Modify/Create

### New Files

| File | Purpose |
|------|---------|
| `src/models/qwen3omni.cpp` | Thinker graph builder |
| `src/models/qwen3omni-talker.cpp` | Talker graph builder |
| `tools/mtmd/models/qwen3omni.cpp` | Vision encoder variant |
| `tools/mtmd/code2wav.cpp` | Audio codec decoder |
| `tools/qwen3-omni/` | CLI tool for omni inference |

### Files to Extend

| File | Changes |
|------|---------|
| `src/llama-arch.h` | Add `LLM_ARCH_QWEN3OMNIMOE`, `LLM_ARCH_QWEN3OMNI_TALKER` |
| `src/llama-arch.cpp` | Add tensor mappings |
| `gguf-py/gguf/constants.py` | Add Talker/Code2Wav metadata keys |
| `gguf-py/gguf/tensor_mapping.py` | Add tensor name mappings |
| `convert_hf_to_gguf.py` | Add `Qwen3OmniMoeForConditionalGeneration` converters |
| `tools/mtmd/mtmd.cpp` | Add omni model type |
| `tools/mtmd/mtmd-audio.cpp` | Support 128 mel bins |
| `tools/mtmd/clip-model.h` | Add encoder hparams |
| `tools/server/server-context.cpp` | Add audio output support |
| `tools/server/server-http.cpp` | Add binary streaming |

## Implementation Complexity Summary

| Component | New Code | Reusable | Complexity |
|-----------|----------|----------|------------|
| Thinker | ~500 LOC | 80% (Qwen3MOE) | Medium |
| Audio Encoder | ~200 LOC | 90% (Whisper) | Low |
| Vision Encoder | ~200 LOC | 95% (Qwen3VL) | Low |
| TMRoPE | ~100 LOC | 70% (M-RoPE) | Low |
| Talker | ~1500 LOC | 40% (MoE base) | High |
| Code Predictor | ~500 LOC | 30% | Medium |
| Code2Wav | ~2000 LOC | 10% | Very High |
| GGUF Converter | ~800 LOC | 50% | Medium |
| Dual-model API | ~500 LOC | 20% | Medium |

**Total estimated new code**: ~6,000-8,000 lines

## References

- [HuggingFace Model](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
- [GitHub Issue #16186](https://github.com/ggml-org/llama.cpp/issues/16186)
- [Transformers PR #41025](https://github.com/huggingface/transformers/pull/41025)
- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215)
- [SNAC: Multi-Scale Neural Audio Codec](https://arxiv.org/abs/2410.14411)
- [SNAC GitHub Repository](https://github.com/hubertsiuzdak/snac)
