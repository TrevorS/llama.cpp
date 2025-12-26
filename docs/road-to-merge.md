# Qwen3-Omni: Road to Merge

> Strategy for upstreaming Qwen3-Omni support to ggml-org/llama.cpp

## Overview

Qwen3-Omni is a multimodal model with text, audio, image, and video input, plus text AND speech output. The implementation spans ~5,200 lines across 29 core files.

Per [llama.cpp contribution guidelines](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md), we should:
1. Focus on CPU support first (CUDA in follow-up PRs)
2. Split large changes into reviewable chunks
3. Include tests and documentation

## PR Strategy

```
PR 1: Thinker (text-only)     ─────────────────────────────┐
      ~1,800 lines                                         │
                                                           ▼
PR 2: Audio Encoder ──────────────────────────────► PR 4: TTS Pipeline
      ~400 lines                                          ~2,800 lines
                                                           ▲
PR 3: Vision Encoder ─────────────────────────────────────┘
      ~400 lines
```

---

## PR 1: Thinker (Text-Only MoE)

**Priority:** Must merge first - foundation for all other PRs

**Scope:** Text input → Text output only

### Files Changed (~1,800 lines)

| File | Changes |
|------|---------|
| `convert_hf_to_gguf.py` | `Qwen3OmniMoeModel` class (~300 lines) |
| `gguf-py/gguf/constants.py` | `MODEL_ARCH.QWEN3OMNIMOE`, tensor lists (~150 lines) |
| `gguf-py/gguf/tensor_mapping.py` | HF→GGUF tensor mappings (~100 lines) |
| `src/llama-arch.h` | `LLM_ARCH_QWEN3OMNIMOE` enum |
| `src/llama-arch.cpp` | Arch name, tensor name mappings (~180 lines) |
| `src/llama-model.cpp` | `load_hparams()`, `load_tensors()`, `build_graph()` (~300 lines) |
| `src/llama-hparams.h` | Any new hyperparams |

### Key Technical Details

- 48-layer MoE with 128 experts, 8 active per token
- Uses IMRoPE (Interleaved M-RoPE, `rope_type=40`) with sections `[24, 20, 20, 0]`
- Shared expert support (though Qwen3-Omni has 0 shared experts)
- 152K vocab with special TTS tokens (can ignore for text-only)

### Testing Requirements

```bash
# Convert model
python convert_hf_to_gguf.py Qwen/Qwen3-Omni-30B-A3B-Instruct --outtype f16

# Test inference
./llama-cli -m qwen3-omni-thinker.gguf -p "Hello" -n 64 -st
```

### PR Description Template

```markdown
## Summary
Add support for Qwen3-Omni Thinker (text-only mode).

Qwen3-Omni is Alibaba's multimodal model with text/audio/image/video input
and text+speech output. This PR adds the core Thinker component for text→text.

## Changes
- New architecture: `QWEN3OMNIMOE` (48-layer MoE, 128 experts)
- GGUF conversion for Thinker weights
- IMRoPE (Interleaved M-RoPE) support

## Testing
- [x] llama-cli text generation
- [x] llama-completion
- [x] CPU backend
- [ ] CUDA backend (follow-up PR)
- [ ] Metal backend (follow-up PR)

## Related
- Model: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
- Issue: #16186
```

---

## PR 2: Audio Encoder

**Priority:** After PR 1 merges

**Scope:** Audio input → Text output via mtmd

### Files Changed (~400 lines)

| File | Changes |
|------|---------|
| `convert_hf_to_gguf.py` | Audio encoder tensors in mmproj |
| `gguf-py/gguf/constants.py` | Audio encoder tensor list |
| `tools/mtmd/models/qwen3omni-audio.cpp` | Audio encoder graph (new, ~140 lines) |
| `tools/mtmd/mtmd-audio.cpp` | M-RoPE integration (~90 lines) |
| `tools/mtmd/mtmd.cpp` | `PROJECTOR_TYPE_QWEN3OMNI_AUDIO` |
| `tools/mtmd/clip.cpp` | Audio projector loading |

### Key Technical Details

- 32-layer Whisper-style encoder with 1280 hidden dim
- 3x Conv2d stack for mel→embeddings
- Projects to 2048 dim (matches Thinker)
- Requires M-RoPE position IDs (4 per embedding)

### Testing Requirements

```bash
./llama-mtmd-cli \
  -m thinker.gguf \
  --mmproj mmproj.gguf \
  --audio test.wav \
  -p "What did you hear?" -st
```

---

## PR 3: Vision Encoder

**Priority:** After PR 1 merges (parallel to PR 2)

**Scope:** Image input → Text output via mtmd

### Files Changed (~400 lines)

| File | Changes |
|------|---------|
| `convert_hf_to_gguf.py` | Vision encoder tensors in mmproj |
| `gguf-py/gguf/constants.py` | Vision tensor list |
| `src/models/qwen3vl-moe.cpp` | Extend for Qwen3-Omni (~40 lines) |
| `tools/mtmd/mtmd.cpp` | `PROJECTOR_TYPE_QWEN3OMNI_VISION` |
| `tools/mtmd/clip.cpp` | Vision projector + deepstack |

### Key Technical Details

- 27-layer ViT with 1152 hidden dim
- Deepstack: 4 merger layers (main + layers 8, 16, 24)
- Output: 8192 dim (2048 × 4 from deepstack)
- Mostly reuses existing Qwen3-VL code

### Testing Requirements

```bash
./llama-mtmd-cli \
  -m thinker.gguf \
  --mmproj mmproj.gguf \
  --image test.jpg \
  -p "What is in this image?" -st
```

---

## PR 4: TTS Pipeline (Talker + Code2Wav)

**Priority:** After PR 1 merges

**Scope:** Any input → Speech output

### Files Changed (~2,800 lines)

| File | Changes |
|------|---------|
| `convert_hf_to_gguf.py` | Talker + Code Predictor + Code2Wav tensors |
| `gguf-py/gguf/constants.py` | `MODEL_ARCH.QWEN3OMNITALKER`, tensor lists |
| `src/llama-arch.h` | `LLM_ARCH_QWEN3OMNITALKER` |
| `src/llama-arch.cpp` | Talker tensor mappings |
| `src/llama-model.h` | Talker tensor pointers (~100 lines) |
| `src/llama-model.cpp` | Load Talker/CP/C2W tensors |
| `src/models/qwen3omni_talker.cpp` | Talker graph builder (new, ~625 lines) |
| `tools/mtmd/mtmd-tts.h` | TTS API (new, ~170 lines) |
| `tools/mtmd/mtmd-tts.cpp` | TTS pipeline (new, ~1,170 lines) |
| `tools/mtmd/mtmd-tts-code2wav.cpp` | HiFi-GAN vocoder (new, ~850 lines) |
| `tools/mtmd/mtmd-cli.cpp` | `--tts-model`, `--speak` flags (~150 lines) |

### Key Technical Details

**Talker (20-layer MoE):**
- 1024 hidden dim, 128 experts, 8 active
- Generates codec tokens autoregressively
- Uses text projection MLP (2048→1024)

**Code Predictor (5-layer dense):**
- Expands 1 codec token to 16 codebooks
- GQA attention with QK normalization

**Code2Wav (HiFi-GAN):**
- 16 VQ codebooks → 24kHz waveform
- ConvNeXt + HiFi-GAN architecture
- Snake activation functions

### Testing Requirements

```bash
./llama-mtmd-cli \
  -m thinker.gguf \
  --mmproj mmproj.gguf \
  --tts-model talker.gguf \
  -p "Hello world" \
  --speak \
  --tts-output output.wav
```

---

## Combined mmproj GGUF

PRs 2-4 require a combined multimodal projector GGUF containing:

```
Audio Encoder:   32 transformer layers + conv stack + projection
Vision Encoder:  27 transformer layers + deepstack mergers
Talker:          20 MoE layers + text projection MLP
Code Predictor:  5 dense layers + 15 codebook embeddings
Code2Wav:        ConvNeXt blocks + HiFi-GAN upsampler
```

Current combined size: ~2.2 GB (F16)

---

## Timeline Estimate

| PR | Dependencies | Status |
|----|--------------|--------|
| PR 1: Thinker | None | Ready to submit |
| PR 2: Audio | PR 1 merged | Ready after PR 1 |
| PR 3: Vision | PR 1 merged | Ready after PR 1 |
| PR 4: TTS | PR 1 merged | Ready after PR 1 |

---

## Pre-Submission Checklist

- [ ] Rebase onto latest master
- [ ] Remove debug code and test scripts
- [ ] Run `git clang-format` on C++ files
- [ ] Run `flake8` on Python files
- [ ] Test on CPU backend
- [ ] Write PR description with test commands
- [ ] Link to HuggingFace model and issue #16186

---

## References

- [HOWTO-add-model.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md)
- [Adding new architectures discussion](https://github.com/ggml-org/llama.cpp/discussions/16770)
- [Qwen3-VL PR #16780](https://github.com/ggml-org/llama.cpp/pull/16780) - Similar multimodal addition
- [Kimi-Linear PR #18381](https://github.com/ggml-org/llama.cpp/pull/18381) - Recent MoE addition
- [Issue #16186](https://github.com/ggml-org/llama.cpp/issues/16186) - Qwen3-Omni request
