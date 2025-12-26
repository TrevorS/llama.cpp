#!/usr/bin/env python3
"""
HuggingFace reference script for debugging Qwen3-Omni TTS pipeline.

Extracts and saves tensor values at each stage for comparison with llama.cpp.

Usage:
    python debug_hf_pipeline.py --text "Hello world" --output /models/debug
"""

import argparse
import struct
import sys
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoProcessor, Qwen3OmniMoeForConditionalGeneration
except ImportError:
    print("Error: Required packages not found. Install with:")
    print("  pip install torch transformers numpy")
    sys.exit(1)


def save_tensor(tensor: torch.Tensor, path: Path, name: str) -> None:
    """Save tensor to binary file with dimension header."""
    tensor = tensor.detach().float().cpu()
    shape = tensor.shape

    output_file = path / f"{name}.bin"
    with open(output_file, 'wb') as f:
        f.write(struct.pack('<I', len(shape)))
        for dim in shape:
            f.write(struct.pack('<I', dim))
        f.write(tensor.numpy().astype('<f4').tobytes())

    print(f"  Saved {name}: shape {list(shape)}, {output_file.stat().st_size} bytes")


def save_tokens(tokens: torch.Tensor, path: Path, name: str) -> None:
    """Save token IDs to binary file."""
    tokens = tokens.detach().cpu()

    output_file = path / f"{name}.bin"
    with open(output_file, 'wb') as f:
        f.write(struct.pack('<I', 1))  # 1 dimension
        f.write(struct.pack('<I', tokens.numel()))
        f.write(tokens.numpy().astype('<i4').tobytes())

    print(f"  Saved {name}: {tokens.numel()} tokens")


class TensorCapture:
    """Hook-based tensor capture for model internals."""

    def __init__(self):
        self.captured = {}
        self.hooks = []

    def capture(self, name: str, tensor: torch.Tensor):
        """Capture a tensor by name."""
        self.captured[name] = tensor.clone()

    def save_all(self, path: Path):
        """Save all captured tensors."""
        for name, tensor in self.captured.items():
            if tensor.dtype in (torch.int32, torch.int64, torch.long):
                save_tokens(tensor, path, name)
            else:
                save_tensor(tensor, path, name)


def debug_tts_pipeline(
    text: str,
    output_dir: str,
    model_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    speaker: str = "Ethan",
    device: str = "auto",
) -> Dict[str, Any]:
    """Run TTS pipeline and extract intermediate tensors."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    capture = TensorCapture()

    print(f"Loading model from {model_path}...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Load to CUDA directly to avoid meta tensor issues
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # Check talker
    if not model.has_talker:
        print("Enabling talker...")
        model.enable_talker()

    # Get config info
    config = model.config
    print(f"\n=== Configuration ===")
    print(f"Text: {text!r}")
    print(f"Speaker: {speaker}")
    print(f"im_start_token_id: {config.im_start_token_id}")
    print(f"tts_bos_token_id: {config.tts_bos_token_id}")
    print(f"tts_eos_token_id: {config.tts_eos_token_id}")
    print(f"tts_pad_token_id: {config.tts_pad_token_id}")

    # Format input to match llama.cpp (NO system message for fair comparison)
    # llama.cpp uses: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    formatted_text = (
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    print(f"\n=== Formatted Input ===")
    print(formatted_text[:500] + "..." if len(formatted_text) > 500 else formatted_text)

    # Tokenize
    input_ids = tokenizer(formatted_text, return_tensors="pt").input_ids.to(device)

    print(f"\n=== Stage 1: Input Tokenization ===")
    print(f"Input shape: {input_ids.shape}")
    print(f"First 20 tokens: {input_ids[0, :20].tolist()}")
    capture.capture("01_input_ids", input_ids)

    # Get Thinker tok_embd for input
    print(f"\n=== Stage 2: Thinker Token Embeddings (tok_embd) ===")
    with torch.no_grad():
        thinker_tok_embd = model.thinker.model.embed_tokens(input_ids)
    print(f"tok_embd shape: {thinker_tok_embd.shape}")
    capture.capture("02_thinker_tok_embd", thinker_tok_embd[:, :10])  # First 10 positions

    # Get TTS special token embeddings (before projection)
    print(f"\n=== Stage 3: TTS Special Token Embeddings ===")
    tts_special_ids = torch.tensor([
        [config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]
    ], device=device)
    with torch.no_grad():
        tts_special_embeds_raw = model.thinker.model.embed_tokens(tts_special_ids)
    print(f"TTS special embeds (raw) shape: {tts_special_embeds_raw.shape}")
    capture.capture("03_tts_special_embeds_raw", tts_special_embeds_raw)

    # Project TTS special tokens through text_projection (key for comparison!)
    print(f"\n=== Stage 3b: Projected TTS Special Token Embeddings ===")
    with torch.no_grad():
        tts_special_projected = model.talker.text_projection(tts_special_embeds_raw)
        # Split into individual embeddings: bos, eos, pad
        tts_bos_embed, tts_eos_embed, tts_pad_embed = tts_special_projected.chunk(3, dim=1)
    print(f"tts_bos_embed shape: {tts_bos_embed.shape}")
    print(f"tts_eos_embed shape: {tts_eos_embed.shape}")
    print(f"tts_pad_embed shape: {tts_pad_embed.shape}")
    capture.capture("03b_tts_bos_embed", tts_bos_embed.squeeze(1))  # [1, 1024]
    capture.capture("03b_tts_eos_embed", tts_eos_embed.squeeze(1))  # [1, 1024]
    capture.capture("03b_tts_pad_embed", tts_pad_embed.squeeze(1))  # [1, 1024]

    # Run Thinker forward pass to get hidden states and generated text
    # Match llama.cpp: it generated 22 tokens for "Hello world"
    print(f"\n=== Stage 3c: Thinker Generation ===")
    with torch.no_grad():
        thinker_outputs = model.thinker.generate(
            input_ids=input_ids,
            max_new_tokens=30,  # Match llama.cpp's ~22 generated tokens
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    thinker_output_ids = thinker_outputs.sequences
    print(f"Thinker output shape: {thinker_output_ids.shape}")
    print(f"Thinker decoded: {tokenizer.decode(thinker_output_ids[0], skip_special_tokens=True)[:200]}")
    capture.capture("03c_thinker_output_ids", thinker_output_ids)

    # Get the embeddings for the full sequence (input + generated)
    print(f"\n=== Stage 3d: Full Sequence Embeddings ===")
    with torch.no_grad():
        full_thinker_embeds = model.thinker.model.embed_tokens(thinker_output_ids)
    print(f"Full thinker embeds shape: {full_thinker_embeds.shape}")

    # Find assistant response start
    # The assistant segment is the first im_start AFTER the input prompt
    im_start_id = config.im_start_token_id
    im_end_id = config.im_end_token_id
    tokens_list = thinker_output_ids[0].tolist()
    n_input = input_ids.shape[1]

    # Find the first im_start that appears in the generated portion (after input)
    # This is where the assistant response begins
    assistant_start_idx = None
    assistant_end_idx = None

    # Start from just before the input ends (the last im_start in input)
    for i in range(n_input - 1, -1, -1):
        if tokens_list[i] == im_start_id:
            assistant_start_idx = i
            break

    if assistant_start_idx is None:
        print("WARNING: Could not find assistant im_start!")
        assistant_start_idx = n_input - 1

    # Find the end of this assistant segment
    # Match llama.cpp logic: use end of sequence, exclude im_end if present
    assistant_end_idx = len(tokens_list)
    if tokens_list[-1] == im_end_id:
        assistant_end_idx = len(tokens_list) - 1  # Exclude the im_end token

    print(f"Assistant segment: tokens {assistant_start_idx} to {assistant_end_idx}")
    print(f"  (input had {n_input} tokens, using {assistant_end_idx - assistant_start_idx} assistant tokens)")

    # Project assistant embeddings through text_projection
    print(f"\n=== Stage 3e: Assistant Hidden States Projection ===")
    with torch.no_grad():
        # Get embeddings for assistant portion (INCLUDING im_start to match llama.cpp)
        # llama.cpp uses all tokens from assistant_start_pos to assistant_end_pos
        assistant_embeds = full_thinker_embeds[:, assistant_start_idx:assistant_end_idx]
        print(f"Assistant embeds (before projection) shape: {assistant_embeds.shape}")

        # Project through text_projection
        assistant_hidden = model.talker.text_projection(assistant_embeds)
        print(f"Assistant hidden (after projection) shape: {assistant_hidden.shape}")
    capture.capture("03e_assistant_hidden", assistant_hidden.squeeze(0))  # [n_tokens, 1024]

    # Construct trailing_text_hidden (assistant_hidden[:, 4:] + tts_eos_embed)
    # This is what gets injected during generation
    print(f"\n=== Stage 3f: Trailing Text Hidden ===")
    if assistant_hidden.shape[1] > 4:
        trailing_text_hidden = torch.cat([
            assistant_hidden[:, 4:],  # Skip first 4 tokens
            tts_eos_embed,            # Add EOS at end
        ], dim=1)
    else:
        trailing_text_hidden = tts_eos_embed
    print(f"trailing_text_hidden shape: {trailing_text_hidden.shape}")
    capture.capture("03f_trailing_text_hidden", trailing_text_hidden.squeeze(0))  # [n_trailing, 1024]

    # Construct prefill embeddings (9 positions)
    print(f"\n=== Stage 3g: Prefill Embeddings ===")
    # Get talker config for special token IDs
    talker_config = config.talker_config if hasattr(config, 'talker_config') else config
    codec_nothink_id = getattr(talker_config, 'codec_nothink_id', 2155)
    codec_think_bos_id = getattr(talker_config, 'codec_think_bos_id', 2156)
    codec_think_eos_id = getattr(talker_config, 'codec_think_eos_id', 2157)
    codec_pad_id = getattr(talker_config, 'codec_pad_id', 2148)
    codec_bos_id = getattr(talker_config, 'codec_bos_id', 2149)

    # Get speaker ID
    speaker_map = {"Ethan": 2302, "Chelsie": 2303, "default": 2302}
    speaker_id = speaker_map.get(speaker, 2302)

    print(f"  codec_nothink_id: {codec_nothink_id}")
    print(f"  codec_think_bos_id: {codec_think_bos_id}")
    print(f"  codec_think_eos_id: {codec_think_eos_id}")
    print(f"  codec_pad_id: {codec_pad_id}")
    print(f"  codec_bos_id: {codec_bos_id}")
    print(f"  speaker_id: {speaker_id}")

    with torch.no_grad():
        # Text embeddings for prefill:
        # Positions 0-2: assistant_hidden[:, :3]
        # Positions 3-6: tts_pad_embed (4 times)
        # Position 7: tts_bos_embed
        # Position 8: assistant_hidden[:, 3:4]
        if assistant_hidden.shape[1] >= 4:
            text_prefill = torch.cat([
                assistant_hidden[:, :3],              # Positions 0-2
                tts_pad_embed.expand(-1, 4, -1),      # Positions 3-6
                tts_bos_embed,                        # Position 7
                assistant_hidden[:, 3:4],             # Position 8
            ], dim=1)
        else:
            # Fallback for very short responses
            text_prefill = torch.cat([
                assistant_hidden,
                tts_pad_embed.expand(-1, 9 - assistant_hidden.shape[1], -1),
            ], dim=1)

        print(f"text_prefill shape: {text_prefill.shape}")

        # Codec embeddings for prefill:
        # Positions 0-2: zeros
        # Positions 3-8: codec special tokens
        codec_ids = torch.tensor([[
            codec_nothink_id,
            codec_think_bos_id,
            codec_think_eos_id,
            speaker_id,
            codec_pad_id,
            codec_bos_id,
        ]], device=device, dtype=torch.long)
        codec_embeds = model.talker.get_input_embeddings()(codec_ids)
        print(f"codec_embeds shape: {codec_embeds.shape}")

        # Full codec prefill with zeros for first 3 positions
        n_hidden = text_prefill.shape[-1]
        codec_prefill = torch.cat([
            torch.zeros(1, 3, n_hidden, device=device, dtype=codec_embeds.dtype),
            codec_embeds,
        ], dim=1)
        print(f"codec_prefill shape: {codec_prefill.shape}")

        # Combined prefill = text + codec
        prefill_embeds = text_prefill + codec_prefill
        print(f"prefill_embeds shape: {prefill_embeds.shape}")

    capture.capture("03g_text_prefill", text_prefill.squeeze(0))     # [9, 1024]
    capture.capture("03g_codec_prefill", codec_prefill.squeeze(0))   # [9, 1024]
    capture.capture("03g_prefill_embeds", prefill_embeds.squeeze(0)) # [9, 1024]

    # Run full generation with return_audio=True
    print(f"\n=== Stage 4: Full Generation (Thinker + Talker + Code2Wav) ===")
    print("Running model.generate()...")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            speaker=speaker,
            return_audio=True,
            thinker_max_new_tokens=256,
            talker_max_new_tokens=500,  # Match llama.cpp max
        )

    # Handle different output formats
    print(f"\n=== Stage 5: Output Analysis ===")
    print(f"Output type: {type(outputs)}")

    if isinstance(outputs, tuple):
        text_output = outputs[0]
        audio_output = outputs[1] if len(outputs) > 1 else None
    else:
        # GenerateDecoderOnlyOutput object
        text_output = outputs.sequences if hasattr(outputs, 'sequences') else outputs
        audio_output = outputs.audio if hasattr(outputs, 'audio') else None
        print(f"Output attributes: {[a for a in dir(outputs) if not a.startswith('_')]}")

    if hasattr(text_output, 'shape'):
        print(f"Text tokens shape: {text_output.shape}")
        print(f"Decoded: {tokenizer.decode(text_output[0], skip_special_tokens=True)[:200]}")
        capture.capture("05_thinker_output_ids", text_output)
    else:
        print(f"Text output: {text_output}")

    if audio_output is not None:
        print(f"\n=== Stage 6: Audio Output ===")
        print(f"Audio shape: {audio_output.shape}")
        print(f"Audio duration: {audio_output.shape[-1] / 24000:.2f}s")

        # Save audio as WAV
        import scipy.io.wavfile as wavfile
        audio_np = audio_output.cpu().numpy().squeeze()
        audio_np = np.clip(audio_np, -1, 1)
        audio_path = output_path / "output.wav"
        wavfile.write(str(audio_path), 24000, (audio_np * 32767).astype(np.int16))
        print(f"  Saved audio: {audio_path}")

    # Save all captured tensors
    print(f"\n=== Saving Captured Tensors ===")
    capture.save_all(output_path)

    # Write metadata
    metadata = {
        "text": text,
        "speaker": speaker,
        "model_path": model_path,
        "im_start_token_id": config.im_start_token_id,
        "tts_bos_token_id": config.tts_bos_token_id,
        "tts_eos_token_id": config.tts_eos_token_id,
        "tts_pad_token_id": config.tts_pad_token_id,
    }

    import json
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata.json")

    print(f"\n=== Done ===")
    print(f"Output directory: {output_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Debug Qwen3-Omni TTS pipeline")
    parser.add_argument("--text", "-t", required=True, help="Input text")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--model", "-m", default="/models/Qwen3-Omni-30B-A3B-Instruct", help="Model path")
    parser.add_argument("--speaker", "-s", default="Ethan", help="Speaker name")
    parser.add_argument("--device", "-d", default="auto", help="Device (auto, cuda, cpu)")

    args = parser.parse_args()

    try:
        metadata = debug_tts_pipeline(
            text=args.text,
            output_dir=args.output,
            model_path=args.model,
            speaker=args.speaker,
            device=args.device,
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
