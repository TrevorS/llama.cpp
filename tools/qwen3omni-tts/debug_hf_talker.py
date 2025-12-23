#!/usr/bin/env python3
"""
HuggingFace Talker debug script for systematic comparison with llama.cpp.

Extracts per-step tensors during Talker generation:
- Prefill embeddings (positions 0-8)
- Hidden states after each decode step
- Logits at each step
- Code Predictor inputs/outputs
- All sampled tokens (16 codebooks × N frames)

Usage:
    python debug_hf_talker.py --text "The quick brown fox jumps over the lazy dog" \
        --output /models/debug/hf_talker --max-tokens 20
"""

import argparse
import json
import os
import struct
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch


def save_tensor(tensor: torch.Tensor, path: Path, name: str) -> None:
    """Save tensor to .npy file (easier to load in Python)."""
    arr = tensor.detach().float().cpu().numpy()
    output_file = path / f"{name}.npy"
    np.save(output_file, arr)
    print(f"  Saved {name}: shape={arr.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}")


def save_tensor_bin(tensor: torch.Tensor, path: Path, name: str) -> None:
    """Save tensor to binary file with dimension header (matches C++ format)."""
    tensor = tensor.detach().float().cpu()
    shape = tensor.shape

    output_file = path / f"{name}.bin"
    with open(output_file, 'wb') as f:
        f.write(struct.pack('<I', len(shape)))
        for dim in shape:
            f.write(struct.pack('<I', dim))
        f.write(tensor.numpy().astype('<f4').tobytes())


def debug_talker_generation(
    text: str,
    output_dir: str,
    model_path: str = "/models/Qwen3-Omni-30B-A3B-Instruct",
    speaker: str = "Ethan",
    max_tokens: int = 50,
    device: str = "cuda",
):
    """Run Talker generation with per-step tensor extraction."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    
    from transformers import AutoTokenizer, Qwen3OmniMoeForConditionalGeneration
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    
    if not model.has_talker:
        print("Enabling talker...")
        model.enable_talker()
    
    config = model.config
    
    # Format input text (NO system message for fair comparison)
    formatted_text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    print(f"\n=== Input ===")
    print(f"Text: {text}")
    print(f"Formatted: {formatted_text[:200]}...")
    
    # Tokenize
    input_ids = tokenizer(formatted_text, return_tensors="pt").input_ids.to(device)
    print(f"Input tokens: {input_ids.shape}")
    
    # Save input tokens
    np.save(output_path / "00_input_ids.npy", input_ids.cpu().numpy())
    
    with torch.no_grad():
        # ========================================
        # Stage 1: Run Thinker to get hidden states
        # ========================================
        print(f"\n=== Stage 1: Thinker Generation ===")
        
        thinker_outputs = model.thinker.generate(
            input_ids=input_ids,
            max_new_tokens=50,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        thinker_output_ids = thinker_outputs.sequences
        print(f"Thinker output: {thinker_output_ids.shape}")
        print(f"Decoded: {tokenizer.decode(thinker_output_ids[0], skip_special_tokens=True)[:200]}")
        
        np.save(output_path / "01_thinker_output_ids.npy", thinker_output_ids.cpu().numpy())
        
        # ========================================
        # Stage 2: Get embeddings and project
        # ========================================
        print(f"\n=== Stage 2: Text Projection ===")
        
        # Get full sequence embeddings
        full_embeds = model.thinker.model.embed_tokens(thinker_output_ids)
        print(f"Full embeds: {full_embeds.shape}")
        
        # Find assistant segment
        im_start_id = config.im_start_token_id
        im_end_id = config.im_end_token_id
        tokens_list = thinker_output_ids[0].tolist()
        n_input = input_ids.shape[1]
        
        # Find assistant start (last im_start in input)
        assistant_start_idx = None
        for i in range(n_input - 1, -1, -1):
            if tokens_list[i] == im_start_id:
                assistant_start_idx = i
                break
        if assistant_start_idx is None:
            assistant_start_idx = n_input - 1
        
        # Find assistant end (exclude trailing im_end if present)
        assistant_end_idx = len(tokens_list)
        if tokens_list[-1] == im_end_id:
            assistant_end_idx = len(tokens_list) - 1
        
        print(f"Assistant segment: {assistant_start_idx} to {assistant_end_idx}")
        
        # Get assistant embeddings and project
        assistant_embeds = full_embeds[:, assistant_start_idx:assistant_end_idx]
        print(f"Assistant embeds (before projection): {assistant_embeds.shape}")
        
        assistant_hidden = model.talker.text_projection(assistant_embeds)
        print(f"Assistant hidden (after projection): {assistant_hidden.shape}")
        
        save_tensor(assistant_hidden.squeeze(0), output_path, "02_assistant_hidden")
        
        # ========================================
        # Stage 3: TTS Special Token Embeddings
        # ========================================
        print(f"\n=== Stage 3: TTS Special Embeddings ===")
        
        tts_special_ids = torch.tensor([
            [config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]
        ], device=device)
        tts_special_raw = model.thinker.model.embed_tokens(tts_special_ids)
        tts_special_projected = model.talker.text_projection(tts_special_raw)
        
        tts_bos_embed = tts_special_projected[:, 0:1]  # [1, 1, 1024]
        tts_eos_embed = tts_special_projected[:, 1:2]
        tts_pad_embed = tts_special_projected[:, 2:3]
        
        save_tensor(tts_bos_embed.squeeze(0), output_path, "03_tts_bos_embed")
        save_tensor(tts_eos_embed.squeeze(0), output_path, "03_tts_eos_embed")
        save_tensor(tts_pad_embed.squeeze(0), output_path, "03_tts_pad_embed")
        
        # ========================================
        # Stage 4: Trailing Text Hidden
        # ========================================
        print(f"\n=== Stage 4: Trailing Text Hidden ===")
        
        n_text = assistant_hidden.shape[1]
        if n_text > 4:
            trailing_text_hidden = torch.cat([
                assistant_hidden[:, 4:],
                tts_eos_embed,
            ], dim=1)
        else:
            trailing_text_hidden = tts_eos_embed
        print(f"Trailing text hidden: {trailing_text_hidden.shape}")
        
        save_tensor(trailing_text_hidden.squeeze(0), output_path, "04_trailing_text_hidden")
        
        # ========================================
        # Stage 5: Prefill Construction
        # ========================================
        print(f"\n=== Stage 5: Prefill Construction ===")
        
        # Get talker config
        talker_config = config.talker_config if hasattr(config, 'talker_config') else config
        codec_nothink_id = getattr(talker_config, 'codec_nothink_id', 2155)
        codec_think_bos_id = getattr(talker_config, 'codec_think_bos_id', 2156)
        codec_think_eos_id = getattr(talker_config, 'codec_think_eos_id', 2157)
        codec_pad_id = getattr(talker_config, 'codec_pad_id', 2148)
        codec_bos_id = getattr(talker_config, 'codec_bos_id', 2149)
        
        speaker_map = {"Ethan": 2302, "Chelsie": 2303, "default": 2302}
        speaker_id = speaker_map.get(speaker, 2302)
        
        print(f"  codec_nothink_id: {codec_nothink_id}")
        print(f"  codec_think_bos_id: {codec_think_bos_id}")
        print(f"  codec_think_eos_id: {codec_think_eos_id}")
        print(f"  codec_pad_id: {codec_pad_id}")
        print(f"  codec_bos_id: {codec_bos_id}")
        print(f"  speaker_id: {speaker_id}")
        
        # Text prefill: [text[0:3], pad×4, bos, text[3]]
        if n_text >= 4:
            text_prefill = torch.cat([
                assistant_hidden[:, :3],
                tts_pad_embed.expand(-1, 4, -1),
                tts_bos_embed,
                assistant_hidden[:, 3:4],
            ], dim=1)
        else:
            text_prefill = torch.cat([
                assistant_hidden,
                tts_pad_embed.expand(-1, 9 - n_text, -1),
            ], dim=1)
        
        print(f"Text prefill: {text_prefill.shape}")
        save_tensor(text_prefill.squeeze(0), output_path, "05_text_prefill")
        
        # Codec prefill: [zeros×3, nothink, think_bos, think_eos, speaker, pad, bos]
        codec_ids = torch.tensor([[
            codec_nothink_id,
            codec_think_bos_id,
            codec_think_eos_id,
            speaker_id,
            codec_pad_id,
            codec_bos_id,
        ]], device=device, dtype=torch.long)
        codec_embeds = model.talker.get_input_embeddings()(codec_ids)
        
        n_hidden = text_prefill.shape[-1]
        codec_prefill = torch.cat([
            torch.zeros(1, 3, n_hidden, device=device, dtype=codec_embeds.dtype),
            codec_embeds,
        ], dim=1)
        
        print(f"Codec prefill: {codec_prefill.shape}")
        save_tensor(codec_prefill.squeeze(0), output_path, "05_codec_prefill")
        
        # Combined prefill
        prefill_embeds = text_prefill + codec_prefill
        print(f"Combined prefill: {prefill_embeds.shape}")
        save_tensor(prefill_embeds.squeeze(0), output_path, "05_prefill_embeds")
        
        # Save each prefill position separately for detailed comparison
        for pos in range(9):
            save_tensor(prefill_embeds[0, pos], output_path, f"05_prefill_pos{pos}")
        
        # ========================================
        # Stage 6: Run Talker Generation with Hooks
        # ========================================
        print(f"\n=== Stage 6: Talker Generation (temp=0, max={max_tokens}) ===")
        
        # We need to hook into the generation to capture per-step data
        # Use the talker directly with our constructed prefill
        
        talker = model.talker
        
        # Capture hidden states and logits per step
        step_hidden_states = []
        step_logits = []
        step_tokens = []
        
        # Run Talker with custom generation to capture intermediates
        # Use temperature=0 for deterministic greedy sampling
        talker_outputs = talker.generate(
            inputs_embeds=prefill_embeds,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            max_new_tokens=max_tokens,
            do_sample=False,  # Greedy for deterministic
            temperature=1.0,  # Ignored when do_sample=False
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        
        print(f"Talker generation complete")
        print(f"Output sequences shape: {talker_outputs.sequences.shape}")
        
        # Extract generated tokens (excluding prefill position)
        generated_tokens = talker_outputs.sequences[0].tolist()
        print(f"Generated tokens: {generated_tokens[:20]}...")
        
        np.save(output_path / "06_generated_tokens.npy", np.array(generated_tokens))
        
        # Extract hidden states per step if available
        if hasattr(talker_outputs, 'hidden_states') and talker_outputs.hidden_states:
            print(f"\nExtracting per-step hidden states...")
            for step_idx, hs in enumerate(talker_outputs.hidden_states):
                if hs is not None and len(hs) > 0:
                    # hs is a tuple of layer hidden states, take last layer
                    last_layer_hs = hs[-1]  # [batch, seq, hidden]
                    # Get the last token's hidden state
                    if last_layer_hs.shape[1] > 0:
                        step_hs = last_layer_hs[:, -1, :]  # [batch, hidden]
                        save_tensor(step_hs.squeeze(0), output_path, f"07_step{step_idx:03d}_hidden")
                        step_hidden_states.append(step_hs.squeeze(0).cpu().numpy())
        
        # ========================================
        # Stage 7: Full Pipeline for Audio Output
        # ========================================
        print(f"\n=== Stage 7: Full Pipeline (for audio verification) ===")
        
        full_outputs = model.generate(
            input_ids=input_ids,
            speaker=speaker,
            return_audio=True,
            thinker_max_new_tokens=50,
            talker_max_new_tokens=max_tokens,
            talker_do_sample=False,  # Greedy
        )
        
        if isinstance(full_outputs, tuple) and len(full_outputs) > 1:
            audio = full_outputs[1]
            if audio is not None:
                print(f"Audio shape: {audio.shape}")
                print(f"Audio duration: {audio.shape[-1] / 24000:.2f}s")
                
                # Save audio
                import scipy.io.wavfile as wavfile
                audio_np = audio.cpu().numpy().squeeze()
                audio_np = np.clip(audio_np, -1, 1)
                wavfile.write(str(output_path / "output.wav"), 24000, 
                            (audio_np * 32767).astype(np.int16))
                print(f"Saved output.wav")
    
    # Save metadata
    metadata = {
        "text": text,
        "speaker": speaker,
        "max_tokens": max_tokens,
        "n_text_tokens": n_text,
        "n_trailing": trailing_text_hidden.shape[1],
        "n_generated": len(generated_tokens),
        "codec_nothink_id": codec_nothink_id,
        "codec_think_bos_id": codec_think_bos_id,
        "codec_think_eos_id": codec_think_eos_id,
        "codec_pad_id": codec_pad_id,
        "codec_bos_id": codec_bos_id,
        "speaker_id": speaker_id,
        "temperature": 0,  # Greedy
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n=== Done ===")
    print(f"Output directory: {output_path}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Debug HuggingFace Talker generation")
    parser.add_argument("--text", "-t", required=True, help="Input text")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--model", "-m", default="/models/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--speaker", "-s", default="Ethan")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--device", "-d", default="cuda")
    
    args = parser.parse_args()
    
    try:
        debug_talker_generation(
            text=args.text,
            output_dir=args.output,
            model_path=args.model,
            speaker=args.speaker,
            max_tokens=args.max_tokens,
            device=args.device,
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
