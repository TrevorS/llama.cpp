#!/usr/bin/env python3
"""
Debug script to dump intermediate tensors from HuggingFace Code2Wav.
Used to compare against GGML implementation for debugging.

Usage:
    python debug_hf_code2wav.py --tokens tokens.txt --output-dir /tmp/hf_tensors/

Tokens file format: 16 lines of space-separated integers (one line per codebook)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch


def load_tokens(token_file: str) -> torch.Tensor:
    """Load codec tokens from file.

    Format: 16 lines, each with space-separated token IDs.
    Returns tensor of shape [1, 16, seq_len]
    """
    lines = Path(token_file).read_text().strip().split('\n')
    if len(lines) != 16:
        raise ValueError(f"Expected 16 codebook lines, got {len(lines)}")

    tokens = []
    for line in lines:
        tokens.append([int(x) for x in line.strip().split()])

    # Verify all codebooks have same length
    seq_len = len(tokens[0])
    for i, t in enumerate(tokens):
        if len(t) != seq_len:
            raise ValueError(f"Codebook {i} has {len(t)} tokens, expected {seq_len}")

    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # [1, 16, seq_len]


def save_tensor(tensor: torch.Tensor, path: str, name: str):
    """Save tensor as numpy file."""
    arr = tensor.detach().cpu().numpy()
    filepath = os.path.join(path, f"{name}.npy")
    np.save(filepath, arr)
    print(f"  Saved {name}: shape={arr.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}")


def main():
    parser = argparse.ArgumentParser(description="Dump HuggingFace Code2Wav intermediate tensors")
    parser.add_argument("--tokens", required=True, help="Path to tokens file (16 lines, space-separated)")
    parser.add_argument("--output-dir", required=True, help="Output directory for tensor dumps")
    parser.add_argument("--model-path", default="/models/Qwen3-Omni-30B-A3B-Instruct",
                        help="Path to Qwen3-Omni model directory")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokens
    print(f"Loading tokens from {args.tokens}")
    codes = load_tokens(args.tokens)
    print(f"  Loaded codes: shape={codes.shape}")

    # Load full model to access code2wav
    print(f"\nLoading model from {args.model_path}")

    # Import here to avoid slow import if just checking help
    from transformers import AutoModelForSpeechSeq2Seq, AutoConfig

    # Load the full model with code2wav
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    # Load only code2wav weights for efficiency
    print("  Loading code2wav weights...")
    import safetensors.torch
    from transformers.models.qwen3_omni_moe.modular_qwen3_omni_moe import Qwen3OmniMoeCode2Wav

    # Manually construct Code2Wav with required config patches
    code2wav_config = config.code2wav_config
    # Patch missing attrs that the transformer model expects
    code2wav_config.vocab_size = 1024 * 16 + 1  # 16 codebooks * 1024 vocab + 1
    code2wav_config.pad_token_id = 0
    # NOTE: Do NOT override hidden_act - model uses "silu" per checkpoint
    print(f"  Using hidden_act: {code2wav_config.hidden_act}")

    model = Qwen3OmniMoeCode2Wav(code2wav_config)

    # Load state dict (only code2wav weights)
    state_dict = {}
    model_dir = Path(args.model_path)
    for sf_file in sorted(model_dir.glob("*.safetensors")):
        print(f"  Loading {sf_file.name}...")
        sd = safetensors.torch.load_file(str(sf_file))
        for k, v in sd.items():
            if k.startswith("code2wav."):
                new_key = k[len("code2wav."):]
                state_dict[new_key] = v

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded state dict: {len(state_dict)} tensors")
    if missing:
        print(f"  WARNING: Missing keys: {len(missing)} keys")
    if unexpected:
        print(f"  WARNING: Unexpected keys: {unexpected[:5]}...")

    model.eval()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    codes = codes.to(model.device)

    print(f"\nRunning Code2Wav forward pass with tensor dumps...")

    with torch.no_grad():
        # Step 1: Codebook embedding + mean
        hidden = model.code_embedding(codes + model.code_offset)
        save_tensor(hidden, args.output_dir, "01_embd_raw")  # [1, 16, seq, hidden]

        hidden = hidden.mean(1)
        save_tensor(hidden, args.output_dir, "02_embd_mean")  # [1, seq, hidden]

        # Step 2: Pre-transformer (8 layers)
        pre_out = model.pre_transformer(inputs_embeds=hidden)
        hidden = pre_out.last_hidden_state
        save_tensor(hidden, args.output_dir, "03_pre_xfmr_out")  # [1, seq, hidden]

        # Step 3: Permute for convolutions
        hidden = hidden.permute(0, 2, 1)
        save_tensor(hidden, args.output_dir, "04_permuted")  # [1, hidden, seq]

        # Step 4: Upsample blocks (TransConv + ConvNeXt pairs)
        for i, blocks in enumerate(model.upsample):
            # First block is TransConv
            hidden = blocks[0](hidden)
            save_tensor(hidden, args.output_dir, f"05_up{i}_transconv")

            # Second block is ConvNeXt
            # Hook into ConvNeXt internals
            convnext = blocks[1]
            cnxt_input = hidden

            # dwconv
            cnxt_hidden = convnext.dwconv(cnxt_input)
            save_tensor(cnxt_hidden, args.output_dir, f"06_cnxt{i}_dwconv")

            # permute + norm
            cnxt_hidden = cnxt_hidden.permute(0, 2, 1)
            cnxt_hidden = convnext.norm(cnxt_hidden)
            save_tensor(cnxt_hidden.permute(0, 2, 1), args.output_dir, f"07_cnxt{i}_norm")

            # pwconv1
            cnxt_hidden = convnext.pwconv1(cnxt_hidden)
            save_tensor(cnxt_hidden.permute(0, 2, 1), args.output_dir, f"08_cnxt{i}_pw1")

            # GELU
            cnxt_hidden = convnext.act(cnxt_hidden)
            save_tensor(cnxt_hidden.permute(0, 2, 1), args.output_dir, f"09_cnxt{i}_gelu")

            # pwconv2
            cnxt_hidden = convnext.pwconv2(cnxt_hidden)
            save_tensor(cnxt_hidden.permute(0, 2, 1), args.output_dir, f"10_cnxt{i}_pw2")

            # gamma (LayerScale)
            cnxt_hidden = convnext.gamma * cnxt_hidden

            # permute back + residual
            cnxt_hidden = cnxt_hidden.permute(0, 2, 1)
            hidden = cnxt_input + cnxt_hidden
            save_tensor(hidden, args.output_dir, f"11_cnxt{i}_out")

        # Step 5: HiFi-GAN decoder
        wav = hidden
        for i, block in enumerate(model.decoder):
            wav = block(wav)
            save_tensor(wav, args.output_dir, f"12_dec{i}")

        # Step 6: Final clamp
        wav = wav.clamp(min=-1, max=1)
        save_tensor(wav, args.output_dir, "13_final_wav")

    print(f"\nDone! Tensors saved to {args.output_dir}")
    print(f"Final wav shape: {wav.shape}, samples: {wav.shape[-1]}")


if __name__ == "__main__":
    main()
