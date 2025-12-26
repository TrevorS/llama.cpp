#!/usr/bin/env python3
"""
Debug script to dump per-layer intermediate tensors from HuggingFace pre-transformer.
Used to compare against GGML implementation for debugging the std ratio issue.

Usage:
    python debug_hf_pretrans_layers.py --tokens tokens.txt --output-dir /tmp/hf_layers/
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch


def load_tokens(token_file: str) -> torch.Tensor:
    """Load codec tokens from file."""
    lines = Path(token_file).read_text().strip().split('\n')
    if len(lines) != 16:
        raise ValueError(f"Expected 16 codebook lines, got {len(lines)}")

    tokens = []
    for line in lines:
        tokens.append([int(x) for x in line.strip().split()])

    seq_len = len(tokens[0])
    for i, t in enumerate(tokens):
        if len(t) != seq_len:
            raise ValueError(f"Codebook {i} has {len(t)} tokens, expected {seq_len}")

    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)


def save_tensor(tensor: torch.Tensor, path: str, name: str):
    """Save tensor as numpy file."""
    arr = tensor.detach().float().cpu().numpy()
    filepath = os.path.join(path, f"{name}.npy")
    np.save(filepath, arr)
    print(f"  {name}: shape={arr.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default="/models/Qwen3-Omni-30B-A3B-Instruct")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokens
    print(f"Loading tokens from {args.tokens}")
    codes = load_tokens(args.tokens)
    print(f"  Loaded codes: shape={codes.shape}")

    # Load model
    print(f"\nLoading model from {args.model_path}")

    from transformers import AutoConfig
    import safetensors.torch
    from transformers.models.qwen3_omni_moe.modular_qwen3_omni_moe import Qwen3OmniMoeCode2Wav

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    code2wav_config = config.code2wav_config
    code2wav_config.vocab_size = 1024 * 16 + 1
    code2wav_config.pad_token_id = 0
    # Note: hidden_act should be "silu" per the checkpoint, NOT "gelu"
    # The model was trained with SiLU activation
    print(f"  Using hidden_act: {code2wav_config.hidden_act}")

    model = Qwen3OmniMoeCode2Wav(code2wav_config)

    # Load state dict
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
    print(f"  Loaded: {len(state_dict)} tensors, {len(missing)} missing")

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    codes = codes.to(device)

    print(f"\nRunning pre-transformer with per-layer dumps using hooks...")

    captured_tensors = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            captured_tensors[name] = out.clone().detach()
        return hook

    with torch.no_grad():
        # Step 1: Codebook embedding + mean
        hidden = model.code_embedding(codes + model.code_offset)
        hidden = hidden.mean(1)
        save_tensor(hidden, args.output_dir, "00_embd_mean")

        # Get the pre-transformer model
        pre_trans = model.pre_transformer
        model_layers = pre_trans.model.layers if hasattr(pre_trans, 'model') else pre_trans.layers

        print(f"\n  Pre-transformer has {len(model_layers)} layers")

        # Register hooks for each layer and sub-component
        hooks = []
        for i, layer in enumerate(model_layers):
            # Hook the entire layer output
            hooks.append(layer.register_forward_hook(make_hook(f"layer{i}_output")))

            # Hook sub-components
            hooks.append(layer.input_layernorm.register_forward_hook(make_hook(f"layer{i}_attn_norm")))
            hooks.append(layer.self_attn.register_forward_hook(make_hook(f"layer{i}_attn_out")))
            if hasattr(layer, 'self_attn_layer_scale'):
                hooks.append(layer.self_attn_layer_scale.register_forward_hook(make_hook(f"layer{i}_attn_scaled")))
            hooks.append(layer.post_attention_layernorm.register_forward_hook(make_hook(f"layer{i}_ffn_norm")))
            hooks.append(layer.mlp.register_forward_hook(make_hook(f"layer{i}_mlp_out")))
            if hasattr(layer, 'mlp_layer_scale'):
                hooks.append(layer.mlp_layer_scale.register_forward_hook(make_hook(f"layer{i}_mlp_scaled")))

            # For layer 7, also hook FFN sub-components
            if i == 7:
                hooks.append(layer.mlp.gate_proj.register_forward_hook(make_hook(f"layer{i}_ffn_gate_proj")))
                hooks.append(layer.mlp.up_proj.register_forward_hook(make_hook(f"layer{i}_ffn_up_proj")))
                hooks.append(layer.mlp.down_proj.register_forward_hook(make_hook(f"layer{i}_ffn_down_proj")))

        # Hook the final norm
        final_norm = pre_trans.model.norm if hasattr(pre_trans, 'model') else pre_trans.norm
        hooks.append(final_norm.register_forward_hook(make_hook("final_norm")))

        # Run the full forward pass
        pre_out = pre_trans(inputs_embeds=hidden)
        final_hidden = pre_out.last_hidden_state

        # Remove hooks
        for h in hooks:
            h.remove()

        # Save all captured tensors
        print("\n  Captured tensors:")
        for name in sorted(captured_tensors.keys()):
            save_tensor(captured_tensors[name], args.output_dir, name)

        # Save final output
        save_tensor(final_hidden, args.output_dir, "final_output")
        print(f"\n  Final output std: {final_hidden.std():.6f}")

    print(f"\nDone! Tensors saved to {args.output_dir}")


if __name__ == "__main__":
    main()
