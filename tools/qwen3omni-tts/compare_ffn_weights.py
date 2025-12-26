#!/usr/bin/env python3
"""Compare FFN weights between GGUF and HuggingFace for all layers."""

import numpy as np
import struct
from pathlib import Path


def load_hf_weights(model_path: str):
    """Load HuggingFace Code2Wav pre-transformer weights."""
    import safetensors.torch
    import torch

    model_dir = Path(model_path)
    weights = {}

    for sf_file in sorted(model_dir.glob("*.safetensors")):
        print(f"  Loading {sf_file.name}...")
        sd = safetensors.torch.load_file(str(sf_file))
        for k, v in sd.items():
            if k.startswith("code2wav.pre_transformer."):
                # Extract just the pre_transformer part
                new_key = k.replace("code2wav.pre_transformer.", "")
                weights[new_key] = v.float().numpy()

    return weights


def compare_weights(gguf_path: str, hf_model_path: str):
    """Compare FFN weights between GGUF and HuggingFace."""
    from gguf import GGUFReader

    print("Loading GGUF model...")
    reader = GGUFReader(gguf_path)

    print("Loading HuggingFace weights...")
    hf_weights = load_hf_weights(hf_model_path)

    print("\n" + "=" * 80)
    print("FFN Weight Comparison: GGUF vs HuggingFace")
    print("=" * 80)

    # Map GGUF names to HF names for FFN
    # GGUF: code2wav.pre.blk.{i}.ffn_gate.weight, ffn_up.weight, ffn_down.weight
    # HF: model.layers.{i}.mlp.gate_proj.weight, up_proj.weight, down_proj.weight

    print(f"\n{'Layer':<8} | {'Component':<12} | {'GGUF shape':<20} | {'HF shape':<20} | {'Corr':>8} | {'Status':<10}")
    print("-" * 100)

    for i in range(8):  # 8 pre-transformer layers
        for component, gguf_suffix, hf_suffix in [
            ("gate", "ffn_gate.weight", "mlp.gate_proj.weight"),
            ("up", "ffn_up.weight", "mlp.up_proj.weight"),
            ("down", "ffn_down.weight", "mlp.down_proj.weight"),
        ]:
            gguf_name = f"code2wav.pre.blk.{i}.{gguf_suffix}"
            hf_name = f"model.layers.{i}.{hf_suffix}"

            # Find in GGUF
            gguf_tensor = None
            for tensor in reader.tensors:
                if tensor.name == gguf_name:
                    gguf_tensor = tensor
                    break

            if gguf_tensor is None:
                print(f"Layer {i:<2} | {component:<12} | {'GGUF missing':<20}")
                continue

            if hf_name not in hf_weights:
                print(f"Layer {i:<2} | {component:<12} | {str(gguf_tensor.shape):<20} | {'HF missing':<20}")
                continue

            # Get data
            gguf_data = gguf_tensor.data.astype(np.float32)
            hf_data = hf_weights[hf_name]

            # GGUF uses reversed dimensions
            gguf_shape = tuple(gguf_tensor.shape)
            hf_shape = hf_data.shape

            # Reshape GGUF to match HF
            # GGUF stores as [out_features, in_features] in reversed order
            gguf_reshaped = gguf_data.reshape(gguf_shape[::-1], order='F')

            # Compare
            if gguf_reshaped.shape == hf_data.shape:
                corr = np.corrcoef(gguf_reshaped.flatten(), hf_data.flatten())[0, 1]
                status = "PERFECT" if corr > 0.9999 else ("OK" if corr > 0.99 else "MISMATCH")
                print(f"Layer {i:<2} | {component:<12} | {str(gguf_shape):<20} | {str(hf_shape):<20} | {corr:>8.6f} | {status:<10}")
            else:
                # Try transpose
                if gguf_reshaped.T.shape == hf_data.shape:
                    corr = np.corrcoef(gguf_reshaped.T.flatten(), hf_data.flatten())[0, 1]
                    status = "PERFECT" if corr > 0.9999 else ("OK" if corr > 0.99 else "MISMATCH")
                    print(f"Layer {i:<2} | {component:<12} | {str(gguf_shape):<20}T| {str(hf_shape):<20} | {corr:>8.6f} | {status:<10}")
                else:
                    print(f"Layer {i:<2} | {component:<12} | {str(gguf_shape):<20} | {str(hf_shape):<20} | SHAPE MISMATCH")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", default="/models/qwen3-omni-30b-talker-f16-v4.gguf")
    parser.add_argument("--hf", default="/models/Qwen3-Omni-30B-A3B-Instruct")
    args = parser.parse_args()

    compare_weights(args.gguf, args.hf)


if __name__ == "__main__":
    main()
