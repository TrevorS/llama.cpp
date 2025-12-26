#!/usr/bin/env python3
"""Compare LayerScale values between GGUF and HuggingFace."""

import numpy as np
import os
from pathlib import Path
from safetensors import safe_open
from gguf import GGUFReader


def main():
    model_path = "/models/Qwen3-Omni-30B-A3B-Instruct"
    gguf_path = "/models/qwen3-omni-30b-talker-f16-v4.gguf"

    print("=" * 60)
    print("LayerScale Comparison: GGUF vs HuggingFace")
    print("=" * 60)

    # Load from HuggingFace
    print("\n=== Loading from HuggingFace ===\n")
    hf_scales = {}
    for f in sorted(os.listdir(model_path)):
        if f.endswith('.safetensors'):
            path = os.path.join(model_path, f)
            with safe_open(path, framework='pt') as sf:
                for name in sf.keys():
                    if 'code2wav.pre_transformer' in name and 'layer_scale' in name:
                        tensor = sf.get_tensor(name).float().numpy()
                        hf_scales[name] = tensor
                        print(f"{name}: mean={tensor.mean():.6f}, std={tensor.std():.6f}")

    # Load from GGUF
    print("\n=== Loading from GGUF ===\n")
    reader = GGUFReader(gguf_path)
    gguf_scales = {}
    for tensor in reader.tensors:
        if "attn_scale" in tensor.name or "ffn_scale" in tensor.name:
            if "code2wav" in tensor.name or "c2w" in tensor.name:
                data = tensor.data.astype(np.float32)
                # Reshape according to tensor shape
                if len(tensor.shape) == 1:
                    data = data.reshape(tensor.shape[0])
                gguf_scales[tensor.name] = data
                print(f"{tensor.name}: mean={data.mean():.6f}, std={data.std():.6f}")

    # Compare
    print("\n=== Comparison ===\n")
    print(f"{'Layer':<8} | {'Type':<10} | {'HF mean':<12} | {'GGUF mean':<12} | {'Diff':<12} | {'Corr':<8}")
    print("-" * 75)

    # Map GGUF names to HF names
    for layer in range(8):
        for scale_type in ['attn', 'ffn']:
            # HF name pattern
            if scale_type == 'attn':
                hf_name = f"code2wav.pre_transformer.layers.{layer}.self_attn_layer_scale.scale"
            else:
                hf_name = f"code2wav.pre_transformer.layers.{layer}.mlp_layer_scale.scale"

            # GGUF name pattern
            gguf_name = f"code2wav.pre.blk.{layer}.{scale_type}_scale"

            if hf_name in hf_scales and gguf_name in gguf_scales:
                hf_data = hf_scales[hf_name]
                gguf_data = gguf_scales[gguf_name]

                diff = np.abs(hf_data - gguf_data).max()
                corr = np.corrcoef(hf_data.flatten(), gguf_data.flatten())[0, 1]

                print(f"{layer:<8} | {scale_type:<10} | {hf_data.mean():.6f}     | {gguf_data.mean():.6f}     | {diff:.6f}     | {corr:.6f}")
            else:
                # Check if GGUF has the scale
                found_gguf = any(f"blk.{layer}" in n and scale_type in n for n in gguf_scales.keys())
                print(f"{layer:<8} | {scale_type:<10} | {'found' if hf_name in hf_scales else 'missing':<12} | {'found' if found_gguf else 'missing':<12}")

    # Summary
    print("\n=== Summary ===\n")
    total_hf = len(hf_scales)
    total_gguf = len(gguf_scales)
    print(f"Total HF layer scales: {total_hf}")
    print(f"Total GGUF layer scales: {total_gguf}")

    # Check if any are missing
    if total_gguf < total_hf:
        print("\nWARNING: Some layer scales are missing in GGUF!")
        print("This could cause the std ratio divergence.")


if __name__ == "__main__":
    main()
