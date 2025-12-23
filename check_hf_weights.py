#!/usr/bin/env python3
"""Check HuggingFace model weight magnitudes for Code2Wav."""

from safetensors import safe_open
import numpy as np

# Path to the original safetensors file
safetensor_path = "/home/trevor/models/Qwen/Qwen3-Omni-30B-A3B-Instruct/model-00015-of-00015.safetensors"

with safe_open(safetensor_path, framework="numpy") as f:
    print("=== ConvNeXt Upsample Block Weights ===")

    # Check upsample block weights
    for i in range(2):
        prefix = f"code2wav.upsample.{i}"

        # Transposed conv
        conv_name = f"{prefix}.conv.weight"
        if conv_name in f.keys():
            w = f.get_tensor(conv_name)
            print(f"\n{conv_name}:")
            print(f"  shape: {w.shape}, dtype: {w.dtype}")
            print(f"  min: {w.min():.6f}, max: {w.max():.6f}, mean: {w.mean():.6f}, std: {w.std():.6f}")

        # pwconv1
        pw1_name = f"{prefix}.pwconv1.weight"
        if pw1_name in f.keys():
            w = f.get_tensor(pw1_name)
            print(f"\n{pw1_name}:")
            print(f"  shape: {w.shape}, dtype: {w.dtype}")
            print(f"  min: {w.min():.6f}, max: {w.max():.6f}, mean: {w.mean():.6f}, std: {w.std():.6f}")

        # pwconv2
        pw2_name = f"{prefix}.pwconv2.weight"
        if pw2_name in f.keys():
            w = f.get_tensor(pw2_name)
            print(f"\n{pw2_name}:")
            print(f"  shape: {w.shape}, dtype: {w.dtype}")
            print(f"  min: {w.min():.6f}, max: {w.max():.6f}, mean: {w.mean():.6f}, std: {w.std():.6f}")

        # gamma (layer scale)
        gamma_name = f"{prefix}.gamma"
        if gamma_name in f.keys():
            w = f.get_tensor(gamma_name)
            print(f"\n{gamma_name}:")
            print(f"  shape: {w.shape}, dtype: {w.dtype}")
            print(f"  min: {w.min():.6f}, max: {w.max():.6f}, mean: {w.mean():.6f}, std: {w.std():.6f}")

    print("\n=== Pre-transformer Output Norm ===")
    norm_name = "code2wav.pre_transformer.output_norm.weight"
    if norm_name in f.keys():
        w = f.get_tensor(norm_name)
        print(f"\n{norm_name}:")
        print(f"  shape: {w.shape}, dtype: {w.dtype}")
        print(f"  min: {w.min():.6f}, max: {w.max():.6f}, mean: {w.mean():.6f}, std: {w.std():.6f}")
