#!/usr/bin/env python3
"""
Compare MoE expert weights between GGUF and HuggingFace for layer 2.
This helps debug the massive std divergence in ffn_moe_out.
"""

import struct
import numpy as np
from pathlib import Path


def load_cpp_tensor(path: Path) -> np.ndarray:
    """Load C++ tensor with dimension header."""
    with open(path, 'rb') as f:
        ndims = struct.unpack('<I', f.read(4))[0]
        shape = []
        for _ in range(ndims):
            shape.append(struct.unpack('<I', f.read(4))[0])
        data = np.frombuffer(f.read(), dtype='<f4')
        return data.reshape(shape)


def main():
    cpp_dir = Path("/models/debug/cpp_talker")
    hf_dir = Path("/models/debug/hf_talker")

    print("=" * 80)
    print("Layer 2 MoE Intermediate Comparison")
    print("=" * 80)

    # Compare ffn_inp (input to FFN after attention)
    print("\n--- ffn_inp (after attention) ---")
    for layer_idx in [1, 2]:
        cpp_path = cpp_dir / f"ffn_inp_layer{layer_idx}.bin"
        hf_path = hf_dir / f"ffn_inp_layer{layer_idx}.npy"

        if cpp_path.exists():
            cpp = load_cpp_tensor(cpp_path)
            print(f"C++ Layer {layer_idx} ffn_inp: shape={cpp.shape}, mean={cpp.mean():.6f}, std={cpp.std():.6f}")
            # Show last token (position 8)
            last_token = cpp[-1] if cpp.ndim == 2 else cpp
            print(f"  Last token: mean={last_token.mean():.6f}, std={last_token.std():.6f}, first 5: {last_token[:5]}")

        if hf_path.exists():
            hf = np.load(hf_path)
            print(f"HF  Layer {layer_idx} ffn_inp: shape={hf.shape}, mean={hf.mean():.6f}, std={hf.std():.6f}")
            print(f"  First 5: {hf[:5]}")

        if cpp_path.exists() and hf_path.exists():
            cpp_last = cpp[-1] if cpp.ndim == 2 else cpp
            corr = np.corrcoef(cpp_last.flatten(), hf.flatten())[0, 1]
            print(f"  Correlation: {corr:.6f}")

    # Compare ffn_norm (after FFN norm)
    print("\n--- ffn_norm (after post_attention_layernorm) ---")
    for layer_idx in [1, 2]:
        cpp_path = cpp_dir / f"ffn_norm_layer{layer_idx}.bin"
        hf_path = hf_dir / f"ffn_norm_layer{layer_idx}.npy"

        if cpp_path.exists():
            cpp = load_cpp_tensor(cpp_path)
            last_token = cpp[-1] if cpp.ndim == 2 else cpp
            print(f"C++ Layer {layer_idx} ffn_norm: mean={last_token.mean():.6f}, std={last_token.std():.6f}")

        if hf_path.exists():
            hf = np.load(hf_path)
            print(f"HF  Layer {layer_idx} ffn_norm: mean={hf.mean():.6f}, std={hf.std():.6f}")

        if cpp_path.exists() and hf_path.exists():
            cpp_last = cpp[-1] if cpp.ndim == 2 else cpp
            corr = np.corrcoef(cpp_last.flatten(), hf.flatten())[0, 1]
            print(f"  Correlation: {corr:.6f}")

    # Compare ffn_moe_out (MoE output - THE PROBLEM AREA)
    print("\n--- ffn_moe_out (MoE expert output - PROBLEM AREA) ---")
    for layer_idx in [1, 2]:
        cpp_path = cpp_dir / f"ffn_moe_out_layer{layer_idx}.bin"
        hf_path = hf_dir / f"ffn_moe_out_layer{layer_idx}.npy"

        if cpp_path.exists():
            cpp = load_cpp_tensor(cpp_path)
            last_token = cpp[-1] if cpp.ndim == 2 else cpp
            print(f"C++ Layer {layer_idx} ffn_moe_out: mean={last_token.mean():.6f}, std={last_token.std():.6f}")
            print(f"  min={last_token.min():.4f}, max={last_token.max():.4f}")

        if hf_path.exists():
            hf = np.load(hf_path)
            print(f"HF  Layer {layer_idx} ffn_moe_out: mean={hf.mean():.6f}, std={hf.std():.6f}")
            print(f"  min={hf.min():.4f}, max={hf.max():.4f}")

        if cpp_path.exists() and hf_path.exists():
            cpp_last = cpp[-1] if cpp.ndim == 2 else cpp
            corr = np.corrcoef(cpp_last.flatten(), hf.flatten())[0, 1]
            print(f"  Correlation: {corr:.6f}")
            if corr < 0.9:
                print(f"  *** LOW CORRELATION - POSSIBLE BUG! ***")

    # Compare shared expert
    print("\n--- ffn_shexp (shared expert output) ---")
    for layer_idx in [1, 2]:
        cpp_path = cpp_dir / f"ffn_shexp_layer{layer_idx}.bin"
        hf_path = hf_dir / f"ffn_shexp_layer{layer_idx}.npy"

        if cpp_path.exists():
            cpp = load_cpp_tensor(cpp_path)
            last_token = cpp[-1] if cpp.ndim == 2 else cpp
            print(f"C++ Layer {layer_idx} ffn_shexp: mean={last_token.mean():.6f}, std={last_token.std():.6f}")

        if hf_path.exists():
            hf = np.load(hf_path)
            print(f"HF  Layer {layer_idx} ffn_shexp: mean={hf.mean():.6f}, std={hf.std():.6f}")

        if cpp_path.exists() and hf_path.exists():
            cpp_last = cpp[-1] if cpp.ndim == 2 else cpp
            corr = np.corrcoef(cpp_last.flatten(), hf.flatten())[0, 1]
            print(f"  Correlation: {corr:.6f}")

    print("\n" + "=" * 80)
    print("Summary: Compare the std values above.")
    print("If C++ Layer 2 ffn_moe_out std >> HF Layer 2 ffn_moe_out std,")
    print("the bug is in the C++ MoE expert computation (wrong weights or routing).")
    print("=" * 80)


if __name__ == "__main__":
    main()
