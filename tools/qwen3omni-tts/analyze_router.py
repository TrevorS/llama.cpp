#!/usr/bin/env python3
"""Analyze C++ router logits to see which experts are being selected."""

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

    print("=" * 80)
    print("C++ Router Analysis for Layers 1-2")
    print("=" * 80)

    for layer_idx in [1, 2]:
        print(f"\n=== Layer {layer_idx} ===")

        # Load router logits
        logits_path = cpp_dir / f"ffn_moe_logits_layer{layer_idx}.bin"
        probs_path = cpp_dir / f"ffn_moe_probs_layer{layer_idx}.bin"

        if logits_path.exists():
            logits = load_cpp_tensor(logits_path)
            print(f"\nRouter logits shape: {logits.shape}")
            print(f"  mean={logits.mean():.4f}, std={logits.std():.4f}")
            print(f"  min={logits.min():.4f}, max={logits.max():.4f}")

            # Get last token's logits
            if logits.ndim == 2:
                last_logits = logits[-1]  # (128,)
            else:
                last_logits = logits  # Already 1D

            print(f"\nLast token router logits:")
            print(f"  Shape: {last_logits.shape}")

            # Compute softmax
            exp_logits = np.exp(last_logits - last_logits.max())
            probs = exp_logits / exp_logits.sum()

            # Top-6 experts (Talker uses top_k=6)
            top_k = 6
            top_indices = np.argsort(last_logits)[-top_k:][::-1]
            top_probs = probs[top_indices]

            print(f"\n  Top-{top_k} experts by logits:")
            for i, (idx, logit, prob) in enumerate(zip(top_indices, last_logits[top_indices], top_probs)):
                print(f"    {i+1}. Expert {idx}: logit={logit:.4f}, prob={prob:.4f}")

            # Normalize to top-k sum = 1
            top_k_probs = top_probs / top_probs.sum()
            print(f"\n  Top-{top_k} weights (normalized to sum=1):")
            for i, (idx, w) in enumerate(zip(top_indices, top_k_probs)):
                print(f"    {i+1}. Expert {idx}: weight={w:.4f}")

        if probs_path.exists():
            saved_probs = load_cpp_tensor(probs_path)
            print(f"\nSaved probs shape: {saved_probs.shape}")
            if saved_probs.ndim == 2:
                last_probs = saved_probs[-1]
            else:
                last_probs = saved_probs
            print(f"  Probs sum: {last_probs.sum():.6f}")
            print(f"  Top-6 saved probs: {np.sort(last_probs)[-6:][::-1]}")

        # Load ffn_norm to see input to router
        norm_path = cpp_dir / f"ffn_norm_layer{layer_idx}.bin"
        if norm_path.exists():
            norm = load_cpp_tensor(norm_path)
            if norm.ndim == 2:
                last_norm = norm[-1]
            else:
                last_norm = norm
            print(f"\nffn_norm (router input) last token:")
            print(f"  mean={last_norm.mean():.6f}, std={last_norm.std():.6f}")
            print(f"  first 5: {last_norm[:5]}")

        # Load ffn_moe_out
        moe_out_path = cpp_dir / f"ffn_moe_out_layer{layer_idx}.bin"
        if moe_out_path.exists():
            moe_out = load_cpp_tensor(moe_out_path)
            if moe_out.ndim == 2:
                last_moe = moe_out[-1]
            else:
                last_moe = moe_out
            print(f"\nffn_moe_out (MoE output) last token:")
            print(f"  mean={last_moe.mean():.6f}, std={last_moe.std():.6f}")
            print(f"  first 5: {last_moe[:5]}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
