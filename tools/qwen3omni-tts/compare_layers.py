#!/usr/bin/env python3
"""Compare HF and C++ layer-by-layer outputs."""

import struct
import numpy as np
from pathlib import Path


def load_bin(path: Path) -> np.ndarray:
    """Load binary tensor with dimension header."""
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

    print("=" * 70)
    print("Layer-by-Layer Comparison: C++ vs HuggingFace")
    print("=" * 70)

    # Compare prefill embeddings first
    print("\n--- PREFILL EMBEDDINGS ---")
    cpp_prefill = load_bin(cpp_dir / "prefill_embeds.bin")
    hf_prefill = np.load(hf_dir / "prefill_embeds.npy")

    print(f"C++ prefill shape: {cpp_prefill.shape}")
    print(f"HF  prefill shape: {hf_prefill.shape}")

    prefill_corr = np.corrcoef(cpp_prefill.flatten(), hf_prefill.flatten())[0, 1]
    print(f"Prefill correlation: {prefill_corr:.6f}")

    if prefill_corr < 0.99:
        print("WARNING: Prefill embeddings do not match!")
        print(f"  C++ first 4: {cpp_prefill.flatten()[:4]}")
        print(f"  HF  first 4: {hf_prefill.flatten()[:4]}")
        return

    print("Prefill embeddings MATCH")

    # Compare per-layer outputs
    print("\n--- LAYER-BY-LAYER HIDDEN STATES ---")
    print(f"{'Layer':>8} | {'Correlation':>12} | {'C++ std':>10} | {'HF std':>10} | {'C++ first':>12} | {'HF first':>12}")
    print("-" * 75)

    for layer_idx in range(20):
        cpp_path = cpp_dir / f"hidden_layer{layer_idx}.bin"
        hf_path = hf_dir / f"hidden_layer{layer_idx}.npy"

        if cpp_path.exists() and hf_path.exists():
            cpp_layer = load_bin(cpp_path)
            hf_layer = np.load(hf_path)
            corr = np.corrcoef(cpp_layer.flatten(), hf_layer.flatten())[0, 1]
            print(f"{layer_idx:>8} | {corr:>12.6f} | {cpp_layer.std():>10.4f} | {hf_layer.std():>10.4f} | {cpp_layer.flatten()[0]:>12.4f} | {hf_layer.flatten()[0]:>12.4f}")
        elif hf_path.exists():
            hf_layer = np.load(hf_path)
            print(f"{layer_idx:>8} | {'N/A (C++ missing)':>12} | {'N/A':>10} | {hf_layer.std():>10.4f} | {'N/A':>12} | {hf_layer.flatten()[0]:>12.4f}")
        else:
            print(f"{layer_idx:>8} | {'N/A':>12}")

    # Compare final hidden state
    print("\n--- FINAL HIDDEN STATE (after norm) ---")
    cpp_hidden = load_bin(cpp_dir / "hidden_after_norm.bin")
    hf_hidden = np.load(hf_dir / "hidden_after_norm.npy")

    print(f"C++ shape: {cpp_hidden.shape}, HF shape: {hf_hidden.shape}")
    corr = np.corrcoef(cpp_hidden.flatten(), hf_hidden.flatten())[0, 1]
    print(f"Correlation: {corr:.6f}")
    print(f"C++ stats: mean={cpp_hidden.mean():.4f}, std={cpp_hidden.std():.4f}")
    print(f"HF  stats: mean={hf_hidden.mean():.4f}, std={hf_hidden.std():.4f}")
    print(f"C++ first 8: {cpp_hidden.flatten()[:8]}")
    print(f"HF  first 8: {hf_hidden.flatten()[:8]}")

    # Compare logits
    print("\n--- LOGITS ---")
    cpp_logits = load_bin(cpp_dir / "prefill_logits.bin")
    hf_logits = np.load(hf_dir / "prefill_logits.npy")

    corr = np.corrcoef(cpp_logits.flatten(), hf_logits.flatten())[0, 1]
    print(f"Logits correlation: {corr:.6f}")
    print(f"C++ top token: {np.argmax(cpp_logits)} (logit={cpp_logits.max():.4f})")
    print(f"HF  top token: {np.argmax(hf_logits)} (logit={hf_logits.max():.4f})")


if __name__ == "__main__":
    main()
