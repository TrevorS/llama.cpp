#!/usr/bin/env python3
"""Compare Talker layer outputs between C++ and HuggingFace to find divergence point."""

import struct
import numpy as np
from pathlib import Path


def load_cpp_tensor(path: Path) -> np.ndarray:
    """Load C++ tensor with dimension header.

    Format: [ndims: uint32] [dim0: uint32] [dim1: uint32] ... [data: float32]
    """
    with open(path, 'rb') as f:
        ndims = struct.unpack('<I', f.read(4))[0]
        shape = []
        for _ in range(ndims):
            shape.append(struct.unpack('<I', f.read(4))[0])
        data = np.frombuffer(f.read(), dtype='<f4')
        return data.reshape(shape)


def load_hf_tensor(path: Path) -> np.ndarray:
    """Load HF .npy tensor, removing batch dim if present."""
    arr = np.load(path)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def main():
    cpp_dir = Path("/models/debug/cpp_talker")
    hf_dir = Path("/models/debug/hf_talker")

    print("=" * 80)
    print("Talker Layer-by-Layer Comparison: C++ vs HuggingFace")
    print("=" * 80)

    # Check what files exist
    print("\n--- Available Files ---")
    print(f"C++ dir: {cpp_dir}")
    for f in sorted(cpp_dir.glob("*.bin")):
        size = f.stat().st_size
        print(f"  {f.name}: {size} bytes")

    print(f"\nHF dir: {hf_dir}")
    for f in sorted(hf_dir.glob("*.npy")):
        arr = np.load(f)
        print(f"  {f.name}: shape={arr.shape}")

    # Compare prefill embeddings first
    print("\n" + "=" * 80)
    print("INPUT VERIFICATION: Prefill Embeddings")
    print("=" * 80)

    cpp_prefill_path = cpp_dir / "prefill_embeds.bin"
    hf_prefill_path = hf_dir / "prefill_embeds.npy"

    if cpp_prefill_path.exists() and hf_prefill_path.exists():
        cpp_prefill = load_cpp_tensor(cpp_prefill_path)
        hf_prefill = load_hf_tensor(hf_prefill_path)

        print(f"C++ shape: {cpp_prefill.shape}")
        print(f"HF  shape: {hf_prefill.shape}")

        # Ensure shapes match (may need transpose)
        if cpp_prefill.shape != hf_prefill.shape:
            if cpp_prefill.T.shape == hf_prefill.shape:
                cpp_prefill = cpp_prefill.T
            elif cpp_prefill.shape == hf_prefill.T.shape:
                hf_prefill = hf_prefill.T

        if cpp_prefill.shape == hf_prefill.shape:
            corr = np.corrcoef(cpp_prefill.flatten(), hf_prefill.flatten())[0, 1]
            print(f"Correlation: {corr:.6f}")
            if corr > 0.999:
                print("Prefill embeddings: VERIFIED ✓")
            else:
                print("Prefill embeddings: MISMATCH ✗")
                print(f"  C++ first 4: {cpp_prefill.flatten()[:4]}")
                print(f"  HF  first 4: {hf_prefill.flatten()[:4]}")
        else:
            print(f"Shape mismatch after transpose attempt!")
    else:
        print("Missing prefill files!")

    # Compare layer-by-layer
    print("\n" + "=" * 80)
    print("LAYER-BY-LAYER COMPARISON")
    print("=" * 80)
    print(f"{'Layer':>8} | {'C++ std':>12} | {'HF std':>12} | {'Correlation':>12} | {'Status':>10}")
    print("-" * 70)

    first_divergence = None

    for layer_idx in range(20):
        cpp_path = cpp_dir / f"hidden_layer{layer_idx}.bin"
        hf_path = hf_dir / f"hidden_layer{layer_idx}.npy"

        if not cpp_path.exists():
            print(f"{layer_idx:>8} | {'(C++ missing)':>12} | {'':>12} | {'N/A':>12} | {'MISSING':>10}")
            continue

        if not hf_path.exists():
            print(f"{layer_idx:>8} | {'':>12} | {'(HF missing)':>12} | {'N/A':>12} | {'MISSING':>10}")
            continue

        cpp_layer = load_cpp_tensor(cpp_path)
        hf_layer = load_hf_tensor(hf_path)

        # Handle shape differences
        # HF may save only last token (1024,) while C++ saves all tokens (9, 1024)
        if cpp_layer.ndim == 2 and hf_layer.ndim == 1:
            # Extract last token from C++ to match HF
            cpp_layer = cpp_layer[-1]
        elif cpp_layer.ndim == 1 and hf_layer.ndim == 2:
            # Extract last token from HF to match C++
            hf_layer = hf_layer[-1]

        if cpp_layer.shape != hf_layer.shape:
            if cpp_layer.T.shape == hf_layer.shape:
                cpp_layer = cpp_layer.T
            elif cpp_layer.shape == hf_layer.T.shape:
                hf_layer = hf_layer.T

        if cpp_layer.shape != hf_layer.shape:
            print(f"{layer_idx:>8} | Shape mismatch: C++ {cpp_layer.shape} vs HF {hf_layer.shape}")
            continue

        corr = np.corrcoef(cpp_layer.flatten(), hf_layer.flatten())[0, 1]
        cpp_std = cpp_layer.std()
        hf_std = hf_layer.std()

        status = "✓" if corr > 0.99 else "✗ DIVERGE"
        print(f"{layer_idx:>8} | {cpp_std:>12.6f} | {hf_std:>12.6f} | {corr:>12.6f} | {status:>10}")

        if corr < 0.99 and first_divergence is None:
            first_divergence = layer_idx

    # Compare final hidden state
    print("\n" + "-" * 70)
    cpp_norm_path = cpp_dir / "hidden_after_norm.bin"
    hf_norm_path = hf_dir / "hidden_after_norm.npy"

    if cpp_norm_path.exists() and hf_norm_path.exists():
        cpp_norm = load_cpp_tensor(cpp_norm_path)
        hf_norm = load_hf_tensor(hf_norm_path)

        # Handle shape differences (HF may save only last token)
        if cpp_norm.ndim == 2 and hf_norm.ndim == 1:
            cpp_norm = cpp_norm[-1]
        elif cpp_norm.ndim == 1 and hf_norm.ndim == 2:
            hf_norm = hf_norm[-1]

        if cpp_norm.shape != hf_norm.shape:
            if cpp_norm.T.shape == hf_norm.shape:
                cpp_norm = cpp_norm.T

        if cpp_norm.shape == hf_norm.shape:
            corr = np.corrcoef(cpp_norm.flatten(), hf_norm.flatten())[0, 1]
            status = "✓" if corr > 0.99 else "✗ DIVERGE"
            print(f"{'Norm out':>8} | {cpp_norm.std():>12.6f} | {hf_norm.std():>12.6f} | {corr:>12.6f} | {status:>10}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if first_divergence is not None:
        print(f"\nFirst divergence at: LAYER {first_divergence}")
        print(f"\nThe bug is likely in:")
        if first_divergence == 0:
            print("  - Input processing (embedding lookup)")
            print("  - First layer attention or MoE FFN")
        else:
            print(f"  - Layer {first_divergence} attention")
            print(f"  - Layer {first_divergence} MoE FFN")
            print(f"  - Layer {first_divergence} residual connections")

        print("\nNext steps:")
        print(f"  1. Compare layer {first_divergence-1 if first_divergence > 0 else 0} output in detail")
        print(f"  2. Add debug dumps within layer {first_divergence}")
        print("  3. Check attention, MoE routing, and shared expert separately")
    else:
        print("\nAll layers match! ✓")
        print("If final output still diverges, check:")
        print("  - Output norm implementation")
        print("  - LM head (codec_head) weights")


if __name__ == "__main__":
    main()
