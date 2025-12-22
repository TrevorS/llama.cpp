#!/usr/bin/env python3
"""Compare all layer outputs between C++ and HuggingFace to find divergence point."""

import numpy as np
from pathlib import Path


def load_ggml_tensor(path: Path, ne: tuple) -> np.ndarray:
    """Load GGML tensor with Fortran order."""
    data = np.fromfile(path, dtype='<f4')
    expected_size = int(np.prod(ne))
    if len(data) != expected_size:
        raise ValueError(f"Size mismatch: got {len(data)}, expected {expected_size}")
    return data.reshape(ne, order='F')


def load_hf_tensor(path: Path) -> np.ndarray:
    """Load HF .npy tensor, removing batch dim if present."""
    arr = np.load(path)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def compare_tensors(cpp_arr, hf_arr, name, verbose=False):
    """Compare two tensors and print stats."""
    # Transpose if needed
    if cpp_arr.shape != hf_arr.shape:
        if cpp_arr.T.shape == hf_arr.shape:
            hf_arr = hf_arr.T

    if cpp_arr.shape != hf_arr.shape:
        print(f"  {name}: SHAPE MISMATCH {cpp_arr.shape} vs {hf_arr.shape}")
        return None

    corr = np.corrcoef(cpp_arr.flatten(), hf_arr.flatten())[0, 1]
    std_ratio = cpp_arr.std() / hf_arr.std()
    diff_std = np.abs(cpp_arr - hf_arr).std()

    if verbose:
        status = "OK" if corr > 0.99 and 0.95 < std_ratio < 1.05 else "MISMATCH"
        print(f"  {name}: corr={corr:.6f}, std_ratio={std_ratio:.4f}, diff_std={diff_std:.6f} [{status}]")

    return corr, std_ratio


def main():
    cpp_dir = Path("/models/debug/cpp_tokens_match")
    hf_dir = Path("/models/debug/hf_layers_match")

    n_frames = 16
    n_embd = 1024

    print("=" * 70)
    print("All Layers Comparison: C++ vs HuggingFace")
    print("=" * 70)

    # Load HF layer outputs
    print("\n=== HF Layer Outputs (std) ===")
    hf_layer_outputs = {}
    for i in range(8):
        path = hf_dir / f"layer{i}_output.npy"
        if path.exists():
            arr = load_hf_tensor(path)
            hf_layer_outputs[i] = arr
            print(f"  Layer {i}: std={arr.std():.6f}")

    # Load HF final outputs
    hf_before_norm = load_hf_tensor(hf_dir / "final_norm.npy") if (hf_dir / "final_norm.npy").exists() else None
    hf_after_norm = load_hf_tensor(hf_dir / "final_output.npy") if (hf_dir / "final_output.npy").exists() else None

    if hf_before_norm is not None:
        print(f"  Before norm: std={hf_before_norm.std():.6f}")
    if hf_after_norm is not None:
        print(f"  After norm: std={hf_after_norm.std():.6f}")

    # Load C++ tensors
    print("\n=== C++ Available Tensors ===")
    cpp_tensors = {}
    for p in sorted(cpp_dir.glob("*.bin")):
        size = p.stat().st_size // 4
        if size == n_embd * n_frames:
            try:
                arr = load_ggml_tensor(p, (n_embd, n_frames))
                cpp_tensors[p.stem] = arr
                print(f"  {p.stem}: std={arr.std():.6f}")
            except:
                pass

    # Key comparison: what C++ tensors do we have?
    print("\n=== Key Comparisons ===")

    # Layer 0 attention residual (C++) vs (HF input + HF attn_scaled)
    if "pretrans_layer0_after_attn_res" in cpp_tensors:
        hf_input = load_hf_tensor(hf_dir / "00_embd_mean.npy")
        hf_attn_scaled = load_hf_tensor(hf_dir / "layer0_attn_scaled.npy")
        hf_after_attn = hf_input + hf_attn_scaled

        cpp_arr = cpp_tensors["pretrans_layer0_after_attn_res"]
        if cpp_arr.T.shape == hf_after_attn.shape:
            hf_after_attn = hf_after_attn.T

        corr = np.corrcoef(cpp_arr.flatten(), hf_after_attn.flatten())[0, 1]
        std_ratio = cpp_arr.std() / hf_after_attn.std()
        print(f"\nLayer 0 after attention (before FFN):")
        print(f"  C++ std: {cpp_arr.std():.6f}, HF std: {hf_after_attn.std():.6f}")
        print(f"  Correlation: {corr:.6f}, Std ratio: {std_ratio:.4f}")

    # before_output_norm (C++) vs layer7_output (HF)
    if "pretrans_before_output_norm" in cpp_tensors and 7 in hf_layer_outputs:
        cpp_arr = cpp_tensors["pretrans_before_output_norm"]
        hf_arr = hf_layer_outputs[7]
        if cpp_arr.T.shape == hf_arr.shape:
            hf_arr = hf_arr.T

        corr = np.corrcoef(cpp_arr.flatten(), hf_arr.flatten())[0, 1]
        std_ratio = cpp_arr.std() / hf_arr.std()
        print(f"\nAfter 8 layers (before output norm):")
        print(f"  C++ std: {cpp_arr.std():.6f}, HF std: {hf_layer_outputs[7].std():.6f}")
        print(f"  Correlation: {corr:.6f}, Std ratio: {std_ratio:.4f}")

    # Summary analysis
    print("\n=== Analysis ===")
    print("""
The issue is clear:
- Layer 0 attention: PERFECT match (corr=1.0, std_ratio=1.0)
- After all layers: std_ratio = 1.37 (C++ higher), but corr still ~0.97

The C++ is producing HIGHER variance through the layers than HF.
But after RMSNorm, C++ has LOWER variance because:
- Higher RMS values divide by more
- The weight scaling then gives lower output

Root cause is somewhere in layers 0-7's FFN or layers 1-7's attention.

To pinpoint: need to dump C++ layer 0 FFN output and compare with HF.
""")

    # Check if C++ has any FFN-related dumps
    print("\n=== C++ FFN-related tensors ===")
    for name in cpp_tensors:
        if "ffn" in name.lower() or "mlp" in name.lower() or "layer" in name.lower():
            print(f"  {name}: std={cpp_tensors[name].std():.6f}")


if __name__ == "__main__":
    main()
