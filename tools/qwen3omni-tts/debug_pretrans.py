#!/usr/bin/env python3
"""Debug the pre-transformer to find the source of 0.976 correlation."""

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


def main():
    cpp_dir = Path("/models/debug/cpp_tokens_match")
    hf_dir = Path("/models/debug/hf_tensors_16f")

    n_frames = 16

    # Load all available pre-transformer tensors
    print("=== Pre-transformer analysis ===\n")

    # Check what tensors we have
    cpp_tensors = sorted(cpp_dir.glob("*.bin"))
    hf_tensors = sorted(hf_dir.glob("*.npy"))

    print("Available C++ tensors:")
    for t in cpp_tensors[:15]:
        size = t.stat().st_size // 4
        print(f"  {t.name}: {size} floats")

    print("\nAvailable HF tensors:")
    for t in hf_tensors[:15]:
        arr = np.load(t)
        print(f"  {t.name}: shape {arr.shape}")

    # Load and compare embeddings
    print("\n=== Comparing embedding input ===")
    hf_embd = load_hf_tensor(hf_dir / "02_embd_mean.npy")
    print(f"HF embd_mean shape: {hf_embd.shape}")

    # Check if we have C++ embedding dump
    if (cpp_dir / "embd_input.bin").exists():
        cpp_embd = load_ggml_tensor(cpp_dir / "embd_input.bin", (1024, n_frames))
        print(f"C++ embd shape: {cpp_embd.shape}")
        corr = np.corrcoef(cpp_embd.flatten(), hf_embd.flatten())[0, 1]
        print(f"Embedding correlation: {corr:.6f}")

    # Load and compare pre-transformer output
    print("\n=== Pre-transformer output ===")
    cpp_pretrans = load_ggml_tensor(cpp_dir / "after_pretrans.bin", (1024, n_frames))
    hf_pretrans = load_hf_tensor(hf_dir / "03_pre_xfmr_out.npy")

    print(f"C++ pretrans shape: {cpp_pretrans.shape}")
    print(f"HF pretrans shape:  {hf_pretrans.shape}")

    # Check if transpose needed
    if cpp_pretrans.shape != hf_pretrans.shape:
        if cpp_pretrans.T.shape == hf_pretrans.shape:
            hf_pretrans = hf_pretrans.T
            print("Transposed HF to match C++")

    corr = np.corrcoef(cpp_pretrans.flatten(), hf_pretrans.flatten())[0, 1]
    print(f"Pre-transformer correlation: {corr:.6f}")

    # Per-position analysis
    print("\n=== Per-position correlation ===")
    for pos in range(min(8, cpp_pretrans.shape[1])):
        c = np.corrcoef(cpp_pretrans[:, pos], hf_pretrans[:, pos])[0, 1]
        print(f"Position {pos}: {c:.6f}")

    # Check statistics
    print("\n=== Statistics ===")
    print(f"C++ pretrans - mean: {cpp_pretrans.mean():.6f}, std: {cpp_pretrans.std():.6f}")
    print(f"HF pretrans  - mean: {hf_pretrans.mean():.6f}, std: {hf_pretrans.std():.6f}")

    # Diff analysis
    diff = cpp_pretrans - hf_pretrans
    print(f"\nDiff - mean: {diff.mean():.6f}, std: {diff.std():.6f}")
    print(f"Diff - max abs: {np.abs(diff).max():.6f}")

    # Check first/last positions (sliding window boundaries)
    print("\n=== Boundary position analysis ===")
    for pos in [0, 1, n_frames-2, n_frames-1]:
        c = np.corrcoef(cpp_pretrans[:, pos], hf_pretrans[:, pos])[0, 1]
        diff_pos = cpp_pretrans[:, pos] - hf_pretrans[:, pos]
        print(f"Pos {pos:2d}: corr={c:.4f}, diff_mean={diff_pos.mean():.4f}, diff_std={diff_pos.std():.4f}")

    # Check if there's a systematic pattern
    print("\n=== Sample values at position 0 ===")
    print(f"C++ [:10]: {cpp_pretrans[:10, 0]}")
    print(f"HF  [:10]: {hf_pretrans[:10, 0]}")
    print(f"Diff[:10]: {diff[:10, 0]}")

    # Check intermediate layer dumps if available
    print("\n=== Checking for intermediate layer dumps ===")
    for layer_file in sorted(cpp_dir.glob("pretrans_layer*")):
        print(f"Found: {layer_file.name}")


if __name__ == "__main__":
    main()
