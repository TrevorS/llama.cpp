#!/usr/bin/env python3
"""Debug the dwconv -> norm transition to find the divergence source."""

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
    if arr.ndim >= 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def apply_layernorm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply LayerNorm: normalize across channels (dim 0), then scale/shift."""
    # x is [channels, seq], normalize each column (position) across channels
    mean = x.mean(axis=0, keepdims=True)
    var = x.var(axis=0, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    # weight and bias are [channels], reshape to [channels, 1] for broadcast
    return x_norm * weight.reshape(-1, 1) + bias.reshape(-1, 1)


def main():
    # Docker mounts ~/models at /models
    cpp_dir = Path("/models/debug/cpp_tokens_match")
    hf_dir = Path("/models/debug/hf_tensors_16f")

    n_frames = 16
    seq = n_frames * 2  # After first upsample = 32

    print("=== Comparing dwconv -> norm transition ===\n")

    # Load C++ tensors
    cpp_dwconv = load_ggml_tensor(cpp_dir / "cnxt0_dwconv.bin", (1024, seq))
    cpp_norm = load_ggml_tensor(cpp_dir / "cnxt0_norm.bin", (1024, seq))

    # Load HF tensors
    hf_dwconv = load_hf_tensor(hf_dir / "06_cnxt0_dwconv.npy")
    hf_norm = load_hf_tensor(hf_dir / "07_cnxt0_norm.npy")

    print(f"C++ dwconv shape: {cpp_dwconv.shape}")
    print(f"HF dwconv shape:  {hf_dwconv.shape}")
    print(f"C++ norm shape:   {cpp_norm.shape}")
    print(f"HF norm shape:    {hf_norm.shape}")

    # Check if shapes match (may need transpose)
    if cpp_dwconv.shape != hf_dwconv.shape:
        if cpp_dwconv.T.shape == hf_dwconv.shape:
            print("\nNote: C++ tensors need transpose to match HF")
            cpp_dwconv = cpp_dwconv.T
            cpp_norm = cpp_norm.T
        else:
            print(f"\nShape mismatch! Cannot compare directly.")
            return

    # Compute correlations
    corr_dwconv = np.corrcoef(cpp_dwconv.flatten(), hf_dwconv.flatten())[0, 1]
    corr_norm = np.corrcoef(cpp_norm.flatten(), hf_norm.flatten())[0, 1]

    print(f"\n=== Correlations ===")
    print(f"dwconv: {corr_dwconv:.6f}")
    print(f"norm:   {corr_norm:.6f}")
    print(f"Drop:   {corr_dwconv - corr_norm:.6f}")

    # Now let's manually apply LayerNorm to C++ dwconv and compare
    print(f"\n=== Manual LayerNorm Analysis ===")

    # Load norm weights from HF model (we'll need to extract these)
    # For now, let's compute what the normalized values should be

    # C++ normalized (before scale/shift)
    cpp_dwconv_normalized = (cpp_dwconv - cpp_dwconv.mean(axis=0, keepdims=True)) / \
                            np.sqrt(cpp_dwconv.var(axis=0, keepdims=True) + 1e-6)

    # HF normalized (before scale/shift)
    hf_dwconv_normalized = (hf_dwconv - hf_dwconv.mean(axis=0, keepdims=True)) / \
                           np.sqrt(hf_dwconv.var(axis=0, keepdims=True) + 1e-6)

    # Compare normalized values (before gamma/beta)
    corr_normalized = np.corrcoef(cpp_dwconv_normalized.flatten(), hf_dwconv_normalized.flatten())[0, 1]
    print(f"Normalized (before γ/β) correlation: {corr_normalized:.6f}")

    # Check if the issue is in the normalization axis
    # What if C++ normalizes over wrong axis?
    cpp_dwconv_wrong_axis = (cpp_dwconv - cpp_dwconv.mean(axis=1, keepdims=True)) / \
                            np.sqrt(cpp_dwconv.var(axis=1, keepdims=True) + 1e-6)
    corr_wrong = np.corrcoef(cpp_dwconv_wrong_axis.flatten(), hf_dwconv_normalized.flatten())[0, 1]
    print(f"If normalized over seq axis (wrong): {corr_wrong:.6f}")

    # Check statistics at specific positions
    print(f"\n=== Position-wise analysis ===")
    print(f"C++ dwconv pos 0 - mean: {cpp_dwconv[:, 0].mean():.4f}, std: {cpp_dwconv[:, 0].std():.4f}")
    print(f"HF dwconv pos 0  - mean: {hf_dwconv[:, 0].mean():.4f}, std: {hf_dwconv[:, 0].std():.4f}")

    print(f"\nC++ norm pos 0 - mean: {cpp_norm[:, 0].mean():.4f}, std: {cpp_norm[:, 0].std():.4f}")
    print(f"HF norm pos 0  - mean: {hf_norm[:, 0].mean():.4f}, std: {hf_norm[:, 0].std():.4f}")

    # Compare first few values
    print(f"\n=== Sample values at position 0 ===")
    print(f"C++ dwconv[:5, 0]: {cpp_dwconv[:5, 0]}")
    print(f"HF dwconv[:5, 0]:  {hf_dwconv[:5, 0]}")
    print(f"\nC++ norm[:5, 0]: {cpp_norm[:5, 0]}")
    print(f"HF norm[:5, 0]:  {hf_norm[:5, 0]}")

    # Check per-position correlations for dwconv and norm
    print(f"\n=== Per-position correlations ===")
    print("Position | dwconv corr | norm corr")
    print("-" * 40)
    for pos in range(min(8, cpp_dwconv.shape[1])):
        c_dw = np.corrcoef(cpp_dwconv[:, pos], hf_dwconv[:, pos])[0, 1]
        c_nm = np.corrcoef(cpp_norm[:, pos], hf_norm[:, pos])[0, 1]
        print(f"  {pos:3d}    |   {c_dw:.4f}   |  {c_nm:.4f}")

    # Check epsilon sensitivity
    print(f"\n=== Epsilon sensitivity ===")
    for eps in [1e-5, 1e-6, 1e-7, 1e-8]:
        cpp_norm_test = (cpp_dwconv - cpp_dwconv.mean(axis=0, keepdims=True)) / \
                        np.sqrt(cpp_dwconv.var(axis=0, keepdims=True) + eps)
        hf_norm_test = (hf_dwconv - hf_dwconv.mean(axis=0, keepdims=True)) / \
                       np.sqrt(hf_dwconv.var(axis=0, keepdims=True) + eps)
        corr = np.corrcoef(cpp_norm_test.flatten(), hf_norm_test.flatten())[0, 1]
        print(f"eps={eps:.0e}: normalized correlation = {corr:.6f}")


if __name__ == "__main__":
    main()
