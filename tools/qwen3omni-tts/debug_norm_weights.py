#!/usr/bin/env python3
"""Debug LayerNorm weight/bias application."""

import numpy as np
from pathlib import Path
from safetensors import safe_open
import os


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


def main():
    cpp_dir = Path("/models/debug/cpp_tokens_match")
    hf_dir = Path("/models/debug/hf_tensors_16f")
    model_path = "/models/Qwen3-Omni-30B-A3B-Instruct"

    n_frames = 16
    seq = n_frames * 2  # 32

    print("=== Loading dwconv output ===")
    cpp_dwconv = load_ggml_tensor(cpp_dir / "cnxt0_dwconv.bin", (1024, seq))
    hf_dwconv = load_hf_tensor(hf_dir / "06_cnxt0_dwconv.npy")
    print(f"C++ dwconv shape: {cpp_dwconv.shape}")
    print(f"HF dwconv shape:  {hf_dwconv.shape}")

    print("\n=== Loading norm output ===")
    cpp_norm = load_ggml_tensor(cpp_dir / "cnxt0_norm.bin", (1024, seq))
    hf_norm = load_hf_tensor(hf_dir / "07_cnxt0_norm.npy")
    print(f"C++ norm shape: {cpp_norm.shape}")
    print(f"HF norm shape:  {hf_norm.shape}")

    # Load HF norm weights from safetensors
    print("\n=== Loading HF LayerNorm weights ===")
    hf_weight = None
    hf_bias = None
    for f in os.listdir(model_path):
        if f.endswith('.safetensors'):
            path = os.path.join(model_path, f)
            with safe_open(path, framework='pt') as sf:
                for name in sf.keys():
                    # ConvNeXt block 0 norm
                    if 'upsample.0.1.norm.weight' in name:
                        hf_weight = sf.get_tensor(name).float().numpy()
                        print(f"Found weight: {name}, shape: {hf_weight.shape}")
                    elif 'upsample.0.1.norm.bias' in name:
                        hf_bias = sf.get_tensor(name).float().numpy()
                        print(f"Found bias: {name}, shape: {hf_bias.shape}")
            if hf_weight is not None and hf_bias is not None:
                break

    if hf_weight is None or hf_bias is None:
        print("Could not find HF norm weights!")
        return

    print(f"\nHF weight[:5]: {hf_weight[:5]}")
    print(f"HF bias[:5]:   {hf_bias[:5]}")

    # Manually apply LayerNorm with HF weights to C++ dwconv
    print("\n=== Applying LayerNorm to C++ dwconv with HF weights ===")
    eps = 1e-6
    # Normalize each column (position) across channels
    mean = cpp_dwconv.mean(axis=0, keepdims=True)
    var = cpp_dwconv.var(axis=0, keepdims=True)
    cpp_normalized = (cpp_dwconv - mean) / np.sqrt(var + eps)

    # Apply weight and bias - weight/bias are [1024], need to broadcast
    cpp_norm_manual = cpp_normalized * hf_weight.reshape(-1, 1) + hf_bias.reshape(-1, 1)

    corr_manual = np.corrcoef(cpp_norm_manual.flatten(), hf_norm.flatten())[0, 1]
    print(f"Manual LayerNorm on C++ dwconv vs HF norm: {corr_manual:.6f}")

    # Also try with HF dwconv input
    print("\n=== Applying LayerNorm to HF dwconv with HF weights ===")
    mean_hf = hf_dwconv.mean(axis=0, keepdims=True)
    var_hf = hf_dwconv.var(axis=0, keepdims=True)
    hf_normalized = (hf_dwconv - mean_hf) / np.sqrt(var_hf + eps)
    hf_norm_manual = hf_normalized * hf_weight.reshape(-1, 1) + hf_bias.reshape(-1, 1)

    corr_hf_manual = np.corrcoef(hf_norm_manual.flatten(), hf_norm.flatten())[0, 1]
    print(f"Manual LayerNorm on HF dwconv vs HF norm: {corr_hf_manual:.6f}")

    # Check if the C++ LayerNorm weights are correct
    # We need to compare the weights actually used in C++
    print("\n=== Comparing specific values ===")
    print(f"C++ norm[:5, 0]:       {cpp_norm[:5, 0]}")
    print(f"HF norm[:5, 0]:        {hf_norm[:5, 0]}")
    print(f"Manual norm[:5, 0]:    {cpp_norm_manual[:5, 0]}")
    print(f"HF manual norm[:5, 0]: {hf_norm_manual[:5, 0]}")

    # Check the difference pattern
    print("\n=== Difference analysis ===")
    diff_cpp = cpp_norm - hf_norm
    diff_manual = cpp_norm_manual - hf_norm
    print(f"C++ norm error - mean: {diff_cpp.mean():.6f}, std: {diff_cpp.std():.6f}")
    print(f"Manual norm error - mean: {diff_manual.mean():.6f}, std: {diff_manual.std():.6f}")

    # Check if the issue is in dwconv itself
    print("\n=== Checking if dwconv difference explains norm difference ===")
    dwconv_diff = cpp_dwconv - hf_dwconv
    print(f"dwconv diff - mean: {dwconv_diff.mean():.6f}, std: {dwconv_diff.std():.6f}")

    # What's the expected norm output from C++ normalized?
    print("\n=== Checking normalization consistency ===")
    cpp_norm_expected_mean = cpp_normalized.mean(axis=0)
    cpp_norm_expected_std = cpp_normalized.std(axis=0)
    print(f"C++ normalized mean (should be ~0): min={cpp_norm_expected_mean.min():.6f}, max={cpp_norm_expected_mean.max():.6f}")
    print(f"C++ normalized std (should be ~1):  min={cpp_norm_expected_std.min():.6f}, max={cpp_norm_expected_std.max():.6f}")

    # Check actual C++ norm output statistics
    cpp_norm_output_mean = cpp_norm.mean(axis=0)
    hf_norm_output_mean = hf_norm.mean(axis=0)
    print(f"\nC++ norm output mean per position: min={cpp_norm_output_mean.min():.6f}, max={cpp_norm_output_mean.max():.6f}")
    print(f"HF norm output mean per position:  min={hf_norm_output_mean.min():.6f}, max={hf_norm_output_mean.max():.6f}")


if __name__ == "__main__":
    main()
