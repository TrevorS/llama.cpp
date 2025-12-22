#!/usr/bin/env python3
"""Debug the dwconv operation to find divergence source."""

import numpy as np
import torch
import torch.nn.functional as F
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
    seq_in = n_frames * 2  # 32 (after transconv)

    print("=== Loading transconv output (dwconv input) ===")
    cpp_transconv = load_ggml_tensor(cpp_dir / "cnxt0_transconv_raw.bin", (1024, seq_in))
    hf_transconv = load_hf_tensor(hf_dir / "05_up0_transconv.npy")
    print(f"C++ transconv shape: {cpp_transconv.shape}")
    print(f"HF transconv shape:  {hf_transconv.shape}")

    corr_transconv = np.corrcoef(cpp_transconv.flatten(), hf_transconv.flatten())[0, 1]
    print(f"Transconv correlation: {corr_transconv:.6f}")

    print("\n=== Loading dwconv output ===")
    cpp_dwconv = load_ggml_tensor(cpp_dir / "cnxt0_dwconv.bin", (1024, seq_in))
    hf_dwconv = load_hf_tensor(hf_dir / "06_cnxt0_dwconv.npy")

    corr_dwconv = np.corrcoef(cpp_dwconv.flatten(), hf_dwconv.flatten())[0, 1]
    print(f"dwconv correlation: {corr_dwconv:.6f}")
    print(f"Correlation drop transconv->dwconv: {corr_transconv - corr_dwconv:.6f}")

    # Load HF dwconv weights
    print("\n=== Loading HF dwconv weights ===")
    hf_dwconv_weight = None
    hf_dwconv_bias = None
    for f in os.listdir(model_path):
        if f.endswith('.safetensors'):
            path = os.path.join(model_path, f)
            with safe_open(path, framework='pt') as sf:
                for name in sf.keys():
                    if 'upsample.0.1.dwconv.conv.weight' in name:
                        hf_dwconv_weight = sf.get_tensor(name).float()
                        print(f"Found weight: {name}, shape: {list(hf_dwconv_weight.shape)}")
                    elif 'upsample.0.1.dwconv.conv.bias' in name:
                        hf_dwconv_bias = sf.get_tensor(name).float()
                        print(f"Found bias: {name}, shape: {list(hf_dwconv_bias.shape)}")
            if hf_dwconv_weight is not None and hf_dwconv_bias is not None:
                break

    if hf_dwconv_weight is None:
        print("Could not find HF dwconv weights!")
        return

    print(f"\nHF dwconv weight shape: {list(hf_dwconv_weight.shape)}")

    # Apply HF dwconv to C++ transconv input and compare
    print("\n=== Applying HF dwconv to C++ transconv input ===")

    kernel_size = hf_dwconv_weight.shape[2]
    left_pad = kernel_size - 1

    cpp_input = torch.from_numpy(cpp_transconv.copy()).unsqueeze(0)
    print(f"Input shape for conv: {list(cpp_input.shape)}")

    cpp_padded = F.pad(cpp_input, (left_pad, 0))
    print(f"Padded shape: {list(cpp_padded.shape)}")

    cpp_dwconv_manual = F.conv1d(cpp_padded, hf_dwconv_weight, hf_dwconv_bias, groups=1024)
    print(f"Manual dwconv output shape: {list(cpp_dwconv_manual.shape)}")

    cpp_dwconv_manual_np = cpp_dwconv_manual[0].detach().numpy()

    corr_manual = np.corrcoef(cpp_dwconv_manual_np.flatten(), hf_dwconv.flatten())[0, 1]
    print(f"Manual dwconv on C++ input vs HF output: {corr_manual:.6f}")

    # Also test with HF transconv input
    print("\n=== Applying HF dwconv to HF transconv input ===")
    hf_input = torch.from_numpy(hf_transconv.copy()).unsqueeze(0)
    hf_padded = F.pad(hf_input, (left_pad, 0))
    hf_dwconv_manual = F.conv1d(hf_padded, hf_dwconv_weight, hf_dwconv_bias, groups=1024)
    hf_dwconv_manual_np = hf_dwconv_manual[0].detach().numpy()

    corr_hf_manual = np.corrcoef(hf_dwconv_manual_np.flatten(), hf_dwconv.flatten())[0, 1]
    print(f"Manual dwconv on HF input vs HF output: {corr_hf_manual:.6f}")

    # Compare C++ dwconv output with manual
    print("\n=== Comparing C++ dwconv output with manual ===")
    corr_cpp_vs_manual = np.corrcoef(cpp_dwconv.flatten(), cpp_dwconv_manual_np.flatten())[0, 1]
    print(f"C++ dwconv vs manual on C++ input: {corr_cpp_vs_manual:.6f}")

    # Check specific values
    print("\n=== Sample values at position 0 ===")
    print(f"C++ transconv[:5, 0]:     {cpp_transconv[:5, 0]}")
    print(f"HF transconv[:5, 0]:      {hf_transconv[:5, 0]}")
    print(f"\nC++ dwconv[:5, 0]:        {cpp_dwconv[:5, 0]}")
    print(f"HF dwconv[:5, 0]:         {hf_dwconv[:5, 0]}")
    print(f"Manual dwconv[:5, 0]:     {cpp_dwconv_manual_np[:5, 0]}")

    # Per-channel analysis
    print("\n=== Per-channel correlation (first 10 channels) ===")
    print("Channel | transconv | dwconv | cpp_vs_manual")
    print("-" * 50)
    for ch in range(10):
        c_trans = np.corrcoef(cpp_transconv[ch, :], hf_transconv[ch, :])[0, 1]
        c_dw = np.corrcoef(cpp_dwconv[ch, :], hf_dwconv[ch, :])[0, 1]
        c_man = np.corrcoef(cpp_dwconv[ch, :], cpp_dwconv_manual_np[ch, :])[0, 1]
        print(f"  {ch:3d}   |  {c_trans:.4f}  |  {c_dw:.4f}  |  {c_man:.4f}")


if __name__ == "__main__":
    main()
