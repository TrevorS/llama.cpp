#!/usr/bin/env python3
"""
Debug dwconv operation by applying PyTorch dwconv to C++ transconv output.
This isolates whether the issue is in the operation or the input.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import glob
import safetensors.torch


def load_ggml_tensor(path, shape):
    """Load GGML tensor with correct memory layout."""
    arr = np.fromfile(str(path), dtype=np.float32)
    ne0, ne1 = shape
    expected_size = ne0 * ne1
    if len(arr) != expected_size:
        return None, f"Size mismatch: expected {expected_size}, got {len(arr)}"
    # GGML column-major: reshape to (ne1, ne0) then transpose
    return arr.reshape(ne1, ne0).T, None


def main():
    hf_dir = Path('/models/debug/hf_tensors')
    cpp_dir = Path('/models/debug/cpp_tensors_new')
    model_path = '/models/Qwen3-Omni-30B-A3B-Instruct'

    # Load tensors
    print("Loading tensors...")

    # HF transconv output (dwconv input)
    hf_transconv = np.load(str(hf_dir / '05_up0_transconv.npy'))  # [1, 1024, 40]
    print(f"HF transconv shape: {hf_transconv.shape}")

    # HF dwconv output
    hf_dwconv = np.load(str(hf_dir / '06_cnxt0_dwconv.npy'))  # [1, 1024, 40]
    print(f"HF dwconv shape: {hf_dwconv.shape}")

    # C++ transconv output (dwconv input)
    cpp_transconv, err = load_ggml_tensor(cpp_dir / 'cnxt0_transconv_raw.bin', (1024, 40))
    if err:
        print(f"Error loading C++ transconv: {err}")
        return
    print(f"C++ transconv shape: {cpp_transconv.shape}")

    # C++ dwconv output
    cpp_dwconv, err = load_ggml_tensor(cpp_dir / 'cnxt0_dwconv.bin', (1024, 40))
    if err:
        print(f"Error loading C++ dwconv: {err}")
        return
    print(f"C++ dwconv shape: {cpp_dwconv.shape}")

    # Check transconv correlation
    corr_transconv = np.corrcoef(hf_transconv.flatten(), cpp_transconv.flatten())[0, 1]
    print(f"\nTransconv correlation (HF vs C++): {corr_transconv:.4f}")

    # Check dwconv output correlation
    corr_dwconv = np.corrcoef(hf_dwconv.flatten(), cpp_dwconv.flatten())[0, 1]
    print(f"Dwconv correlation (HF vs C++): {corr_dwconv:.4f}")

    # Load HF weights
    print("\nLoading HF dwconv weights...")
    hf_weight = None
    hf_bias = None
    for sf in sorted(glob.glob(f'{model_path}/*.safetensors')):
        weights = safetensors.torch.load_file(sf)
        if 'code2wav.upsample.0.1.dwconv.conv.weight' in weights:
            hf_weight = weights['code2wav.upsample.0.1.dwconv.conv.weight'].float()
        if 'code2wav.upsample.0.1.dwconv.conv.bias' in weights:
            hf_bias = weights['code2wav.upsample.0.1.dwconv.conv.bias'].float()

    if hf_weight is None:
        print("Could not find dwconv weight")
        return

    print(f"HF dwconv weight shape: {hf_weight.shape}")  # [1024, 1, 7]

    # Create PyTorch dwconv
    dwconv = nn.Conv1d(
        in_channels=1024,
        out_channels=1024,
        kernel_size=7,
        padding=3,  # same padding
        groups=1024,  # depthwise
        bias=True
    )
    dwconv.weight.data = hf_weight
    if hf_bias is not None:
        dwconv.bias.data = hf_bias
    dwconv.eval()

    # Apply PyTorch dwconv to HF transconv output
    with torch.no_grad():
        pytorch_from_hf = dwconv(torch.from_numpy(hf_transconv)).numpy()

    # Compare PyTorch dwconv on HF input vs HF dwconv output
    corr_pt_hf = np.corrcoef(pytorch_from_hf.flatten(), hf_dwconv.flatten())[0, 1]
    print(f"\nPyTorch(HF input) vs HF dwconv: {corr_pt_hf:.4f}")

    # Apply PyTorch dwconv to C++ transconv output
    cpp_transconv_torch = cpp_transconv.reshape(1, 1024, 40).astype(np.float32)
    with torch.no_grad():
        pytorch_from_cpp = dwconv(torch.from_numpy(cpp_transconv_torch)).numpy()

    # Compare PyTorch dwconv on C++ input vs C++ dwconv output
    corr_pt_cpp = np.corrcoef(pytorch_from_cpp.flatten(), cpp_dwconv.flatten())[0, 1]
    print(f"PyTorch(C++ input) vs C++ dwconv: {corr_pt_cpp:.4f}")

    # Compare PyTorch dwconv on C++ input vs HF dwconv output
    corr_pt_cpp_hf = np.corrcoef(pytorch_from_cpp.flatten(), hf_dwconv.flatten())[0, 1]
    print(f"PyTorch(C++ input) vs HF dwconv: {corr_pt_cpp_hf:.4f}")

    print("\n=== Detailed Analysis ===")

    # Check if HF transconv output and C++ transconv output are in different layouts
    print("\nHF transconv first values [0, :5, 0]:", hf_transconv[0, :5, 0])
    print("C++ transconv first values [:5, 0]:", cpp_transconv[:5, 0])

    print("\nHF dwconv first values [0, :5, 0]:", hf_dwconv[0, :5, 0])
    print("C++ dwconv first values [:5, 0]:", cpp_dwconv[:5, 0])

    # Check actual value ranges
    print("\n=== Value Ranges ===")
    print(f"HF transconv: min={hf_transconv.min():.4f}, max={hf_transconv.max():.4f}, mean={hf_transconv.mean():.4f}")
    print(f"C++ transconv: min={cpp_transconv.min():.4f}, max={cpp_transconv.max():.4f}, mean={cpp_transconv.mean():.4f}")
    print(f"HF dwconv: min={hf_dwconv.min():.4f}, max={hf_dwconv.max():.4f}, mean={hf_dwconv.mean():.4f}")
    print(f"C++ dwconv: min={cpp_dwconv.min():.4f}, max={cpp_dwconv.max():.4f}, mean={cpp_dwconv.mean():.4f}")


def check_hf_dwconv():
    """Check what HF dwconv actually is."""
    import torch
    from transformers import AutoModel

    print("\n=== Checking HF convnext.dwconv ===")

    # Load model
    model = AutoModel.from_pretrained(
        '/models/Qwen3-Omni-30B-A3B-Instruct',
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map='cpu'
    )

    c2w = model.code2wav
    convnext0 = c2w.upsample[0][1]  # First ConvNeXt block

    print(f"convnext type: {type(convnext0)}")
    print(f"convnext.dwconv type: {type(convnext0.dwconv)}")
    print(f"convnext.dwconv: {convnext0.dwconv}")

    # Check if it's a module or something else
    if hasattr(convnext0.dwconv, 'conv'):
        print(f"convnext.dwconv.conv: {convnext0.dwconv.conv}")


if __name__ == '__main__':
    main()
    check_hf_dwconv()
