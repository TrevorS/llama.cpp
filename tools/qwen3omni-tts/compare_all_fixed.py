#!/usr/bin/env python3
"""Compare all fixed tensors against HuggingFace."""

import numpy as np
import os
from pathlib import Path

def load_ggml_tensor(path, ne):
    """Load tensor from GGML dump with correct memory layout (Fortran order)."""
    data = np.fromfile(path, dtype='<f4')
    expected_size = int(np.prod(ne))
    if len(data) != expected_size:
        print(f"  Warning: Size mismatch: {len(data)} vs {expected_size}")
        return None
    return data.reshape(ne, order='F')

def load_hf_tensor(path):
    """Load tensor from HuggingFace format (with header)."""
    with open(path, 'rb') as f:
        import struct
        n_dims = struct.unpack('<I', f.read(4))[0]
        shape = tuple(struct.unpack('<I', f.read(4))[0] for _ in range(n_dims))
        data = np.frombuffer(f.read(), dtype='<f4').reshape(shape)
    return data

def compare_tensors(name, cpp_path, hf_path, cpp_ne, hf_shape=None):
    """Compare C++ and HuggingFace tensors."""
    if not cpp_path.exists():
        print(f"  {name}: C++ file not found")
        return None
    if not hf_path.exists():
        print(f"  {name}: HF file not found")
        return None

    cpp_data = load_ggml_tensor(cpp_path, cpp_ne)
    if cpp_data is None:
        return None

    hf_data = load_hf_tensor(hf_path)

    # Check if we need to transpose C++ data to match HF
    if cpp_data.shape != hf_data.shape:
        if cpp_data.T.shape == hf_data.shape:
            cpp_data = cpp_data.T
        else:
            print(f"  {name}: Shape mismatch - C++ {cpp_data.shape} vs HF {hf_data.shape}")
            return None

    corr = np.corrcoef(cpp_data.flatten(), hf_data.flatten())[0, 1]
    max_diff = np.abs(cpp_data - hf_data).max()

    status = "✓" if corr > 0.99 else "⚠" if corr > 0.9 else "✗"
    print(f"  {status} {name}: corr={corr:.6f}, max_diff={max_diff:.6f}")
    return corr

def main():
    cpp_dir = Path("/models/debug/cpp_fixed")
    hf_dir = Path("/models/debug/hf_tensors_v3")

    print("=== Comparing Fixed C++ vs HuggingFace ===\n")

    # Define tensor mappings: (name, cpp_ne, hf_name or None)
    tensors = [
        ("after_pretrans", (1024, 3), "03_cnxt_input.bin"),  # [channels, seq]
        ("cnxt0_transconv_raw", (1024, 6), None),  # After first upsample
        ("cnxt0_dwconv", (1024, 6), None),
        ("cnxt0_norm", (1024, 6), None),
        ("after_convnext", (1024, 12), "06_after_convnext.bin"),  # [channels, seq]
    ]

    print("Checking available HF tensors:")
    if hf_dir.exists():
        for f in sorted(hf_dir.glob("*.bin"))[:10]:
            print(f"  {f.name}")
    else:
        print(f"  HF dir not found: {hf_dir}")

    print("\n=== Tensor Comparisons ===")
    for name, cpp_ne, hf_name in tensors:
        cpp_path = cpp_dir / f"{name}.bin"
        if hf_name:
            hf_path = hf_dir / hf_name
            compare_tensors(name, cpp_path, hf_path, cpp_ne)
        else:
            # Just show stats for C++ tensor
            if cpp_path.exists():
                data = load_ggml_tensor(cpp_path, cpp_ne)
                if data is not None:
                    print(f"  ℹ {name}: shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}")

    # Check final output
    print("\n=== Final Output Stats ===")
    before_clamp = cpp_dir / "before_clamp.bin"
    if before_clamp.exists():
        data = np.fromfile(before_clamp, dtype='<f4')
        print(f"  before_clamp: {len(data)} samples, min={data.min():.4f}, max={data.max():.4f}")

if __name__ == "__main__":
    main()
