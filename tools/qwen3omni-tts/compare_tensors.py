#!/usr/bin/env python3
"""
Compare intermediate tensors between HuggingFace and GGML Code2Wav implementations.

Usage:
    python compare_tensors.py /tmp/hf_tensors/ /tmp/cpp_tensors/
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np


def load_npy(path: str) -> np.ndarray:
    """Load a numpy array from file."""
    return np.load(path)


def load_raw(path: str, dtype=np.float32) -> np.ndarray:
    """Load raw binary float32 array."""
    return np.fromfile(path, dtype=dtype)


def load_tensor_bin(path: str) -> np.ndarray:
    """
    Load tensor from binary file with dimension header.
    Format: [num_dims: uint32] [dim0: uint32] [dim1: uint32] ... [data: float32]
    """
    with open(path, 'rb') as f:
        # Read number of dimensions
        num_dims = struct.unpack('<I', f.read(4))[0]

        # Read each dimension
        shape = []
        for _ in range(num_dims):
            dim = struct.unpack('<I', f.read(4))[0]
            shape.append(dim)

        # Read data
        num_elements = int(np.prod(shape))
        data = np.frombuffer(f.read(num_elements * 4), dtype='<f4')

        # Reshape
        tensor = data.reshape(shape)

    return tensor


def compute_stats(arr: np.ndarray) -> dict:
    """Compute summary statistics for an array."""
    return {
        "shape": arr.shape,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "absmax": float(np.max(np.abs(arr))),
    }


def compare_tensors(hf_arr: np.ndarray, cpp_arr: np.ndarray, name: str, tol: float = 0.01) -> dict:
    """Compare two tensors and return comparison metrics."""
    result = {
        "name": name,
        "hf_shape": hf_arr.shape,
        "cpp_shape": cpp_arr.shape,
        "shape_match": hf_arr.shape == cpp_arr.shape,
    }

    # Flatten for comparison if shapes differ
    hf_flat = hf_arr.flatten()
    cpp_flat = cpp_arr.flatten()

    if len(hf_flat) != len(cpp_flat):
        result["error"] = f"Size mismatch: HF={len(hf_flat)}, CPP={len(cpp_flat)}"
        result["diverged"] = True
        return result

    # Compute difference metrics
    diff = np.abs(hf_flat - cpp_flat)
    result["mse"] = float(np.mean(diff ** 2))
    result["max_diff"] = float(np.max(diff))
    result["mean_diff"] = float(np.mean(diff))

    # Relative error (avoid divide by zero)
    hf_absmax = np.max(np.abs(hf_flat))
    if hf_absmax > 1e-10:
        result["rel_max_diff"] = result["max_diff"] / hf_absmax
        result["rel_mean_diff"] = result["mean_diff"] / hf_absmax
    else:
        result["rel_max_diff"] = result["max_diff"]
        result["rel_mean_diff"] = result["mean_diff"]

    # Correlation
    if np.std(hf_flat) > 1e-10 and np.std(cpp_flat) > 1e-10:
        result["correlation"] = float(np.corrcoef(hf_flat, cpp_flat)[0, 1])
    else:
        result["correlation"] = 0.0

    # Stats
    result["hf_stats"] = compute_stats(hf_arr)
    result["cpp_stats"] = compute_stats(cpp_arr)

    # Check if diverged
    result["diverged"] = result["rel_max_diff"] > tol or result["correlation"] < 0.9

    return result


def main():
    parser = argparse.ArgumentParser(description="Compare HuggingFace and GGML intermediate tensors")
    parser.add_argument("hf_dir", help="Directory containing HuggingFace .npy tensors")
    parser.add_argument("cpp_dir", help="Directory containing GGML .bin or .npy tensors")
    parser.add_argument("--tol", type=float, default=0.01, help="Relative tolerance for divergence (default: 0.01)")
    args = parser.parse_args()

    hf_dir = Path(args.hf_dir)
    cpp_dir = Path(args.cpp_dir)

    if not hf_dir.exists():
        print(f"Error: HF directory not found: {hf_dir}")
        sys.exit(1)
    if not cpp_dir.exists():
        print(f"Error: CPP directory not found: {cpp_dir}")
        sys.exit(1)

    # Find matching tensor files (support both .npy and .bin)
    hf_npy_files = sorted(hf_dir.glob("*.npy"))
    hf_bin_files = sorted(hf_dir.glob("*.bin"))

    if hf_npy_files:
        hf_files = hf_npy_files
        hf_format = "npy"
    elif hf_bin_files:
        hf_files = hf_bin_files
        hf_format = "bin"
    else:
        print(f"Error: No .npy or .bin files found in {hf_dir}")
        sys.exit(1)

    print(f"Found {len(hf_files)} HuggingFace tensor files (.{hf_format})")

    results = []
    first_divergence = None

    for hf_file in hf_files:
        name = hf_file.stem

        # Load HF tensor
        if hf_format == "npy":
            hf_arr = load_npy(str(hf_file))
        else:
            hf_arr = load_tensor_bin(str(hf_file))

        # Try to find matching CPP file (.bin with header, .npy, or raw .bin)
        cpp_bin = cpp_dir / f"{name}.bin"
        cpp_npy = cpp_dir / f"{name}.npy"

        if cpp_bin.exists():
            cpp_file = cpp_bin
            # Try loading as tensor_bin first (with header), fall back to raw
            try:
                cpp_arr = load_tensor_bin(str(cpp_file))
            except Exception:
                cpp_arr = load_raw(str(cpp_file))
        elif cpp_npy.exists():
            cpp_file = cpp_npy
            cpp_arr = load_npy(str(cpp_file))
        else:
            print(f"  SKIP {name}: No matching CPP file")
            continue

        result = compare_tensors(hf_arr, cpp_arr, name, args.tol)
        results.append(result)

        # Print result
        status = "DIVERGED" if result["diverged"] else "OK"
        print(f"  {status:8} {name}")
        print(f"           HF shape: {result['hf_shape']}, CPP shape: {result['cpp_shape']}")
        print(f"           MSE: {result['mse']:.2e}, MaxDiff: {result['max_diff']:.2e}, Corr: {result.get('correlation', 'N/A'):.4f}")
        print(f"           HF: mean={result['hf_stats']['mean']:.4f}, std={result['hf_stats']['std']:.4f}")
        print(f"           CPP: mean={result['cpp_stats']['mean']:.4f}, std={result['cpp_stats']['std']:.4f}")

        if result["diverged"] and first_divergence is None:
            first_divergence = name

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    n_ok = sum(1 for r in results if not r["diverged"])
    n_diverged = sum(1 for r in results if r["diverged"])

    print(f"Total tensors compared: {len(results)}")
    print(f"  OK: {n_ok}")
    print(f"  Diverged: {n_diverged}")

    if first_divergence:
        print(f"\nFIRST DIVERGENCE: {first_divergence}")
        print("This is likely where the bug is!")
    else:
        print("\nAll tensors match within tolerance!")


if __name__ == "__main__":
    main()
