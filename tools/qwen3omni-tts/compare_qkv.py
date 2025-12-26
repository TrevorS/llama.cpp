#!/usr/bin/env python3
"""Compare Q/K/V tensors between C++ and HuggingFace to find divergence source."""

import struct
import numpy as np
from pathlib import Path


def load_cpp_tensor(path):
    """Load C++ tensor in our binary format."""
    with open(path, 'rb') as f:
        ndims = struct.unpack('<I', f.read(4))[0]
        shape = [struct.unpack('<I', f.read(4))[0] for _ in range(ndims)]
        data = np.frombuffer(f.read(), dtype='<f4')
        return data.reshape(shape)


def print_tensor_info(name, arr):
    """Print tensor statistics."""
    if np.isnan(arr).any() or np.isinf(arr).any():
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        print(f"  {name}: shape={arr.shape}, NaN={n_nan}, Inf={n_inf}")
    else:
        print(f"  {name}: shape={arr.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}")
        print(f"    first 5: [{', '.join(f'{x:.4f}' for x in arr.flatten()[:5])}]")


def compare_tensors(name, cpp_arr, hf_arr, transpose_cpp=True):
    """Compare C++ and HF tensors."""
    # C++ tensors are stored in ggml order, need to reverse dimensions
    if transpose_cpp and cpp_arr.ndim >= 2:
        # Reverse all dimensions for ggml->pytorch ordering
        cpp_arr = cpp_arr.T if cpp_arr.ndim == 2 else np.transpose(cpp_arr, list(reversed(range(cpp_arr.ndim))))

    print(f"\n=== {name} ===")
    print(f"C++ shape: {cpp_arr.shape}")
    print(f"HF  shape: {hf_arr.shape}")

    # Handle shape mismatches
    if cpp_arr.shape != hf_arr.shape:
        print("  Shape mismatch!")
        # Try to reshape if total elements match
        if cpp_arr.size == hf_arr.size:
            cpp_arr = cpp_arr.reshape(hf_arr.shape)
            print(f"  Reshaped C++ to {cpp_arr.shape}")
        else:
            print(f"  Cannot compare: size mismatch {cpp_arr.size} vs {hf_arr.size}")
            return

    # Check for NaN/Inf
    if np.isnan(cpp_arr).any() or np.isinf(cpp_arr).any():
        n_nan = np.isnan(cpp_arr).sum()
        n_inf = np.isinf(cpp_arr).sum()
        print(f"  C++ has NaN={n_nan}, Inf={n_inf}")
        return

    if np.isnan(hf_arr).any() or np.isinf(hf_arr).any():
        n_nan = np.isnan(hf_arr).sum()
        n_inf = np.isinf(hf_arr).sum()
        print(f"  HF has NaN={n_nan}, Inf={n_inf}")
        return

    # Overall correlation
    corr = np.corrcoef(cpp_arr.flatten(), hf_arr.flatten())[0, 1]
    print(f"Overall correlation: {corr:.6f}")

    # Per-token correlation (if applicable)
    if cpp_arr.ndim >= 2 and cpp_arr.shape[0] == 9:
        print("\nPer-token correlation:")
        for tok in range(min(9, cpp_arr.shape[0])):
            cpp_tok = cpp_arr[tok].flatten()
            hf_tok = hf_arr[tok].flatten()
            tok_corr = np.corrcoef(cpp_tok, hf_tok)[0, 1]
            print(f"  Token {tok}: {tok_corr:.6f}")

    # Statistics
    print(f"\nC++ stats: mean={cpp_arr.mean():.6f}, std={cpp_arr.std():.6f}")
    print(f"HF  stats: mean={hf_arr.mean():.6f}, std={hf_arr.std():.6f}")

    # Difference
    diff = np.abs(cpp_arr - hf_arr)
    print(f"Mean abs diff: {diff.mean():.6f}")
    print(f"Max abs diff: {diff.max():.6f} at {np.unravel_index(np.argmax(diff), diff.shape)}")


def main():
    cpp_dir = Path("/models/debug/cpp_talker")
    hf_dir = Path("/models/debug/hf_talker")

    print("=" * 80)
    print("Q/K/V Comparison: C++ vs HuggingFace")
    print("=" * 80)

    # Compare Q projection (before reshape/norm)
    cpp_q_proj = cpp_dir / "debug_Qcur_layer0.bin"
    hf_q_proj = hf_dir / "debug_layer0_q_proj.npy"

    if cpp_q_proj.exists() and hf_q_proj.exists():
        cpp_q = load_cpp_tensor(cpp_q_proj)
        hf_q = np.load(hf_q_proj)
        # HF shape is [1, 9, 2048], C++ is [9, 2048]
        hf_q = hf_q.squeeze(0)  # Remove batch dim
        compare_tensors("Q Projection (before reshape)", cpp_q, hf_q)
    else:
        print(f"\nQ projection files not found:")
        print(f"  C++: {cpp_q_proj.exists()}")
        print(f"  HF: {hf_q_proj.exists()}")

    # Compare K projection
    cpp_k_proj = cpp_dir / "debug_Kcur_layer0.bin"
    hf_k_proj = hf_dir / "debug_layer0_k_proj.npy"

    if cpp_k_proj.exists() and hf_k_proj.exists():
        cpp_k = load_cpp_tensor(cpp_k_proj)
        hf_k = np.load(hf_k_proj)
        hf_k = hf_k.squeeze(0)
        compare_tensors("K Projection (before reshape)", cpp_k, hf_k)

    # Compare V projection
    cpp_v_proj = cpp_dir / "debug_Vcur_layer0.bin"
    hf_v_proj = hf_dir / "debug_layer0_v_proj.npy"

    if cpp_v_proj.exists() and hf_v_proj.exists():
        cpp_v = load_cpp_tensor(cpp_v_proj)
        hf_v = np.load(hf_v_proj)
        hf_v = hf_v.squeeze(0)
        compare_tensors("V Projection (before reshape)", cpp_v, hf_v)

    # Compare Q after norm (before RoPE)
    cpp_q_norm = cpp_dir / "debug_Qcur_normed_layer0.bin"
    hf_q_norm = hf_dir / "debug_layer0_q_norm.npy"

    if cpp_q_norm.exists() and hf_q_norm.exists():
        cpp_qn = load_cpp_tensor(cpp_q_norm)
        hf_qn = np.load(hf_q_norm)
        hf_qn = hf_qn.squeeze(0)  # [9, 16, 128]
        compare_tensors("Q Normed (after RMSNorm, before RoPE)", cpp_qn, hf_qn)

    # Compare K after norm
    cpp_k_norm = cpp_dir / "debug_Kcur_normed_layer0.bin"
    hf_k_norm = hf_dir / "debug_layer0_k_norm.npy"

    if cpp_k_norm.exists() and hf_k_norm.exists():
        cpp_kn = load_cpp_tensor(cpp_k_norm)
        hf_kn = np.load(hf_k_norm)
        hf_kn = hf_kn.squeeze(0)  # [9, 2, 128]
        compare_tensors("K Normed (after RMSNorm, before RoPE)", cpp_kn, hf_kn)

    # Compare attn_norm (input layernorm output)
    cpp_attn_norm = cpp_dir / "debug_attn_norm_layer0.bin"
    hf_attn_norm = hf_dir / "hooked_layer0_attn_norm_out.npy"

    if cpp_attn_norm.exists() and hf_attn_norm.exists():
        cpp_an = load_cpp_tensor(cpp_attn_norm)
        hf_an = np.load(hf_attn_norm)
        hf_an = hf_an.squeeze(0)  # [9, 1024]
        compare_tensors("Attention Norm Output (input layernorm)", cpp_an, hf_an)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
