#!/usr/bin/env python3
"""Compare layer 0 intermediate tensors between C++ and HuggingFace."""

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


def compare_tensors(cpp_arr, hf_arr, name):
    """Compare two tensors and print stats."""
    # Transpose if needed
    if cpp_arr.shape != hf_arr.shape:
        if cpp_arr.T.shape == hf_arr.shape:
            hf_arr = hf_arr.T

    if cpp_arr.shape != hf_arr.shape:
        print(f"  {name}: SHAPE MISMATCH {cpp_arr.shape} vs {hf_arr.shape}")
        return

    corr = np.corrcoef(cpp_arr.flatten(), hf_arr.flatten())[0, 1]
    std_ratio = cpp_arr.std() / hf_arr.std()
    diff_std = np.abs(cpp_arr - hf_arr).std()

    status = "OK" if corr > 0.99 and 0.95 < std_ratio < 1.05 else "MISMATCH"
    print(f"  {name}: corr={corr:.6f}, std_ratio={std_ratio:.4f}, diff_std={diff_std:.6f} [{status}]")


def main():
    cpp_dir = Path("/models/debug/cpp_tokens_match")
    hf_dir = Path("/models/debug/hf_layers_match")

    n_frames = 16
    n_embd = 1024

    print("=" * 60)
    print("Layer 0 Comparison: C++ vs HuggingFace")
    print("=" * 60)

    # Load C++ tensors
    print("\n=== Loading C++ tensors ===")
    cpp_tensors = {}
    for name, shape in [
        ("pretrans_layer0_input", (n_embd, n_frames)),
        ("pretrans_layer0_after_attn_norm", (n_embd, n_frames)),
        ("pretrans_layer0_after_attn_res", (n_embd, n_frames)),
        ("pretrans_before_output_norm", (n_embd, n_frames)),
        ("after_pretrans", (n_embd, n_frames)),
    ]:
        path = cpp_dir / f"{name}.bin"
        if path.exists():
            arr = load_ggml_tensor(path, shape)
            cpp_tensors[name] = arr
            print(f"  {name}: shape={arr.shape}, std={arr.std():.6f}")
        else:
            print(f"  {name}: NOT FOUND")

    # Load HF tensors
    print("\n=== Loading HF tensors ===")
    hf_tensors = {}
    for name in ["00_embd_mean", "layer0_attn_norm", "layer0_attn_out", "layer0_attn_scaled",
                 "layer0_ffn_norm", "layer0_mlp_out", "layer0_mlp_scaled", "layer0_output",
                 "layer7_output", "final_norm", "final_output"]:
        path = hf_dir / f"{name}.npy"
        if path.exists():
            arr = load_hf_tensor(path)
            hf_tensors[name] = arr
            print(f"  {name}: shape={arr.shape}, std={arr.std():.6f}")
        else:
            print(f"  {name}: NOT FOUND")

    # Compare corresponding tensors
    print("\n=== Comparison ===")

    # Layer 0 input (should match embedding mean)
    if "pretrans_layer0_input" in cpp_tensors and "00_embd_mean" in hf_tensors:
        print("\nInput (embedding mean):")
        compare_tensors(cpp_tensors["pretrans_layer0_input"], hf_tensors["00_embd_mean"], "layer0_input")

    # After attention norm
    if "pretrans_layer0_after_attn_norm" in cpp_tensors and "layer0_attn_norm" in hf_tensors:
        print("\nAfter attention norm:")
        compare_tensors(cpp_tensors["pretrans_layer0_after_attn_norm"], hf_tensors["layer0_attn_norm"], "attn_norm")

    # C++ "after_attn_res" = input + attn_scaled
    # We need to compute this from HF
    if "00_embd_mean" in hf_tensors and "layer0_attn_scaled" in hf_tensors:
        hf_after_attn_res = hf_tensors["00_embd_mean"] + hf_tensors["layer0_attn_scaled"]
        if "pretrans_layer0_after_attn_res" in cpp_tensors:
            print("\nAfter attention residual:")
            compare_tensors(cpp_tensors["pretrans_layer0_after_attn_res"], hf_after_attn_res, "after_attn_res")

    # Final output
    if "after_pretrans" in cpp_tensors and "final_output" in hf_tensors:
        print("\nFinal output (after output norm):")
        compare_tensors(cpp_tensors["after_pretrans"], hf_tensors["final_output"], "final_output")

    # Check std progression
    print("\n=== Std Progression ===")
    print("\nC++:")
    for name in ["pretrans_layer0_input", "pretrans_layer0_after_attn_norm",
                 "pretrans_layer0_after_attn_res", "pretrans_before_output_norm", "after_pretrans"]:
        if name in cpp_tensors:
            print(f"  {name}: std={cpp_tensors[name].std():.6f}")

    print("\nHF:")
    for name in ["00_embd_mean", "layer0_attn_norm", "layer0_attn_out", "layer0_attn_scaled",
                 "layer0_output", "layer7_output", "final_output"]:
        if name in hf_tensors:
            print(f"  {name}: std={hf_tensors[name].std():.6f}")


if __name__ == "__main__":
    main()
