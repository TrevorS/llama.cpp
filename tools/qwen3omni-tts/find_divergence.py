#!/usr/bin/env python3
"""Find exactly where the C++ and HF pre-transformer diverge."""

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


def compare(cpp_arr, hf_arr, name):
    """Compare two tensors."""
    # Transpose if needed
    if cpp_arr.shape != hf_arr.shape:
        if cpp_arr.T.shape == hf_arr.shape:
            hf_arr = hf_arr.T

    if cpp_arr.shape != hf_arr.shape:
        return f"SHAPE MISMATCH {cpp_arr.shape} vs {hf_arr.shape}", 0, 0

    corr = np.corrcoef(cpp_arr.flatten(), hf_arr.flatten())[0, 1]
    std_ratio = cpp_arr.std() / hf_arr.std()
    return None, corr, std_ratio


def main():
    cpp_dir = Path("/models/debug/cpp_layer0_debug")
    hf_dir = Path("/models/debug/hf_layers_match")

    n_frames = 16
    n_embd = 1024

    print("=" * 70)
    print("Finding Divergence Point in Pre-Transformer")
    print("=" * 70)

    comparisons = [
        # (cpp_name, cpp_shape, hf_name, description)
        ("pretrans_layer0_input", (n_embd, n_frames), "00_embd_mean", "Layer 0 input (embedding mean)"),
        ("pretrans_layer0_after_attn_norm", (n_embd, n_frames), "layer0_attn_norm", "Layer 0 after attn norm"),
        ("pretrans_layer0_ffn_norm", (n_embd, n_frames), "layer0_ffn_norm", "Layer 0 after FFN norm"),
        ("pretrans_layer0_ffn_out", (n_embd, n_frames), "layer0_mlp_out", "Layer 0 FFN output (before LayerScale)"),
        ("pretrans_layer0_ffn_scaled", (n_embd, n_frames), "layer0_mlp_scaled", "Layer 0 FFN scaled"),
        ("pretrans_layer0_output", (n_embd, n_frames), "layer0_output", "Layer 0 full output"),
        ("pretrans_layer1_output", (n_embd, n_frames), "layer1_output", "Layer 1 full output"),
        ("pretrans_before_output_norm", (n_embd, n_frames), "layer7_output", "After 8 layers (before norm)"),
        ("after_pretrans", (n_embd, n_frames), "final_output", "After output norm"),
    ]

    print(f"\n{'Stage':<45} | {'Corr':>8} | {'Std Ratio':>10} | {'Status':<10}")
    print("-" * 80)

    for cpp_name, cpp_shape, hf_name, desc in comparisons:
        cpp_path = cpp_dir / f"{cpp_name}.bin"
        hf_path = hf_dir / f"{hf_name}.npy"

        if not cpp_path.exists():
            print(f"{desc:<45} | {'---':>8} | {'---':>10} | C++ missing")
            continue
        if not hf_path.exists():
            print(f"{desc:<45} | {'---':>8} | {'---':>10} | HF missing")
            continue

        cpp_arr = load_ggml_tensor(cpp_path, cpp_shape)
        hf_arr = load_hf_tensor(hf_path)

        err, corr, std_ratio = compare(cpp_arr, hf_arr, desc)
        if err:
            print(f"{desc:<45} | {err}")
        else:
            if corr > 0.999 and 0.98 < std_ratio < 1.02:
                status = "PERFECT"
            elif corr > 0.99 and 0.95 < std_ratio < 1.05:
                status = "OK"
            elif corr > 0.95:
                status = "DIVERGING"
            else:
                status = "MISMATCH"

            print(f"{desc:<45} | {corr:>8.6f} | {std_ratio:>10.4f} | {status:<10}")

    # Additional: compute intermediate for attention residual
    print("\n=== Detailed Layer 0 Analysis ===")

    # Check attention residual
    cpp_input = load_ggml_tensor(cpp_dir / "pretrans_layer0_input.bin", (n_embd, n_frames))
    hf_input = load_hf_tensor(hf_dir / "00_embd_mean.npy")
    if hf_input.shape[0] == n_frames:
        hf_input = hf_input.T

    hf_attn_scaled = load_hf_tensor(hf_dir / "layer0_attn_scaled.npy")
    if hf_attn_scaled.shape[0] == n_frames:
        hf_attn_scaled = hf_attn_scaled.T

    # HF after_attn_res = input + attn_scaled
    hf_after_attn_res = hf_input + hf_attn_scaled

    cpp_after_attn_res = load_ggml_tensor(cpp_dir / "pretrans_layer0_after_attn_res.bin", (n_embd, n_frames))

    corr = np.corrcoef(cpp_after_attn_res.flatten(), hf_after_attn_res.flatten())[0, 1]
    std_ratio = cpp_after_attn_res.std() / hf_after_attn_res.std()
    print(f"Layer 0 after attention residual: corr={corr:.6f}, std_ratio={std_ratio:.4f}")

    # Check FFN residual
    cpp_ffn_scaled = load_ggml_tensor(cpp_dir / "pretrans_layer0_ffn_scaled.bin", (n_embd, n_frames))
    hf_ffn_scaled = load_hf_tensor(hf_dir / "layer0_mlp_scaled.npy")
    if hf_ffn_scaled.shape[0] == n_frames:
        hf_ffn_scaled = hf_ffn_scaled.T

    # HF layer0_output = after_attn_res + ffn_scaled
    hf_layer0_output = hf_after_attn_res + hf_ffn_scaled

    cpp_layer0_output = load_ggml_tensor(cpp_dir / "pretrans_layer0_output.bin", (n_embd, n_frames))

    corr = np.corrcoef(cpp_layer0_output.flatten(), hf_layer0_output.flatten())[0, 1]
    std_ratio = cpp_layer0_output.std() / hf_layer0_output.std()
    print(f"Layer 0 full output (computed): corr={corr:.6f}, std_ratio={std_ratio:.4f}")

    # Compare intermediate stds
    print("\n=== Std Comparison ===")
    print(f"{'Stage':<35} | {'C++ std':>10} | {'HF std':>10} | {'Ratio':>8}")
    print("-" * 70)

    stages = [
        ("pretrans_layer0_input", "00_embd_mean", "Input"),
        ("pretrans_layer0_output", "layer0_output", "Layer 0 output"),
        ("pretrans_layer1_output", "layer1_output", "Layer 1 output"),
        ("pretrans_layer2_output", "layer2_output", "Layer 2 output"),
        ("pretrans_layer3_output", "layer3_output", "Layer 3 output"),
        ("pretrans_layer4_output", "layer4_output", "Layer 4 output"),
        ("pretrans_layer5_output", "layer5_output", "Layer 5 output"),
        ("pretrans_layer6_output", "layer6_output", "Layer 6 output"),
        ("pretrans_before_output_norm", "layer7_output", "Layer 7 / Before norm"),
    ]

    for cpp_name, hf_name, desc in stages:
        cpp_path = cpp_dir / f"{cpp_name}.bin"
        hf_path = hf_dir / f"{hf_name}.npy"

        if cpp_path.exists() and hf_path.exists():
            cpp_arr = load_ggml_tensor(cpp_path, (n_embd, n_frames))
            hf_arr = load_hf_tensor(hf_path)
            cpp_std = cpp_arr.std()
            hf_std = hf_arr.std()
            ratio = cpp_std / hf_std
            print(f"{desc:<35} | {cpp_std:>10.6f} | {hf_std:>10.6f} | {ratio:>8.4f}")


if __name__ == "__main__":
    main()
