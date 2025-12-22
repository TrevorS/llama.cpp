#!/usr/bin/env python3
"""Compare layer 7 intermediate tensors between C++ and HuggingFace to find amplification source."""

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
        return f"SHAPE MISMATCH {cpp_arr.shape} vs {hf_arr.shape}", 0, 0, 0, 0

    corr = np.corrcoef(cpp_arr.flatten(), hf_arr.flatten())[0, 1]
    cpp_std = cpp_arr.std()
    hf_std = hf_arr.std()
    std_ratio = cpp_std / hf_std
    return None, corr, std_ratio, cpp_std, hf_std


def main():
    cpp_dir = Path("/models/debug/cpp_layer7_debug")
    hf_dir = Path("/models/debug/hf_layers_match")

    n_frames = 16
    n_embd = 1024

    print("=" * 80)
    print("Layer 7 Divergence Analysis: C++ vs HuggingFace")
    print("=" * 80)

    # Layer 7 comparisons
    comparisons = [
        # (cpp_name, cpp_shape, hf_name, description)
        # Input to layer 7 is layer 6 output
        ("pretrans_layer6_output", (n_embd, n_frames), "layer6_output", "Layer 6 output (layer 7 input)"),
        ("pretrans_layer7_after_attn_norm", (n_embd, n_frames), "layer7_attn_norm", "Layer 7 after attn norm"),
        ("pretrans_layer7_attn_out", (n_embd, n_frames), "layer7_attn_out", "Layer 7 attention output"),
        ("pretrans_layer7_attn_scaled", (n_embd, n_frames), "layer7_attn_scaled", "Layer 7 attn scaled"),
        ("pretrans_layer7_ffn_norm", (n_embd, n_frames), "layer7_ffn_norm", "Layer 7 after FFN norm"),
        ("pretrans_layer7_ffn_out", (n_embd, n_frames), "layer7_mlp_out", "Layer 7 FFN output"),
        ("pretrans_layer7_ffn_scaled", (n_embd, n_frames), "layer7_mlp_scaled", "Layer 7 FFN scaled"),
        ("pretrans_before_output_norm", (n_embd, n_frames), "layer7_output", "Layer 7 full output"),
    ]

    print(f"\n{'Stage':<40} | {'Corr':>8} | {'Ratio':>8} | {'C++ std':>10} | {'HF std':>10} | {'Status':<10}")
    print("-" * 100)

    for cpp_name, cpp_shape, hf_name, desc in comparisons:
        cpp_path = cpp_dir / f"{cpp_name}.bin"
        hf_path = hf_dir / f"{hf_name}.npy"

        if not cpp_path.exists():
            print(f"{desc:<40} | {'---':>8} | {'---':>8} | {'---':>10} | {'---':>10} | C++ missing")
            continue
        if not hf_path.exists():
            print(f"{desc:<40} | {'---':>8} | {'---':>8} | {'---':>10} | {'---':>10} | HF missing")
            continue

        cpp_arr = load_ggml_tensor(cpp_path, cpp_shape)
        hf_arr = load_hf_tensor(hf_path)

        err, corr, std_ratio, cpp_std, hf_std = compare(cpp_arr, hf_arr, desc)
        if err:
            print(f"{desc:<40} | {err}")
        else:
            if corr > 0.999 and 0.98 < std_ratio < 1.02:
                status = "PERFECT"
            elif corr > 0.99 and 0.95 < std_ratio < 1.05:
                status = "OK"
            elif corr > 0.95:
                status = "DIVERGING"
            else:
                status = "MISMATCH"

            print(f"{desc:<40} | {corr:>8.6f} | {std_ratio:>8.4f} | {cpp_std:>10.6f} | {hf_std:>10.6f} | {status:<10}")

    # Also check the variance amplification from layer 6 to layer 7
    print("\n" + "=" * 80)
    print("Variance Amplification Analysis (Layer 6 -> Layer 7)")
    print("=" * 80)

    cpp_l6 = load_ggml_tensor(cpp_dir / "pretrans_layer6_output.bin", (n_embd, n_frames))
    cpp_l7 = load_ggml_tensor(cpp_dir / "pretrans_before_output_norm.bin", (n_embd, n_frames))
    hf_l6 = load_hf_tensor(hf_dir / "layer6_output.npy")
    hf_l7 = load_hf_tensor(hf_dir / "layer7_output.npy")

    cpp_amp = cpp_l7.std() / cpp_l6.std()
    hf_amp = hf_l7.std() / hf_l6.std()

    print(f"\nLayer 6 std: C++={cpp_l6.std():.6f}, HF={hf_l6.std():.6f}")
    print(f"Layer 7 std: C++={cpp_l7.std():.6f}, HF={hf_l7.std():.6f}")
    print(f"\nAmplification (L7/L6): C++={cpp_amp:.4f}x, HF={hf_amp:.4f}x")
    print(f"C++ amplifies {cpp_amp/hf_amp:.2f}x more than HF")

    # Check attention contribution vs FFN contribution
    print("\n" + "=" * 80)
    print("Component Contribution Analysis (Layer 7)")
    print("=" * 80)

    # Load layer 7 components
    cpp_attn_scaled = load_ggml_tensor(cpp_dir / "pretrans_layer7_attn_scaled.bin", (n_embd, n_frames))
    cpp_ffn_scaled = load_ggml_tensor(cpp_dir / "pretrans_layer7_ffn_scaled.bin", (n_embd, n_frames))
    hf_attn_scaled = load_hf_tensor(hf_dir / "layer7_attn_scaled.npy")
    hf_ffn_scaled = load_hf_tensor(hf_dir / "layer7_mlp_scaled.npy")

    # Transpose HF if needed
    if hf_attn_scaled.shape[0] == n_frames:
        hf_attn_scaled = hf_attn_scaled.T
    if hf_ffn_scaled.shape[0] == n_frames:
        hf_ffn_scaled = hf_ffn_scaled.T

    print(f"\nAttention scaled std: C++={cpp_attn_scaled.std():.6f}, HF={hf_attn_scaled.std():.6f}, ratio={cpp_attn_scaled.std()/hf_attn_scaled.std():.4f}")
    print(f"FFN scaled std: C++={cpp_ffn_scaled.std():.6f}, HF={hf_ffn_scaled.std():.6f}, ratio={cpp_ffn_scaled.std()/hf_ffn_scaled.std():.4f}")

    # Check attention and FFN correlations
    attn_corr = np.corrcoef(cpp_attn_scaled.flatten(), hf_attn_scaled.flatten())[0, 1]
    ffn_corr = np.corrcoef(cpp_ffn_scaled.flatten(), hf_ffn_scaled.flatten())[0, 1]

    print(f"\nAttention correlation: {attn_corr:.6f}")
    print(f"FFN correlation: {ffn_corr:.6f}")

    # Compute what the layer 7 output SHOULD be
    print("\n" + "=" * 80)
    print("Layer 7 Output Verification")
    print("=" * 80)

    # Layer 7 should be: layer6_output + attn_res + ffn_res
    # where attn_res = attn_scaled and ffn_res = ffn_scaled
    # Actually: layer7_output = layer6_input + attn_scaled + ffn_scaled
    # But the input to layer 7 is layer6_output

    # Let's compute: layer6_output + attn_scaled -> after_attn_res
    cpp_after_attn_res = load_ggml_tensor(cpp_dir / "pretrans_layer7_after_attn_res.bin", (n_embd, n_frames))
    cpp_computed_after_attn = cpp_l6 + cpp_attn_scaled

    res_diff = np.abs(cpp_after_attn_res - cpp_computed_after_attn).mean()
    print(f"\nC++ after_attn_res check: mean diff={res_diff:.10f} (should be ~0)")

    # Similarly: after_attn_res + ffn_scaled -> layer7_output
    cpp_computed_l7 = cpp_after_attn_res + cpp_ffn_scaled
    l7_diff = np.abs(cpp_l7 - cpp_computed_l7).mean()
    print(f"C++ layer7_output check: mean diff={l7_diff:.10f} (should be ~0)")


if __name__ == "__main__":
    main()
