#!/usr/bin/env python3
"""Trace where the 0.72 std ratio originates in the pre-transformer layers."""

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


def main():
    cpp_dir = Path("/models/debug/cpp_tokens_match")
    hf_dir = Path("/models/debug/hf_tensors_16f")

    n_frames = 16
    n_embd = 1024

    print("=" * 60)
    print("Tracing Std Ratio Through Pre-Transformer")
    print("=" * 60)

    # Input (embeddings)
    print("\n=== Embeddings (Input) ===\n")
    hf_embd = load_hf_tensor(hf_dir / "02_embd_mean.npy")
    cpp_layer0_input = load_ggml_tensor(cpp_dir / "pretrans_layer0_input.bin", (n_embd, n_frames))

    # Check if transpose needed
    if hf_embd.shape[0] == n_frames:
        hf_embd = hf_embd.T

    cpp_std = cpp_layer0_input.std()
    hf_std = hf_embd.std()
    ratio = cpp_std / hf_std
    corr = np.corrcoef(cpp_layer0_input.flatten(), hf_embd.flatten())[0, 1]

    print(f"C++ layer0_input: std={cpp_std:.6f}")
    print(f"HF embd_mean:     std={hf_std:.6f}")
    print(f"Std ratio:        {ratio:.6f}")
    print(f"Correlation:      {corr:.6f}")

    # After attention norm (layer 0)
    print("\n=== After Attention RMSNorm (Layer 0) ===\n")
    cpp_after_attn_norm = load_ggml_tensor(cpp_dir / "pretrans_layer0_after_attn_norm.bin", (n_embd, n_frames))
    print(f"C++ after_attn_norm: std={cpp_after_attn_norm.std():.6f}")
    print("(No HF reference for this intermediate tensor)")

    # After attention residual (layer 0)
    print("\n=== After Attention Residual (Layer 0) ===\n")
    cpp_after_attn_res = load_ggml_tensor(cpp_dir / "pretrans_layer0_after_attn_res.bin", (n_embd, n_frames))
    print(f"C++ after_attn_res: std={cpp_after_attn_res.std():.6f}")
    print("(No HF reference for this intermediate tensor)")

    # Before output norm (after all 8 layers)
    print("\n=== Before Output Norm (After 8 Layers) ===\n")
    cpp_before_norm = load_ggml_tensor(cpp_dir / "pretrans_before_output_norm.bin", (n_embd, n_frames))
    print(f"C++ before_norm: std={cpp_before_norm.std():.6f}")
    print("(No HF reference for this intermediate tensor)")

    # After output norm
    print("\n=== After Output Norm (Final Pre-Transformer Output) ===\n")
    cpp_after_norm = load_ggml_tensor(cpp_dir / "after_pretrans.bin", (n_embd, n_frames))
    hf_after_norm = load_hf_tensor(hf_dir / "03_pre_xfmr_out.npy")

    if hf_after_norm.shape[0] == n_frames:
        hf_after_norm = hf_after_norm.T

    cpp_std = cpp_after_norm.std()
    hf_std = hf_after_norm.std()
    ratio = cpp_std / hf_std
    corr = np.corrcoef(cpp_after_norm.flatten(), hf_after_norm.flatten())[0, 1]

    print(f"C++ after_norm:   std={cpp_std:.6f}")
    print(f"HF pre_xfmr_out:  std={hf_std:.6f}")
    print(f"Std ratio:        {ratio:.6f}")
    print(f"Correlation:      {corr:.6f}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key observations:
1. Input (embeddings): std ratio = 1.00 (perfect match)
2. After 8 layers + output norm: std ratio = 0.72

Since input is matched but output is 28% lower variance, the divergence
happens WITHIN the 8 transformer layers.

To find the exact source, we need to:
1. Generate HF intermediate tensors (after each layer)
2. Compare layer-by-layer

The HF Code2Wav debugging script should dump:
- After each layer's attention (before residual)
- After each layer's FFN (before residual)
- After each layer (complete)

This will pinpoint which layer/operation causes the variance reduction.
""")

    # Ratio progression
    print("=== Std Progression Through C++ Pipeline ===\n")
    print(f"1. Layer 0 input:      {cpp_layer0_input.std():.6f}")
    print(f"2. After attn norm:    {cpp_after_attn_norm.std():.6f}")
    print(f"3. After attn res:     {cpp_after_attn_res.std():.6f}")
    print(f"4. Before output norm: {cpp_before_norm.std():.6f}")
    print(f"5. After output norm:  {cpp_after_norm.std():.6f}")


if __name__ == "__main__":
    main()
