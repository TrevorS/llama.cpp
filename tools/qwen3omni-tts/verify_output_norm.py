#!/usr/bin/env python3
"""Verify output norm weight matches between GGUF and HuggingFace."""

import numpy as np
import os
from pathlib import Path
from safetensors import safe_open
from gguf import GGUFReader


def load_ggml_tensor(path: Path, ne: tuple) -> np.ndarray:
    """Load GGML tensor with Fortran order."""
    data = np.fromfile(path, dtype='<f4')
    expected_size = int(np.prod(ne))
    if len(data) != expected_size:
        raise ValueError(f"Size mismatch: got {len(data)}, expected {expected_size}")
    return data.reshape(ne, order='F')


def rms_norm_manual(x, weight, eps=1e-5):
    """Manual RMSNorm computation.
    x: [hidden, seq]
    weight: [hidden]
    Returns: [hidden, seq]
    """
    # Compute RMS along hidden dimension (axis=0) for each position
    rms = np.sqrt(np.mean(x**2, axis=0, keepdims=True) + eps)
    return x / rms * weight[:, np.newaxis]


def main():
    model_path = "/models/Qwen3-Omni-30B-A3B-Instruct"
    gguf_path = "/models/qwen3-omni-30b-talker-f16-v4.gguf"
    cpp_dir = Path("/models/debug/cpp_tokens_match")
    hf_dir = Path("/models/debug/hf_tensors_16f")

    n_frames = 16
    n_embd = 1024

    print("=" * 60)
    print("Output Norm Weight Verification")
    print("=" * 60)

    # Load from HuggingFace
    print("\n=== Loading from HuggingFace safetensors ===\n")
    hf_output_norm = None
    for f in sorted(os.listdir(model_path)):
        if f.endswith('.safetensors'):
            path = os.path.join(model_path, f)
            with safe_open(path, framework='pt') as sf:
                for name in sf.keys():
                    if name == "code2wav.pre_transformer.norm.weight":
                        hf_output_norm = sf.get_tensor(name).float().numpy()
                        print(f"Found: {name}")
                        print(f"  shape: {hf_output_norm.shape}")
                        print(f"  mean:  {hf_output_norm.mean():.6f}")
                        print(f"  std:   {hf_output_norm.std():.6f}")
                        print(f"  min:   {hf_output_norm.min():.6f}")
                        print(f"  max:   {hf_output_norm.max():.6f}")
                        break
            if hf_output_norm is not None:
                break

    # Load from GGUF
    print("\n=== Loading from GGUF ===\n")
    reader = GGUFReader(gguf_path)
    gguf_output_norm = None
    for tensor in reader.tensors:
        if "output_norm" in tensor.name:
            data = tensor.data
            # Reshape according to tensor shape
            gguf_output_norm = data.reshape(tensor.shape, order='F').flatten()
            print(f"Found: {tensor.name}")
            print(f"  shape: {tensor.shape}")
            print(f"  type:  {tensor.tensor_type}")
            print(f"  mean:  {gguf_output_norm.mean():.6f}")
            print(f"  std:   {gguf_output_norm.std():.6f}")

    # Compare
    if hf_output_norm is not None and gguf_output_norm is not None:
        print("\n=== Comparison ===\n")
        corr = np.corrcoef(hf_output_norm.flatten(), gguf_output_norm.flatten())[0, 1]
        diff = np.abs(hf_output_norm.flatten() - gguf_output_norm).max()
        print(f"Correlation: {corr:.6f}")
        print(f"Max abs diff: {diff:.6f}")

    # Now test the RMSNorm computation
    print("\n=== Testing RMSNorm Computation ===\n")

    cpp_before = load_ggml_tensor(cpp_dir / "pretrans_before_output_norm.bin", (n_embd, n_frames))
    cpp_after = load_ggml_tensor(cpp_dir / "after_pretrans.bin", (n_embd, n_frames))

    print(f"C++ before norm: mean={cpp_before.mean():.6f}, std={cpp_before.std():.6f}")
    print(f"C++ after norm:  mean={cpp_after.mean():.6f}, std={cpp_after.std():.6f}")

    # Apply manual RMSNorm with different eps values
    for eps in [1e-5, 1e-6, 1e-8]:
        manual_out = rms_norm_manual(cpp_before, hf_output_norm, eps=eps)
        diff_std = abs(manual_out.std() - cpp_after.std())
        corr = np.corrcoef(manual_out.flatten(), cpp_after.flatten())[0, 1]
        print(f"\nManual RMSNorm (eps={eps}):")
        print(f"  std:   {manual_out.std():.6f}")
        print(f"  diff:  {diff_std:.6f}")
        print(f"  corr:  {corr:.6f}")

    # Compare with HF
    print("\n=== HuggingFace Reference ===\n")
    hf_after = np.load(hf_dir / "03_pre_xfmr_out.npy")
    if hf_after.ndim > 2 and hf_after.shape[0] == 1:
        hf_after = hf_after[0]
    if hf_after.shape[0] == n_frames:
        hf_after = hf_after.T
    print(f"HF after norm: mean={hf_after.mean():.6f}, std={hf_after.std():.6f}")

    # Key insight: if the norm computation is correct but std is different,
    # then the input to the norm must be different
    print("\n=== Key Insight ===\n")
    print("If RMSNorm is computed correctly, then:")
    print("  C++ std ratio = C++ std / HF std = 0.7249")
    print("\nThis means either:")
    print("  1. The 'before norm' tensor is different (C++ vs HF)")
    print("  2. The norm computation is different")
    print("\nLet's check what the 'before norm' std should be to produce HF's output:")

    # Inverse calculation:
    # HF output = HF_before / RMS(HF_before) * weight
    # If weight is the same, then:
    # HF_before / RMS(HF_before) = HF_output / weight
    # For matching std, HF_before.std() should give HF_output.std()

    # Actually, let's compute what C++ 'before' should produce
    expected_cpp_after = rms_norm_manual(cpp_before, hf_output_norm, eps=1e-5)
    print(f"\nWith C++ 'before' and HF weight:")
    print(f"  Expected C++ 'after' std: {expected_cpp_after.std():.6f}")
    print(f"  Actual C++ 'after' std:   {cpp_after.std():.6f}")
    print(f"  Actual HF 'after' std:    {hf_after.std():.6f}")

    # This tells us if the C++ computation is wrong or if the input is different


if __name__ == "__main__":
    main()
