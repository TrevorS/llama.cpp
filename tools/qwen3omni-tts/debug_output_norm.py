#!/usr/bin/env python3
"""Debug the output norm weights and trace the std ratio source."""

import numpy as np
import os
from pathlib import Path
from safetensors import safe_open


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


def rms_norm_manual(x, weight, eps=1e-5):
    """Manual RMSNorm computation."""
    # x: [hidden, seq]
    # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    rms = np.sqrt(np.mean(x**2, axis=0, keepdims=True) + eps)
    return x / rms * weight[:, np.newaxis]


def main():
    model_path = "/models/Qwen3-Omni-30B-A3B-Instruct"
    cpp_dir = Path("/models/debug/cpp_tokens_match")
    hf_dir = Path("/models/debug/hf_tensors_16f")

    n_frames = 16
    n_embd = 1024

    print("=" * 60)
    print("Output Norm Analysis")
    print("=" * 60)

    # Load pre-transformer output norm weight from HuggingFace
    print("\n=== Loading HuggingFace pre_transformer output norm ===\n")
    output_norm_weight = None

    for f in sorted(os.listdir(model_path)):
        if f.endswith('.safetensors'):
            path = os.path.join(model_path, f)
            with safe_open(path, framework='pt') as sf:
                for name in sf.keys():
                    if 'code2wav' in name and 'pre_transformer' in name and 'norm' in name:
                        if 'output' in name or (name.count('.') < 6):  # Not layer norm
                            tensor = sf.get_tensor(name).float().numpy()
                            print(f"{name}: shape {tensor.shape}")
                            print(f"  mean: {tensor.mean():.6f}")
                            print(f"  std:  {tensor.std():.6f}")
                            if 'output' in name:
                                output_norm_weight = tensor

    # Look for any norm in pre_transformer that's not layer-level
    print("\n=== All pre_transformer norms ===\n")
    for f in sorted(os.listdir(model_path)):
        if f.endswith('.safetensors'):
            path = os.path.join(model_path, f)
            with safe_open(path, framework='pt') as sf:
                for name in sf.keys():
                    if 'code2wav' in name and 'pre_transformer' in name:
                        print(f"{name}")

    # Load C++ tensors
    print("\n=== C++ Tensors ===\n")

    cpp_before = load_ggml_tensor(cpp_dir / "pretrans_before_output_norm.bin", (n_embd, n_frames))
    cpp_after = load_ggml_tensor(cpp_dir / "after_pretrans.bin", (n_embd, n_frames))

    print(f"Before output norm: mean={cpp_before.mean():.6f}, std={cpp_before.std():.6f}")
    print(f"After output norm:  mean={cpp_after.mean():.6f}, std={cpp_after.std():.6f}")

    # Check if we can replicate the RMSNorm manually
    print("\n=== Manual RMSNorm Test ===\n")

    # If we have the output norm weight, apply it
    if output_norm_weight is not None:
        manual_out = rms_norm_manual(cpp_before, output_norm_weight, eps=1e-5)
        print(f"Manual RMSNorm output: mean={manual_out.mean():.6f}, std={manual_out.std():.6f}")
        print(f"C++ output:            mean={cpp_after.mean():.6f}, std={cpp_after.std():.6f}")
        print(f"Difference std: {abs(manual_out.std() - cpp_after.std()):.6f}")

    # Compare with HF
    print("\n=== HuggingFace Comparison ===\n")
    hf_after = load_hf_tensor(hf_dir / "03_pre_xfmr_out.npy")
    if hf_after.shape[0] == n_frames:
        hf_after = hf_after.T  # Transpose to [n_embd, n_frames]
    print(f"HF after output norm: mean={hf_after.mean():.6f}, std={hf_after.std():.6f}")

    # Look at the RMS values
    print("\n=== RMS Analysis ===\n")
    cpp_rms = np.sqrt(np.mean(cpp_before**2, axis=0))
    print(f"C++ RMS per position: {cpp_rms[:5]}...")
    print(f"C++ RMS mean: {cpp_rms.mean():.6f}")

    # What would we expect after RMSNorm?
    # after = before / rms * weight
    # If weight has mean ~0.02, then after std = before std / rms * weight std
    if output_norm_weight is not None:
        expected_std = cpp_before.std() / cpp_rms.mean() * output_norm_weight.mean()
        print(f"\nExpected std after norm (rough): {expected_std:.6f}")
        print(f"Actual C++ std:                   {cpp_after.std():.6f}")
        print(f"Actual HF std:                    {hf_after.std():.6f}")

    # The key question: is the BEFORE output norm tensor the same?
    print("\n=== Key Question: Is 'before output norm' the same? ===\n")
    print("We need to compare C++ 'pretrans_before_output_norm' with HF intermediate")
    print("But HF only gives us 'after output norm' (03_pre_xfmr_out)")
    print("\nTo debug this, we need HF to dump 'before output norm' tensor")


if __name__ == "__main__":
    main()
