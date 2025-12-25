#!/usr/bin/env python3
"""Debug the RMS norm discrepancy."""

import struct
import numpy as np
from pathlib import Path


def load_bin(path: Path) -> np.ndarray:
    """Load binary tensor with dimension header."""
    with open(path, 'rb') as f:
        ndims = struct.unpack('<I', f.read(4))[0]
        shape = []
        for _ in range(ndims):
            shape.append(struct.unpack('<I', f.read(4))[0])
        data = np.frombuffer(f.read(), dtype='<f4')
        print(f"  Binary header: ndims={ndims}, shape={shape}")
        return data.reshape(shape)


def main():
    cpp_dir = Path("/models/debug/cpp_talker_cpu")
    hf_dir = Path("/models/debug/hf_talker")

    print("=" * 70)
    print("DEBUG RMS NORM")
    print("=" * 70)

    # Load C++ layer 18
    print("\nLoading C++ layer 18:")
    cpp_l18 = load_bin(cpp_dir / "hidden_layer18.bin")
    print(f"  C++ shape after load: {cpp_l18.shape}")

    # Load HF layer 18
    hf_l18 = np.load(hf_dir / "hidden_layer18.npy")
    print(f"  HF shape: {hf_l18.shape}")

    # Both should be [9, 1024] - no transpose needed
    print("\n=== Shape Analysis ===")
    print(f"  C++ layout: {cpp_l18.shape}")
    print(f"  HF layout:  {hf_l18.shape}")

    corr = np.corrcoef(cpp_l18.flatten(), hf_l18.flatten())[0, 1]
    print(f"  Direct correlation: {corr:.6f}")

    # Check last token
    cpp_last = cpp_l18[-1]  # [1024]
    hf_last = hf_l18[-1]    # [1024]

    print(f"\n=== Last Token ===")
    print(f"  C++ last: mean={cpp_last.mean():.4f}, std={cpp_last.std():.4f}")
    print(f"  HF last:  mean={hf_last.mean():.4f}, std={hf_last.std():.4f}")
    corr_last = np.corrcoef(cpp_last, hf_last)[0, 1]
    print(f"  Correlation: {corr_last:.6f}")

    # RMS computation
    eps = 1e-6
    rms_cpp = np.sqrt(np.mean(cpp_last ** 2) + eps)
    rms_hf = np.sqrt(np.mean(hf_last ** 2) + eps)
    print(f"\n  C++ RMS: {rms_cpp:.6f}")
    print(f"  HF RMS:  {rms_hf:.6f}")

    # Load output_norm weights
    import sys
    sys.path.insert(0, "/app/src/gguf-py")
    from gguf import GGUFReader

    reader = GGUFReader("/models/qwen3-omni-30b-talker-f16-v9.gguf")
    output_norm = None
    for t in reader.tensors:
        if t.name == "output_norm.weight":
            output_norm = t.data.astype(np.float32)
            break

    print(f"  Output norm: shape={output_norm.shape}, mean={output_norm.mean():.4f}")

    # Apply RMS norm
    cpp_normed = cpp_last / rms_cpp * output_norm
    hf_normed_manual = hf_last / rms_hf * output_norm

    print(f"\n=== After RMS Norm (manual) ===")
    print(f"  C++ normed: mean={cpp_normed.mean():.4f}, std={cpp_normed.std():.4f}")
    print(f"  HF manual:  mean={hf_normed_manual.mean():.4f}, std={hf_normed_manual.std():.4f}")
    corr_normed = np.corrcoef(cpp_normed, hf_normed_manual)[0, 1]
    print(f"  Correlation: {corr_normed:.6f}")

    # Compare with actual HF normed output
    hf_after_norm = np.load(hf_dir / "hidden_after_norm.npy")
    hf_normed_actual = hf_after_norm[-1]

    print(f"\n=== HF Actual Normed Output ===")
    print(f"  HF actual: mean={hf_normed_actual.mean():.4f}, std={hf_normed_actual.std():.4f}")
    
    corr_cpp_vs_actual = np.corrcoef(cpp_normed, hf_normed_actual)[0, 1]
    corr_manual_vs_actual = np.corrcoef(hf_normed_manual, hf_normed_actual)[0, 1]
    print(f"  C++ normed vs HF actual: {corr_cpp_vs_actual:.6f}")
    print(f"  HF manual vs HF actual: {corr_manual_vs_actual:.6f}")

    print(f"\n=== First 5 dims ===")
    print(f"  C++ normed:  {cpp_normed[:5]}")
    print(f"  HF manual:   {hf_normed_manual[:5]}")
    print(f"  HF actual:   {hf_normed_actual[:5]}")


if __name__ == "__main__":
    main()
