#!/usr/bin/env python3
"""Compare last token (Token 8) across layers between C++ and HF."""

import struct
import numpy as np
from pathlib import Path


def load_cpp_tensor(path):
    with open(path, 'rb') as f:
        ndims = struct.unpack('<I', f.read(4))[0]
        shape = [struct.unpack('<I', f.read(4))[0] for _ in range(ndims)]
        data = np.frombuffer(f.read(), dtype='<f4')
        return data.reshape(shape)


def main():
    cpp_dir = Path("/models/debug/cpp_talker")
    hf_dir = Path("/models/debug/hf_talker")

    print("=" * 80)
    print("Last Token (Token 8) Comparison Across Layers")
    print("=" * 80)

    # Compare each layer's output for the last token
    for layer_idx in range(4):
        print(f"\n--- Layer {layer_idx} ---")

        # HF layer output (from hooks)
        hf_path = hf_dir / f"hooked_layer{layer_idx}_out.npy"
        if hf_path.exists():
            hf_layer = np.load(hf_path)
            hf_last = hf_layer[0, -1]  # batch 0, last token
        else:
            # Try old-style files
            hf_path = hf_dir / f"hidden_layer{layer_idx}.npy"
            if hf_path.exists():
                hf_last = np.load(hf_path)  # Already just last token
            else:
                print(f"  HF: not found")
                continue

        # C++ layer output
        cpp_path = cpp_dir / f"hidden_layer{layer_idx}.bin"
        if cpp_path.exists():
            cpp_layer = load_cpp_tensor(cpp_path)
            cpp_last = cpp_layer[-1] if cpp_layer.ndim == 2 else cpp_layer
        else:
            print(f"  C++: not found")
            continue

        print(f"  HF:  mean={hf_last.mean():.6f}, std={hf_last.std():.6f}")
        print(f"  C++: mean={cpp_last.mean():.6f}, std={cpp_last.std():.6f}")

        if hf_last.shape == cpp_last.shape:
            corr = np.corrcoef(hf_last.flatten(), cpp_last.flatten())[0, 1]
            diff = np.abs(hf_last - cpp_last)
            print(f"  Correlation: {corr:.6f}")
            print(f"  Max diff: {diff.max():.6f}")

    # Also compare ffn_norm for layer 2
    print("\n" + "=" * 80)
    print("Layer 2 Last Token ffn_norm (after RMSNorm, before MoE)")
    print("=" * 80)

    hf_norm = np.load(hf_dir / "hooked_layer2_ffn_norm_out.npy")
    hf_last_norm = hf_norm[0, -1]

    cpp_norm = load_cpp_tensor(cpp_dir / "ffn_norm_layer2.bin")
    cpp_last_norm = cpp_norm[-1]

    print(f"HF:  mean={hf_last_norm.mean():.6f}, std={hf_last_norm.std():.6f}")
    print(f"C++: mean={cpp_last_norm.mean():.6f}, std={cpp_last_norm.std():.6f}")

    corr = np.corrcoef(hf_last_norm.flatten(), cpp_last_norm.flatten())[0, 1]
    print(f"Correlation: {corr:.6f}")

    # First 10 values
    print(f"\nFirst 10 values:")
    print(f"HF:  {hf_last_norm[:10]}")
    print(f"C++: {cpp_last_norm[:10]}")

    # Compare ffn_inp (before RMSNorm)
    print("\n" + "=" * 80)
    print("Layer 2 Last Token ffn_inp (before RMSNorm)")
    print("=" * 80)

    cpp_inp = load_cpp_tensor(cpp_dir / "ffn_inp_layer2.bin")
    cpp_last_inp = cpp_inp[-1]
    print(f"C++: mean={cpp_last_inp.mean():.6f}, std={cpp_last_inp.std():.6f}")
    print(f"C++ first 10: {cpp_last_inp[:10]}")

    # Compare with Layer 1 output for last token
    print("\n(HF doesn't have ffn_inp, comparing with Layer 1 output...)")

    hf_l1 = np.load(hf_dir / "hooked_layer1_out.npy")
    hf_l1_last = hf_l1[0, -1]
    print(f"HF L1 out: mean={hf_l1_last.mean():.6f}, std={hf_l1_last.std():.6f}")
    print(f"HF L1 first 10: {hf_l1_last[:10]}")

    cpp_l1 = load_cpp_tensor(cpp_dir / "hidden_layer1.bin")
    cpp_l1_last = cpp_l1[-1]
    print(f"C++ L1 out: mean={cpp_l1_last.mean():.6f}, std={cpp_l1_last.std():.6f}")
    print(f"C++ L1 first 10: {cpp_l1_last[:10]}")

    corr = np.corrcoef(hf_l1_last.flatten(), cpp_l1_last.flatten())[0, 1]
    print(f"L1 last token correlation: {corr:.6f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
