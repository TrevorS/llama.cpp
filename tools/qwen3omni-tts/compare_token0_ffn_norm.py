#!/usr/bin/env python3
"""Compare Token 0 ffn_norm between C++ and HF at Layer 2."""

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
    print("Token 0 ffn_norm Comparison at Layer 2")
    print("=" * 80)

    # Load C++ ffn_norm
    cpp_norm = load_cpp_tensor(cpp_dir / "ffn_norm_layer2.bin")
    print(f"\nC++ ffn_norm shape: {cpp_norm.shape}")

    cpp_token0 = cpp_norm[0] if cpp_norm.ndim == 2 else cpp_norm
    print(f"C++ Token 0: mean={cpp_token0.mean():.6f}, std={cpp_token0.std():.6f}")
    print(f"C++ Token 0 first 10: {cpp_token0[:10]}")
    print(f"C++ Token 0 range: [{cpp_token0.min():.4f}, {cpp_token0.max():.4f}]")

    # Load HF ffn_norm (if available)
    hf_path = hf_dir / "ffn_norm_layer2.npy"
    if hf_path.exists():
        hf_norm = np.load(hf_path)
        print(f"\nHF ffn_norm shape: {hf_norm.shape}")
        print(f"HF Token 0: mean={hf_norm.mean():.6f}, std={hf_norm.std():.6f}")
        print(f"HF Token 0 first 10: {hf_norm[:10]}")
        print(f"HF Token 0 range: [{hf_norm.min():.4f}, {hf_norm.max():.4f}]")

        # Compare
        if cpp_token0.shape == hf_norm.shape:
            corr = np.corrcoef(cpp_token0.flatten(), hf_norm.flatten())[0, 1]
            diff = np.abs(cpp_token0 - hf_norm)
            print(f"\nCorrelation: {corr:.6f}")
            print(f"Max diff: {diff.max():.6f}")
            print(f"Mean diff: {diff.mean():.6f}")

            if corr < 0.99:
                print("\n*** ffn_norm MISMATCH! ***")
                print("This explains why Expert 93 output explodes!")
    else:
        print(f"\nWARNING: HF ffn_norm not found at {hf_path}")
        print("HF debug script only saved layer outputs, not intermediate ffn_norm")

    # Check ffn_inp as well (before norm)
    print("\n" + "=" * 80)
    print("ffn_inp (before norm) Comparison")
    print("=" * 80)

    cpp_inp = load_cpp_tensor(cpp_dir / "ffn_inp_layer2.bin")
    print(f"\nC++ ffn_inp shape: {cpp_inp.shape}")
    cpp_inp_token0 = cpp_inp[0] if cpp_inp.ndim == 2 else cpp_inp
    print(f"C++ Token 0: mean={cpp_inp_token0.mean():.6f}, std={cpp_inp_token0.std():.6f}")
    print(f"C++ Token 0 first 10: {cpp_inp_token0[:10]}")

    hf_inp_path = hf_dir / "ffn_inp_layer2.npy"
    if hf_inp_path.exists():
        hf_inp = np.load(hf_inp_path)
        print(f"\nHF ffn_inp shape: {hf_inp.shape}")
        print(f"HF Token 0: mean={hf_inp.mean():.6f}, std={hf_inp.std():.6f}")
        print(f"HF Token 0 first 10: {hf_inp[:10]}")

        if cpp_inp_token0.shape == hf_inp.shape:
            corr = np.corrcoef(cpp_inp_token0.flatten(), hf_inp.flatten())[0, 1]
            print(f"\nCorrelation: {corr:.6f}")
            if corr < 0.99:
                print("*** ffn_inp MISMATCH - bug is in attention! ***")
    else:
        print(f"\nWARNING: HF ffn_inp not found at {hf_inp_path}")

    # Check Layer 1 output
    print("\n" + "=" * 80)
    print("Layer 1 Output (input to Layer 2)")
    print("=" * 80)

    cpp_l1 = load_cpp_tensor(cpp_dir / "hidden_layer1.bin")
    print(f"\nC++ hidden_layer1 shape: {cpp_l1.shape}")
    cpp_l1_token0 = cpp_l1[0] if cpp_l1.ndim == 2 else cpp_l1
    print(f"C++ Token 0: mean={cpp_l1_token0.mean():.6f}, std={cpp_l1_token0.std():.6f}")
    print(f"C++ Token 0 first 10: {cpp_l1_token0[:10]}")

    hf_l1_path = hf_dir / "hidden_layer1.npy"
    if hf_l1_path.exists():
        hf_l1 = np.load(hf_l1_path)
        print(f"\nHF hidden_layer1 shape: {hf_l1.shape}")
        print(f"HF: mean={hf_l1.mean():.6f}, std={hf_l1.std():.6f}")

        # Note: HF might only save last token
        if cpp_l1_token0.shape == hf_l1.shape:
            corr = np.corrcoef(cpp_l1_token0.flatten(), hf_l1.flatten())[0, 1]
            print(f"\nCorrelation (Token 0 vs HF): {corr:.6f}")
        else:
            print(f"\nShape mismatch: C++ {cpp_l1_token0.shape} vs HF {hf_l1.shape}")
            print("(HF likely saved only last token)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
