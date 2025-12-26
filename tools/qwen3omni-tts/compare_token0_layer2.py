#!/usr/bin/env python3
"""Compare Token 0 Layer 2 outputs between C++ and HF."""

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
    print("Token 0 Layer 2 Comparison: C++ vs HuggingFace")
    print("=" * 80)

    # Load HF hooked outputs (all tokens)
    hf_ffn_norm = np.load(hf_dir / "hooked_layer2_ffn_norm_out.npy")
    hf_mlp = np.load(hf_dir / "hooked_layer2_mlp_out.npy")
    hf_layer = np.load(hf_dir / "hooked_layer2_out.npy")

    print(f"\nHF shapes: ffn_norm={hf_ffn_norm.shape}, mlp={hf_mlp.shape}, layer={hf_layer.shape}")

    # Load C++ outputs
    cpp_ffn_norm = load_cpp_tensor(cpp_dir / "ffn_norm_layer2.bin")
    cpp_ffn_moe = load_cpp_tensor(cpp_dir / "ffn_moe_out_layer2.bin")
    cpp_hidden = load_cpp_tensor(cpp_dir / "hidden_layer2.bin")

    print(f"C++ shapes: ffn_norm={cpp_ffn_norm.shape}, moe={cpp_ffn_moe.shape}, hidden={cpp_hidden.shape}")

    # Compare Token 0 ffn_norm (after RMSNorm, before MoE)
    print("\n" + "=" * 80)
    print("Layer 2 Token 0 ffn_norm (after RMSNorm)")
    print("=" * 80)

    hf_t0_norm = hf_ffn_norm[0, 0]
    cpp_t0_norm = cpp_ffn_norm[0] if cpp_ffn_norm.ndim == 2 else cpp_ffn_norm

    print(f"HF:  mean={hf_t0_norm.mean():.6f}, std={hf_t0_norm.std():.6f}")
    print(f"C++: mean={cpp_t0_norm.mean():.6f}, std={cpp_t0_norm.std():.6f}")

    if hf_t0_norm.shape == cpp_t0_norm.shape:
        corr = np.corrcoef(hf_t0_norm.flatten(), cpp_t0_norm.flatten())[0, 1]
        diff = np.abs(hf_t0_norm - cpp_t0_norm)
        print(f"\nCorrelation: {corr:.6f}")
        print(f"Mean diff: {diff.mean():.6f}")
        print(f"Max diff: {diff.max():.6f}")

        # Show first 10 values
        print(f"\nFirst 10 values:")
        print(f"HF:  {hf_t0_norm[:10]}")
        print(f"C++: {cpp_t0_norm[:10]}")

    # Compare Token 0 MLP output
    print("\n" + "=" * 80)
    print("Layer 2 Token 0 MLP output (MoE)")
    print("=" * 80)

    hf_t0_mlp = hf_mlp[0, 0]
    cpp_t0_mlp = cpp_ffn_moe[0] if cpp_ffn_moe.ndim == 2 else cpp_ffn_moe

    print(f"HF:  mean={hf_t0_mlp.mean():.6f}, std={hf_t0_mlp.std():.6f}")
    print(f"C++: mean={cpp_t0_mlp.mean():.6f}, std={cpp_t0_mlp.std():.6f}")

    if hf_t0_mlp.shape == cpp_t0_mlp.shape:
        corr = np.corrcoef(hf_t0_mlp.flatten(), cpp_t0_mlp.flatten())[0, 1]
        diff = np.abs(hf_t0_mlp - cpp_t0_mlp)
        print(f"\nCorrelation: {corr:.6f}")
        print(f"Mean diff: {diff.mean():.6f}")
        print(f"Max diff: {diff.max():.6f}")

    # Compare Token 0 layer output
    print("\n" + "=" * 80)
    print("Layer 2 Token 0 layer output (after residual)")
    print("=" * 80)

    hf_t0_layer = hf_layer[0, 0]
    cpp_t0_layer = cpp_hidden[0] if cpp_hidden.ndim == 2 else cpp_hidden

    print(f"HF:  mean={hf_t0_layer.mean():.6f}, std={hf_t0_layer.std():.6f}")
    print(f"C++: mean={cpp_t0_layer.mean():.6f}, std={cpp_t0_layer.std():.6f}")

    if hf_t0_layer.shape == cpp_t0_layer.shape:
        corr = np.corrcoef(hf_t0_layer.flatten(), cpp_t0_layer.flatten())[0, 1]
        diff = np.abs(hf_t0_layer - cpp_t0_layer)
        print(f"\nCorrelation: {corr:.6f}")
        print(f"Mean diff: {diff.mean():.6f}")
        print(f"Max diff: {diff.max():.6f}")

    # Compare last token (Token 8) which should be "normal"
    print("\n" + "=" * 80)
    print("Layer 2 Token 8 (Last Token) Comparison")
    print("=" * 80)

    hf_t8_mlp = hf_mlp[0, -1]
    cpp_t8_mlp = cpp_ffn_moe[-1] if cpp_ffn_moe.ndim == 2 else None

    if cpp_t8_mlp is not None:
        print(f"HF:  mean={hf_t8_mlp.mean():.6f}, std={hf_t8_mlp.std():.6f}")
        print(f"C++: mean={cpp_t8_mlp.mean():.6f}, std={cpp_t8_mlp.std():.6f}")

        if hf_t8_mlp.shape == cpp_t8_mlp.shape:
            corr = np.corrcoef(hf_t8_mlp.flatten(), cpp_t8_mlp.flatten())[0, 1]
            print(f"Correlation: {corr:.6f}")
    else:
        print("C++ only saved last token, checking it:")
        print(f"HF Token 8:  mean={hf_t8_mlp.mean():.6f}, std={hf_t8_mlp.std():.6f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
