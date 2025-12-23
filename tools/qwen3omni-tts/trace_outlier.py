#!/usr/bin/env python3
"""Trace the outlier in Token 0 Layer 2 to find its source."""

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

    print("=" * 80)
    print("Tracing Token 0 Outlier at Layer 2")
    print("=" * 80)

    # Layer 1 output (input to Layer 2)
    l1_out = load_cpp_tensor(cpp_dir / "hidden_layer1.bin")
    l1_t0 = l1_out[0] if l1_out.ndim == 2 else l1_out
    print(f"\nLayer 1 output Token 0:")
    print(f"  Shape: {l1_t0.shape}")
    print(f"  mean={l1_t0.mean():.6f}, std={l1_t0.std():.6f}")
    print(f"  range: [{l1_t0.min():.4f}, {l1_t0.max():.4f}]")
    l1_max_idx = np.argmax(np.abs(l1_t0))
    print(f"  max|value| at index {l1_max_idx}: {l1_t0[l1_max_idx]:.4f}")

    # Layer 2 ffn_inp (after attention residual)
    ffn_inp = load_cpp_tensor(cpp_dir / "ffn_inp_layer2.bin")
    inp_t0 = ffn_inp[0] if ffn_inp.ndim == 2 else ffn_inp
    print(f"\nLayer 2 ffn_inp Token 0 (after attention):")
    print(f"  mean={inp_t0.mean():.6f}, std={inp_t0.std():.6f}")
    print(f"  range: [{inp_t0.min():.4f}, {inp_t0.max():.4f}]")
    inp_max_idx = np.argmax(np.abs(inp_t0))
    print(f"  max|value| at index {inp_max_idx}: {inp_t0[inp_max_idx]:.4f}")

    # Layer 2 ffn_norm (after RMSNorm)
    ffn_norm = load_cpp_tensor(cpp_dir / "ffn_norm_layer2.bin")
    norm_t0 = ffn_norm[0] if ffn_norm.ndim == 2 else ffn_norm
    print(f"\nLayer 2 ffn_norm Token 0 (after RMSNorm):")
    print(f"  mean={norm_t0.mean():.6f}, std={norm_t0.std():.6f}")
    print(f"  range: [{norm_t0.min():.4f}, {norm_t0.max():.4f}]")
    norm_max_idx = np.argmax(np.abs(norm_t0))
    print(f"  max|value| at index {norm_max_idx}: {norm_t0[norm_max_idx]:.4f}")

    # Manually compute RMSNorm to verify
    print("\n" + "=" * 80)
    print("Verifying RMSNorm computation")
    print("=" * 80)

    # Load ffn_norm weights
    import sys
    sys.path.insert(0, '/app/gguf-py')
    from gguf import GGUFReader

    gguf_path = "/models/qwen3-omni-30b-talker-f16-v4.gguf"
    reader = GGUFReader(gguf_path)

    ffn_norm_weight = None
    for t in reader.tensors:
        if t.name == 'blk.2.ffn_norm.weight':
            ffn_norm_weight = t.data.astype(np.float32)
            break

    if ffn_norm_weight is not None:
        print(f"\nffn_norm.weight shape: {ffn_norm_weight.shape}")
        print(f"  mean={ffn_norm_weight.mean():.6f}, std={ffn_norm_weight.std():.6f}")
        print(f"  range: [{ffn_norm_weight.min():.4f}, {ffn_norm_weight.max():.4f}]")

        # Compute RMSNorm: x / rms(x) * weight
        eps = 1e-6
        rms = np.sqrt(np.mean(inp_t0 ** 2) + eps)
        computed_norm = (inp_t0 / rms) * ffn_norm_weight

        print(f"\nManual RMSNorm computation:")
        print(f"  RMS of input: {rms:.6f}")
        print(f"  Computed output mean={computed_norm.mean():.6f}, std={computed_norm.std():.6f}")
        print(f"  Computed output range: [{computed_norm.min():.4f}, {computed_norm.max():.4f}]")

        # Compare with actual C++ output
        diff = np.abs(norm_t0 - computed_norm)
        print(f"\n  Comparison with C++ ffn_norm:")
        print(f"  Max diff: {diff.max():.6f}")
        print(f"  Mean diff: {diff.mean():.6f}")
        corr = np.corrcoef(norm_t0, computed_norm)[0, 1]
        print(f"  Correlation: {corr:.6f}")

        # Check the outlier dimension
        print(f"\n  At outlier index {norm_max_idx}:")
        print(f"    C++ value: {norm_t0[norm_max_idx]:.6f}")
        print(f"    Computed value: {computed_norm[norm_max_idx]:.6f}")
        print(f"    Input value: {inp_t0[norm_max_idx]:.6f}")
        print(f"    Weight value: {ffn_norm_weight[norm_max_idx]:.6f}")

    # Compare Token 0 with other tokens
    print("\n" + "=" * 80)
    print("Token 0 vs Other Tokens Comparison")
    print("=" * 80)

    for tok_idx in range(ffn_norm.shape[0]):
        tok = ffn_norm[tok_idx]
        max_abs = np.abs(tok).max()
        max_idx = np.argmax(np.abs(tok))
        print(f"  Token {tok_idx}: std={tok.std():.4f}, max|val|={max_abs:.4f} at idx {max_idx}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
