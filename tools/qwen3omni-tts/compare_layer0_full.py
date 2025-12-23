#!/usr/bin/env python3
"""Full Layer 0 comparison - identify where error starts."""

import struct
import numpy as np
from pathlib import Path


def load_cpp_tensor(path):
    with open(path, 'rb') as f:
        ndims = struct.unpack('<I', f.read(4))[0]
        shape = [struct.unpack('<I', f.read(4))[0] for _ in range(ndims)]
        data = np.frombuffer(f.read(), dtype='<f4')
        return data.reshape(shape)


def compare(name, hf_arr, cpp_arr, token_idx=-1):
    """Compare HF and C++ arrays for a specific token."""
    if hf_arr.ndim == 3:
        hf_tok = hf_arr[0, token_idx]
    else:
        hf_tok = hf_arr

    if cpp_arr.ndim == 2:
        cpp_tok = cpp_arr[token_idx]
    else:
        cpp_tok = cpp_arr

    if hf_tok.shape != cpp_tok.shape:
        print(f"  {name}: Shape mismatch HF={hf_tok.shape} vs C++={cpp_tok.shape}")
        return

    corr = np.corrcoef(hf_tok.flatten(), cpp_tok.flatten())[0, 1]
    diff = np.abs(hf_tok - cpp_tok)

    print(f"  {name}:")
    print(f"    HF:  mean={hf_tok.mean():.6f}, std={hf_tok.std():.6f}")
    print(f"    C++: mean={cpp_tok.mean():.6f}, std={cpp_tok.std():.6f}")
    print(f"    Correlation: {corr:.6f}, Max diff: {diff.max():.6f}")

    return corr


def main():
    cpp_dir = Path("/models/debug/cpp_talker")
    hf_dir = Path("/models/debug/hf_talker")

    print("=" * 80)
    print("Full Layer 0 Comparison: Finding Error Source")
    print("=" * 80)

    # Input embeddings
    print("\n=== Input (Prefill Embeddings) ===")
    hf_inp = np.load(hf_dir / "prefill_embeds.npy")
    cpp_inp = load_cpp_tensor(cpp_dir / "prefill_embeds.bin")
    print(f"Shape: HF={hf_inp.shape}, C++={cpp_inp.shape}")

    for tok_idx in [0, -1]:
        name = "Token 0" if tok_idx == 0 else "Token 8 (last)"
        print(f"\n--- {name} ---")

        # Prefill embeddings
        hf_tok = hf_inp[tok_idx]
        cpp_tok = cpp_inp[tok_idx]
        corr = np.corrcoef(hf_tok.flatten(), cpp_tok.flatten())[0, 1]
        print(f"  Input: Correlation={corr:.6f}")

        # Layer 0 attn_norm_out (input_layernorm output)
        hf_attn_norm = np.load(hf_dir / "hooked_layer0_attn_norm_out.npy")
        # C++ doesn't have attn_norm saved separately

        # Layer 0 attn_out (attention output before residual)
        hf_attn_out = np.load(hf_dir / "hooked_layer0_attn_out.npy")
        print(f"  HF attn_out: mean={hf_attn_out[0, tok_idx].mean():.6f}, std={hf_attn_out[0, tok_idx].std():.6f}")

        # Layer 0 ffn_norm_out (after RMSNorm)
        hf_ffn_norm = np.load(hf_dir / "hooked_layer0_ffn_norm_out.npy")
        print(f"  HF ffn_norm_out: mean={hf_ffn_norm[0, tok_idx].mean():.6f}, std={hf_ffn_norm[0, tok_idx].std():.6f}")

        # Layer 0 mlp_out
        hf_mlp = np.load(hf_dir / "hooked_layer0_mlp_out.npy")
        print(f"  HF mlp_out: mean={hf_mlp[0, tok_idx].mean():.6f}, std={hf_mlp[0, tok_idx].std():.6f}")

        # Layer 0 output
        hf_out = np.load(hf_dir / "hooked_layer0_out.npy")
        cpp_out = load_cpp_tensor(cpp_dir / "hidden_layer0.bin")

        hf_tok_out = hf_out[0, tok_idx]
        cpp_tok_out = cpp_out[tok_idx]
        corr = np.corrcoef(hf_tok_out.flatten(), cpp_tok_out.flatten())[0, 1]
        print(f"  Layer 0 output:")
        print(f"    HF:  mean={hf_tok_out.mean():.6f}, std={hf_tok_out.std():.6f}")
        print(f"    C++: mean={cpp_tok_out.mean():.6f}, std={cpp_tok_out.std():.6f}")
        print(f"    Correlation: {corr:.6f}")

    # Check all tokens at Layer 0
    print("\n" + "=" * 80)
    print("Layer 0 Output: All Tokens")
    print("=" * 80)

    for tok_idx in range(9):
        hf_tok = hf_out[0, tok_idx]
        cpp_tok = cpp_out[tok_idx]
        corr = np.corrcoef(hf_tok.flatten(), cpp_tok.flatten())[0, 1]
        print(f"  Token {tok_idx}: Correlation={corr:.6f}")

    # Also check Layer 1
    print("\n" + "=" * 80)
    print("Layer 1 Output: All Tokens")
    print("=" * 80)

    hf_l1 = np.load(hf_dir / "hooked_layer1_out.npy")
    cpp_l1 = load_cpp_tensor(cpp_dir / "hidden_layer1.bin")

    for tok_idx in range(9):
        hf_tok = hf_l1[0, tok_idx]
        cpp_tok = cpp_l1[tok_idx]
        corr = np.corrcoef(hf_tok.flatten(), cpp_tok.flatten())[0, 1]
        print(f"  Token {tok_idx}: Correlation={corr:.6f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
