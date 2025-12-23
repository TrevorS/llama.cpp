#!/usr/bin/env python3
"""Compare all tokens across all layers to find error pattern."""

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
    print("All Tokens, All Layers Correlation Matrix")
    print("=" * 80)
    print()

    # Header
    print(f"{'Layer':<8}", end="")
    for tok_idx in range(9):
        print(f"Tok{tok_idx:d}    ", end="")
    print()
    print("-" * 80)

    # HF has hooked outputs for layers 0-3
    # C++ has hidden_layer*.bin for all 20 layers
    # But HF old-style hidden_layer*.npy files only have last token

    # For layers 0-3, use hooked data (all tokens)
    for layer_idx in range(4):
        hf_path = hf_dir / f"hooked_layer{layer_idx}_out.npy"
        cpp_path = cpp_dir / f"hidden_layer{layer_idx}.bin"

        if hf_path.exists() and cpp_path.exists():
            hf_layer = np.load(hf_path)
            cpp_layer = load_cpp_tensor(cpp_path)

            print(f"L{layer_idx:<7}", end="")
            for tok_idx in range(9):
                hf_tok = hf_layer[0, tok_idx]
                cpp_tok = cpp_layer[tok_idx]
                corr = np.corrcoef(hf_tok.flatten(), cpp_tok.flatten())[0, 1]
                print(f"{corr:.4f}  ", end="")
            print()

    print()
    print("=" * 80)
    print("Layers 4-19: Last Token Only (from old HF captures)")
    print("=" * 80)

    for layer_idx in range(4, 20):
        hf_path = hf_dir / f"hidden_layer{layer_idx}.npy"
        cpp_path = cpp_dir / f"hidden_layer{layer_idx}.bin"

        if hf_path.exists() and cpp_path.exists():
            hf_last = np.load(hf_path)  # Only last token
            cpp_layer = load_cpp_tensor(cpp_path)
            cpp_last = cpp_layer[-1] if cpp_layer.ndim == 2 else cpp_layer

            if hf_last.shape == cpp_last.shape:
                corr = np.corrcoef(hf_last.flatten(), cpp_last.flatten())[0, 1]
                print(f"  Layer {layer_idx:2d}: Correlation={corr:.6f}")

    # Final logits comparison
    print()
    print("=" * 80)
    print("Final Logits")
    print("=" * 80)

    hf_logits_path = hf_dir / "prefill_logits.npy"
    cpp_logits_path = cpp_dir / "prefill_logits.bin"

    if hf_logits_path.exists() and cpp_logits_path.exists():
        hf_logits = np.load(hf_logits_path)
        cpp_logits = load_cpp_tensor(cpp_logits_path)

        print(f"HF logits shape: {hf_logits.shape}")
        print(f"C++ logits shape: {cpp_logits.shape}")

        # Compare last token logits (which is what matters for sampling)
        hf_last = hf_logits[-1] if hf_logits.ndim == 2 else hf_logits.flatten()
        cpp_last = cpp_logits[-1] if cpp_logits.ndim == 2 else cpp_logits.flatten()

        if hf_last.shape[0] == cpp_last.shape[0]:
            corr = np.corrcoef(hf_last, cpp_last)[0, 1]
            print(f"Last token logits correlation: {corr:.6f}")

            # Top-10 predictions
            hf_top10 = np.argsort(hf_last)[-10:][::-1]
            cpp_top10 = np.argsort(cpp_last)[-10:][::-1]

            print("\nTop-10 token predictions:")
            print("HF:  ", hf_top10)
            print("C++: ", cpp_top10)

            # Check if top-1 matches
            if hf_top10[0] == cpp_top10[0]:
                print("\n✓ Top-1 prediction matches!")
            else:
                print(f"\n✗ Top-1 mismatch: HF={hf_top10[0]}, C++={cpp_top10[0]}")

    print()


if __name__ == "__main__":
    main()
