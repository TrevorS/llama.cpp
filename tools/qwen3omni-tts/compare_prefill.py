#!/usr/bin/env python3
"""Compare HF and C++ prefill tensors for Talker debugging."""

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
        return data.reshape(shape)


def compare_tensors(name: str, cpp: np.ndarray, hf: np.ndarray):
    """Compare two tensors and report differences."""
    print(f"\n=== {name} ===")
    print(f"  C++ shape: {cpp.shape}")
    print(f"  HF  shape: {hf.shape}")

    if cpp.shape != hf.shape:
        print(f"  SHAPE MISMATCH!")
        return

    diff = np.abs(cpp - hf)
    corr = np.corrcoef(cpp.flatten(), hf.flatten())[0, 1]

    print(f"  Correlation: {corr:.6f}")
    print(f"  Max diff:    {diff.max():.6f}")
    print(f"  Mean diff:   {diff.mean():.6f}")
    print(f"  C++ mean:    {cpp.mean():.6f}, std: {cpp.std():.6f}")
    print(f"  HF  mean:    {hf.mean():.6f}, std: {hf.std():.6f}")

    # First few values
    print(f"  C++ first 8: [{', '.join(f'{x:.4f}' for x in cpp.flatten()[:8])}]")
    print(f"  HF  first 8: [{', '.join(f'{x:.4f}' for x in hf.flatten()[:8])}]")


def main():
    cpp_dir = Path("/models/debug/cpp_talker")
    hf_dir = Path("/models/debug/hf_talker")

    print("=" * 60)
    print("Talker Prefill Comparison: C++ vs HuggingFace")
    print("=" * 60)

    # Load TTS special embeddings
    print("\n--- TTS Special Embeddings ---")

    for name in ["tts_bos_embed", "tts_eos_embed", "tts_pad_embed"]:
        cpp_path = cpp_dir / f"{name}.bin"
        hf_path = hf_dir / f"03_{name}.npy"

        if cpp_path.exists() and hf_path.exists():
            cpp = load_bin(cpp_path)
            hf = np.load(hf_path)
            compare_tensors(name, cpp, hf)

    # Assistant hidden (projected text)
    print("\n--- Assistant Hidden (Projected Text) ---")
    cpp_path = cpp_dir / "assistant_hidden.bin"
    hf_path = hf_dir / "02_assistant_hidden.npy"

    if cpp_path.exists() and hf_path.exists():
        cpp = load_bin(cpp_path)
        hf = np.load(hf_path)
        print(f"\nC++ assistant_hidden shape: {cpp.shape}")
        print(f"HF  assistant_hidden shape: {hf.shape}")

        if cpp.shape != hf.shape:
            print(f"\n!!! MAJOR ISSUE: Different number of text tokens!")
            print(f"    C++ has {cpp.shape[0]} tokens")
            print(f"    HF  has {hf.shape[0]} tokens")

    # Trailing text hidden
    print("\n--- Trailing Text Hidden ---")
    cpp_path = cpp_dir / "trailing_text_hidden.bin"
    hf_path = hf_dir / "04_trailing_text_hidden.npy"

    if cpp_path.exists() and hf_path.exists():
        cpp = load_bin(cpp_path)
        hf = np.load(hf_path)
        print(f"\nC++ trailing_text_hidden shape: {cpp.shape}")
        print(f"HF  trailing_text_hidden shape: {hf.shape}")

    # Full prefill
    print("\n--- Prefill Embeddings ---")
    cpp_path = cpp_dir / "prefill_embeds.bin"
    hf_path = hf_dir / "05_prefill_embeds.npy"

    if cpp_path.exists() and hf_path.exists():
        cpp = load_bin(cpp_path)
        hf = np.load(hf_path)
        compare_tensors("prefill_embeds", cpp, hf)

        # Per-position comparison
        print("\n--- Per-Position Prefill Comparison ---")
        for pos in range(9):
            hf_pos_path = hf_dir / f"05_prefill_pos{pos}.npy"
            if hf_pos_path.exists():
                hf_pos = np.load(hf_pos_path)
                cpp_pos = cpp[pos]
                corr = np.corrcoef(cpp_pos.flatten(), hf_pos.flatten())[0, 1]
                diff = np.abs(cpp_pos - hf_pos).max()
                print(f"  Position {pos}: corr={corr:.6f}, max_diff={diff:.6f}")

    # Generated tokens
    print("\n--- Generated Tokens ---")
    hf_tokens_path = hf_dir / "06_generated_tokens.npy"
    if hf_tokens_path.exists():
        hf_tokens = np.load(hf_tokens_path)
        print(f"HF  tokens: {hf_tokens[:20].tolist()}")
        print(f"C++ tokens: [2157, 852, 1063, 1589, 908, 632] (from log)")
        print(f"\nFirst token comparison:")
        print(f"  HF  first: {hf_tokens[0]} (should be normal codec token)")
        print(f"  C++ first: 2157 (codec_think_eos_id - WRONG!)")


if __name__ == "__main__":
    main()
