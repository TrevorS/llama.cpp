#!/usr/bin/env python3
"""Compare HF and C++ prefill logits."""

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


def main():
    cpp_dir = Path("/models/debug/cpp_talker")
    hf_dir = Path("/models/debug/hf_talker")

    print("=" * 60)
    print("Prefill Logits Comparison: C++ vs HuggingFace")
    print("=" * 60)

    # Load logits
    cpp_logits = load_bin(cpp_dir / "prefill_logits.bin")
    hf_logits = np.load(hf_dir / "prefill_logits.npy")

    print(f"\nC++ logits shape: {cpp_logits.shape}")
    print(f"HF  logits shape: {hf_logits.shape}")

    # Overall correlation
    corr = np.corrcoef(cpp_logits.flatten(), hf_logits.flatten())[0, 1]
    print(f"\nOverall correlation: {corr:.6f}")

    # Statistics
    print(f"\nC++ logits: mean={cpp_logits.mean():.4f}, std={cpp_logits.std():.4f}, min={cpp_logits.min():.4f}, max={cpp_logits.max():.4f}")
    print(f"HF  logits: mean={hf_logits.mean():.4f}, std={hf_logits.std():.4f}, min={hf_logits.min():.4f}, max={hf_logits.max():.4f}")

    # Top-10 from each
    print("\n--- Top-10 C++ tokens ---")
    cpp_top_idx = np.argsort(cpp_logits)[::-1][:10]
    for i, idx in enumerate(cpp_top_idx):
        print(f"  [{i}] token={idx}, cpp_logit={cpp_logits[idx]:.4f}, hf_logit={hf_logits[idx]:.4f}")

    print("\n--- Top-10 HF tokens ---")
    hf_top_idx = np.argsort(hf_logits)[::-1][:10]
    for i, idx in enumerate(hf_top_idx):
        print(f"  [{i}] token={idx}, hf_logit={hf_logits[idx]:.4f}, cpp_logit={cpp_logits[idx]:.4f}")

    # Check specific tokens
    print("\n--- Key tokens ---")
    key_tokens = [
        (1049, "HF top"),
        (1359, "C++ top"),
        (2148, "codec_pad"),
        (2149, "codec_bos"),
        (2150, "codec_eos"),
        (2302, "speaker_Ethan"),
    ]
    for tok_id, name in key_tokens:
        print(f"  {name} ({tok_id}): cpp={cpp_logits[tok_id]:.4f}, hf={hf_logits[tok_id]:.4f}, diff={cpp_logits[tok_id]-hf_logits[tok_id]:.4f}")

    # Scatter plot of audio tokens only (0-2047)
    audio_cpp = cpp_logits[:2048]
    audio_hf = hf_logits[:2048]
    audio_corr = np.corrcoef(audio_cpp, audio_hf)[0, 1]
    print(f"\nAudio tokens (0-2047) correlation: {audio_corr:.6f}")

    # Find biggest differences
    diff = np.abs(cpp_logits - hf_logits)
    print(f"\nMax difference: {diff.max():.4f} at token {np.argmax(diff)}")
    print(f"Mean difference: {diff.mean():.4f}")

    # Histogram of differences
    print("\n--- Difference histogram ---")
    bins = [0, 0.5, 1, 2, 5, 10, 100]
    for i in range(len(bins) - 1):
        count = np.sum((diff >= bins[i]) & (diff < bins[i+1]))
        print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count} tokens")


if __name__ == "__main__":
    main()
