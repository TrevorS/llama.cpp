#!/usr/bin/env python3
"""Detailed tensor comparison for TTS debugging."""

import struct
import numpy as np
from pathlib import Path

def load_tensor(path):
    with open(path, 'rb') as f:
        ndims = struct.unpack('<I', f.read(4))[0]
        shape = tuple(struct.unpack('<I', f.read(4))[0] for _ in range(ndims))
        data = np.frombuffer(f.read(), dtype='<f4').reshape(shape)
    return data, shape

def compare(name, hf_path, llama_path, threshold=0.01):
    if not hf_path.exists():
        print(f'{name}: HF file not found')
        return None
    if not llama_path.exists():
        print(f'{name}: llama file not found')
        return None

    hf, hf_shape = load_tensor(hf_path)
    llama, llama_shape = load_tensor(llama_path)

    print(f'{name}:')
    print(f'  HF shape: {hf_shape}')
    print(f'  llama shape: {llama_shape}')

    if hf_shape != llama_shape:
        print(f'  ** SHAPE MISMATCH! **')
        # If shapes are compatible for element-wise comparison of matching positions
        min_shape = tuple(min(h, l) for h, l in zip(hf_shape, llama_shape))
        if len(min_shape) == 2:
            hf_slice = hf[:min_shape[0], :min_shape[1]]
            llama_slice = llama[:min_shape[0], :min_shape[1]]
            diff = np.abs(hf_slice - llama_slice)
            print(f'  Comparing first {min_shape} elements:')
            print(f'    max_diff: {diff.max():.6f}')
            print(f'    mean_diff: {diff.mean():.6f}')
        return False

    diff = np.abs(hf - llama)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f'  max_diff: {max_diff:.6f}')
    print(f'  mean_diff: {mean_diff:.6f}')

    # Show where max diff occurs
    max_idx = np.unravel_index(diff.argmax(), diff.shape)
    print(f'  max_diff at index {max_idx}: HF={hf[max_idx]:.6f}, llama={llama[max_idx]:.6f}')

    # Show per-row stats for 2D tensors
    if len(hf_shape) == 2:
        row_diffs = diff.max(axis=1)
        print(f'  Per-row max diffs (first 15):')
        for i, rd in enumerate(row_diffs[:15]):
            marker = ' **' if rd > threshold else ''
            print(f'    row {i}: {rd:.6f}{marker}')

    if max_diff > threshold:
        print(f'  ** VALUES DIVERGE! (>{threshold}) **')
        return False
    return True

def main():
    hf_dir = Path('/models/debug/hf_tensors')
    llama_dir = Path('/models/debug/llama_tensors')

    print("=== Detailed TTS Tensor Comparison ===\n")

    # Compare matching-shape tensors
    tensors = [
        ('tts_bos_embed', '03b_tts_bos_embed.bin', 'tts_bos_embed.bin'),
        ('tts_eos_embed', '03b_tts_eos_embed.bin', 'tts_eos_embed.bin'),
        ('tts_pad_embed', '03b_tts_pad_embed.bin', 'tts_pad_embed.bin'),
        ('text_prefill', '03g_text_prefill.bin', 'text_prefill.bin'),
        ('codec_prefill', '03g_codec_prefill.bin', 'codec_prefill.bin'),
        ('prefill_embeds', '03g_prefill_embeds.bin', 'prefill_embeds.bin'),
        ('assistant_hidden', '03e_assistant_hidden.bin', 'assistant_hidden.bin'),
        ('trailing_text_hidden', '03f_trailing_text_hidden.bin', 'trailing_text_hidden.bin'),
    ]

    results = []
    for name, hf_file, llama_file in tensors:
        result = compare(name, hf_dir / hf_file, llama_dir / llama_file)
        results.append((name, result))
        print()

    print("=== Summary ===")
    for name, result in results:
        if result is None:
            status = "MISSING"
        elif result:
            status = "OK"
        else:
            status = "DIVERGE"
        print(f'  {name}: {status}')

if __name__ == "__main__":
    main()
