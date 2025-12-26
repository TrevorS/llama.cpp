#!/usr/bin/env python3
"""Quick tensor comparison for TTS debugging."""

import struct
import numpy as np
from pathlib import Path

def load_tensor(path):
    with open(path, 'rb') as f:
        ndims = struct.unpack('<I', f.read(4))[0]
        shape = tuple(struct.unpack('<I', f.read(4))[0] for _ in range(ndims))
        data = np.frombuffer(f.read(), dtype='<f4').reshape(shape)
    return data, shape

def compare(name, hf_path, llama_path):
    hf, hf_shape = load_tensor(hf_path)
    llama, llama_shape = load_tensor(llama_path)

    print(f'{name}:')
    print(f'  HF shape: {hf_shape}')
    print(f'  llama shape: {llama_shape}')

    if hf_shape != llama_shape:
        print(f'  ** SHAPE MISMATCH! **')
        return False

    diff = np.abs(hf - llama)
    print(f'  max_diff: {diff.max():.6f}')
    print(f'  mean_diff: {diff.mean():.6f}')
    print(f'  HF first 5: {hf.flat[:5]}')
    print(f'  llama first 5: {llama.flat[:5]}')

    if diff.max() > 0.01:
        print(f'  ** VALUES DIVERGE! **')
        return False
    return True

def main():
    hf_dir = Path('/models/debug/hf_tensors')
    llama_dir = Path('/models/debug/llama_tensors')

    print("=== Comparing TTS Tensors ===\n")

    # Compare tensors with matching shapes
    results = []
    results.append(compare('tts_bos_embed',
                           hf_dir / '03b_tts_bos_embed.bin',
                           llama_dir / 'tts_bos_embed.bin'))
    print()

    results.append(compare('tts_eos_embed',
                           hf_dir / '03b_tts_eos_embed.bin',
                           llama_dir / 'tts_eos_embed.bin'))
    print()

    results.append(compare('tts_pad_embed',
                           hf_dir / '03b_tts_pad_embed.bin',
                           llama_dir / 'tts_pad_embed.bin'))
    print()

    results.append(compare('prefill_embeds',
                           hf_dir / '03g_prefill_embeds.bin',
                           llama_dir / 'prefill_embeds.bin'))
    print()

    # Show shape info for mismatched tensors
    hf, _ = load_tensor(hf_dir / '03e_assistant_hidden.bin')
    llama, _ = load_tensor(llama_dir / 'assistant_hidden.bin')
    print(f'assistant_hidden:')
    print(f'  HF shape: {hf.shape} - {hf.shape[0]} tokens')
    print(f'  llama shape: {llama.shape} - {llama.shape[0]} tokens')
    print(f'  ** SHAPE MISMATCH! ** (HF has {hf.shape[0]} tokens, llama has {llama.shape[0]})')
    print()

    hf, _ = load_tensor(hf_dir / '03f_trailing_text_hidden.bin')
    llama, _ = load_tensor(llama_dir / 'trailing_text_hidden.bin')
    print(f'trailing_text_hidden:')
    print(f'  HF shape: {hf.shape} - {hf.shape[0]} tokens')
    print(f'  llama shape: {llama.shape} - {llama.shape[0]} tokens')
    print(f'  ** SHAPE MISMATCH! ** (HF has {hf.shape[0]} tokens, llama has {llama.shape[0]})')
    print()

    print("=== Summary ===")
    passed = sum(results)
    print(f'Matched tensors: {passed}/4')
    if passed < 4:
        print("Some tensors DIVERGE - fix these first!")

if __name__ == "__main__":
    main()
