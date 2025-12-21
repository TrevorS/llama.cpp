#!/usr/bin/env python3
"""
Test GGML depthwise conv vs PyTorch depthwise Conv1d.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import glob
import safetensors.torch


def load_ggml_tensor(path, shape):
    """Load GGML tensor with correct memory layout."""
    arr = np.fromfile(str(path), dtype=np.float32)
    ne0, ne1 = shape
    expected_size = ne0 * ne1
    if len(arr) != expected_size:
        return None
    return arr.reshape(ne1, ne0).T


def main():
    hf_dir = Path('/models/debug/hf_tensors')
    cpp_dir = Path('/models/debug/cpp_tensors_new')

    # Load HF transconv output (input to dwconv)
    hf_input = np.load(str(hf_dir / '05_up0_transconv.npy'))  # [1, 1024, 4]
    print(f'HF transconv (dwconv input) shape: {hf_input.shape}')

    # Load HF dwconv output
    hf_output = np.load(str(hf_dir / '06_cnxt0_dwconv.npy'))  # [1, 1024, 4]
    print(f'HF dwconv output shape: {hf_output.shape}')

    # Load C++ transconv output (input to dwconv)
    cpp_input = load_ggml_tensor(cpp_dir / 'cnxt0_transconv_raw.bin', (1024, 4))
    print(f'C++ transconv output shape: {cpp_input.shape if cpp_input is not None else "FAILED"}')

    # Load C++ dwconv output
    cpp_output = load_ggml_tensor(cpp_dir / 'cnxt0_dwconv.bin', (1024, 4))
    print(f'C++ dwconv output shape: {cpp_output.shape if cpp_output is not None else "FAILED"}')

    # Reshape for comparison
    hf_input_flat = hf_input.flatten()
    cpp_input_flat = cpp_input.flatten() if cpp_input is not None else None

    if cpp_input_flat is not None:
        # Compare inputs
        corr = np.corrcoef(hf_input_flat, cpp_input_flat)[0, 1]
        print(f'\nInput (transconv output) correlation: {corr:.4f}')

    hf_output_flat = hf_output.flatten()
    cpp_output_flat = cpp_output.flatten() if cpp_output is not None else None

    if cpp_output_flat is not None:
        # Compare outputs
        corr = np.corrcoef(hf_output_flat, cpp_output_flat)[0, 1]
        print(f'Output (dwconv output) correlation: {corr:.4f}')

    # Load HuggingFace dwconv weights
    print('\n=== Loading HF dwconv weights ===')
    model_path = '/models/Qwen3-Omni-30B-A3B-Instruct'
    hf_weight = None
    hf_bias = None

    for sf in sorted(glob.glob(f'{model_path}/*.safetensors')):
        weights = safetensors.torch.load_file(sf)
        if 'code2wav.upsample.0.1.dwconv.conv.weight' in weights:
            hf_weight = weights['code2wav.upsample.0.1.dwconv.conv.weight'].float()
        if 'code2wav.upsample.0.1.dwconv.conv.bias' in weights:
            hf_bias = weights['code2wav.upsample.0.1.dwconv.conv.bias'].float()

    if hf_weight is None:
        print('Could not find dwconv weight')
        return

    print(f'HF dwconv weight shape: {hf_weight.shape}')  # [channels, 1, kernel_size] = [1024, 1, 7]
    print(f'HF dwconv weight first 5: {hf_weight[0, 0, :5].numpy()}')

    # Load GGUF dwconv weights
    from gguf import GGUFReader
    reader = GGUFReader('/models/qwen3-omni-30b-talker-f16-v4.gguf')

    gguf_weight = None
    gguf_bias = None
    for tensor in reader.tensors:
        if 'up.0.dwconv.weight' in tensor.name:
            gguf_weight = tensor.data.flatten().astype(np.float32)
            print(f'\nGGUF dwconv weight size: {len(gguf_weight)}, shape: {tensor.shape}')
            print(f'GGUF dwconv weight first 5: {gguf_weight[:5]}')
        if 'up.0.dwconv.bias' in tensor.name:
            gguf_bias = tensor.data.flatten().astype(np.float32)

    # Compare weights
    if hf_weight is not None and gguf_weight is not None:
        hf_w_flat = hf_weight.numpy().flatten()
        if len(hf_w_flat) == len(gguf_weight):
            corr = np.corrcoef(hf_w_flat, gguf_weight)[0, 1]
            print(f'\nWeight correlation (raw): {corr:.4f}')

    # Test depthwise conv with PyTorch
    print('\n=== PyTorch Depthwise Conv Test ===')

    # Create depthwise conv (groups=channels)
    dwconv = nn.Conv1d(
        in_channels=1024,
        out_channels=1024,
        kernel_size=7,
        padding=3,  # 'same' padding
        groups=1024,  # depthwise
        bias=True
    )
    dwconv.weight.data = hf_weight
    if hf_bias is not None:
        dwconv.bias.data = hf_bias
    dwconv.eval()

    # Run with HF input
    with torch.no_grad():
        hf_input_torch = torch.from_numpy(hf_input)
        pytorch_output = dwconv(hf_input_torch).numpy()

    print(f'PyTorch output shape: {pytorch_output.shape}')
    print(f'PyTorch output: min={pytorch_output.min():.4f}, max={pytorch_output.max():.4f}, mean={pytorch_output.mean():.4f}')

    # Compare PyTorch output with HF output
    corr = np.corrcoef(hf_output.flatten(), pytorch_output.flatten())[0, 1]
    print(f'HF vs PyTorch correlation: {corr:.4f}')

    # Run with C++ input (converted to PyTorch format)
    if cpp_input is not None:
        cpp_input_torch = torch.from_numpy(cpp_input.reshape(1, 1024, 4).astype(np.float32))
        with torch.no_grad():
            pytorch_from_cpp = dwconv(cpp_input_torch).numpy()

        print(f'\nPyTorch with C++ input: min={pytorch_from_cpp.min():.4f}, max={pytorch_from_cpp.max():.4f}')

        if cpp_output is not None:
            cpp_output_for_compare = cpp_output.reshape(1, 1024, 4)
            corr = np.corrcoef(pytorch_from_cpp.flatten(), cpp_output_for_compare.flatten())[0, 1]
            print(f'PyTorch(C++ input) vs C++ output correlation: {corr:.4f}')

            # Print some values to debug
            print('\nFirst channel, all positions:')
            print(f'  PyTorch: {pytorch_from_cpp[0, 0, :].flatten()[:4]}')
            print(f'  C++:     {cpp_output_for_compare[0, 0, :].flatten()[:4]}')


if __name__ == '__main__':
    main()
