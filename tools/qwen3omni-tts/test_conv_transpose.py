#!/usr/bin/env python3
"""
Test GGML conv_transpose_1d vs PyTorch ConvTranspose1d.

This script loads the exact weights and inputs from both implementations
and compares the expected vs actual outputs.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


def main():
    hf_dir = Path('/models/debug/hf_tensors')
    cpp_dir = Path('/models/debug/cpp_tensors_new')

    # Load HF transconv input (04_permuted) - [batch, channels, seq]
    hf_input = np.load(str(hf_dir / '04_permuted.npy'))
    print(f'HF input shape: {hf_input.shape}')  # [1, 1024, 2]

    # Load HF transconv output (05_up0_transconv)
    hf_output = np.load(str(hf_dir / '05_up0_transconv.npy'))
    print(f'HF output shape: {hf_output.shape}')  # [1, 1024, 4]

    # Load C++ transconv input
    cpp_input = np.fromfile(str(cpp_dir / 'cnxt0_transconv_input.bin'), dtype=np.float32)
    print(f'C++ input size: {len(cpp_input)}')  # 2048

    # Load C++ transconv output
    cpp_output = np.fromfile(str(cpp_dir / 'cnxt0_transconv_raw.bin'), dtype=np.float32)
    print(f'C++ output size: {len(cpp_output)}')  # 4096

    # Reshape C++ tensors to match HF shapes
    # GGML uses column-major (Fortran) memory layout!
    #
    # C++ cnxt0_transconv_input is [2, 1024] in GGML notation (ne[0]=2, ne[1]=1024)
    # Memory layout: ne[0] is contiguous, so it's column-major
    # To read correctly: reshape with order='F' or reshape(ne[1], ne[0]).T
    cpp_input_reshaped = cpp_input.reshape(1024, 2).T  # [2, 1024] = [seq, channels]
    cpp_input_for_torch = cpp_input_reshaped.T.reshape(1, 1024, 2)  # [batch, channels, seq]

    # C++ cnxt0_transconv_raw is [1024, 4] in GGML notation (ne[0]=1024, ne[1]=4)
    # Memory layout: ne[0]=1024 is contiguous
    cpp_output_reshaped = cpp_output.reshape(4, 1024).T  # [1024, 4] = [channels, seq]
    cpp_output_for_compare = cpp_output_reshaped.reshape(1, 1024, 4)  # [batch, channels, seq]

    print('\n=== Input Comparison ===')
    print(f'HF input: min={hf_input.min():.4f}, max={hf_input.max():.4f}, mean={hf_input.mean():.4f}')
    print(f'CPP input (reshaped): min={cpp_input_for_torch.min():.4f}, max={cpp_input_for_torch.max():.4f}, mean={cpp_input_for_torch.mean():.4f}')
    corr = np.corrcoef(hf_input.flatten(), cpp_input_for_torch.flatten())[0, 1]
    print(f'Input correlation: {corr:.4f}')

    print('\n=== Output Comparison ===')
    print(f'HF output: min={hf_output.min():.4f}, max={hf_output.max():.4f}, mean={hf_output.mean():.4f}')
    print(f'CPP output: min={cpp_output.min():.4f}, max={cpp_output.max():.4f}, mean={cpp_output.mean():.4f}')
    corr = np.corrcoef(hf_output.flatten(), cpp_output_for_compare.flatten())[0, 1]
    print(f'Output correlation: {corr:.4f}')

    # Now let's reproduce the conv_transpose_1d with PyTorch using the same input
    print('\n=== PyTorch Conv Transpose Test ===')

    # Load the transconv weight from GGUF model
    import safetensors.torch
    import glob

    hf_weight = None
    hf_bias = None
    model_path = '/models/Qwen3-Omni-30B-A3B-Instruct'
    for sf in sorted(glob.glob(f'{model_path}/*.safetensors')):
        weights = safetensors.torch.load_file(sf)
        if 'code2wav.upsample.0.0.conv.weight' in weights:
            hf_weight = weights['code2wav.upsample.0.0.conv.weight'].float()
        if 'code2wav.upsample.0.0.conv.bias' in weights:
            hf_bias = weights['code2wav.upsample.0.0.conv.bias'].float()

    if hf_weight is None:
        print('Could not find transconv weight')
        return

    print(f'HF transconv weight shape: {hf_weight.shape}')  # [in_ch, out_ch, kernel]
    print(f'HF transconv bias shape: {hf_bias.shape}' if hf_bias is not None else 'No bias')

    # Create ConvTranspose1d layer with same settings
    conv = nn.ConvTranspose1d(
        in_channels=1024,
        out_channels=1024,
        kernel_size=2,
        stride=2,
        padding=0,
        bias=True
    )
    conv.weight.data = hf_weight
    if hf_bias is not None:
        conv.bias.data = hf_bias
    conv.eval()

    # Run with HF input
    with torch.no_grad():
        hf_input_torch = torch.from_numpy(hf_input)
        pytorch_output = conv(hf_input_torch).numpy()

    print(f'PyTorch output shape: {pytorch_output.shape}')
    print(f'PyTorch output: min={pytorch_output.min():.4f}, max={pytorch_output.max():.4f}, mean={pytorch_output.mean():.4f}')

    # Compare PyTorch output with HF output (should be ~1.0)
    corr = np.corrcoef(hf_output.flatten(), pytorch_output.flatten())[0, 1]
    print(f'HF vs PyTorch correlation: {corr:.4f}')

    # Now test with C++ input
    with torch.no_grad():
        cpp_input_torch = torch.from_numpy(cpp_input_for_torch.astype(np.float32))
        pytorch_from_cpp = conv(cpp_input_torch).numpy()

    print(f'\nPyTorch with C++ input: min={pytorch_from_cpp.min():.4f}, max={pytorch_from_cpp.max():.4f}, mean={pytorch_from_cpp.mean():.4f}')
    corr = np.corrcoef(pytorch_from_cpp.flatten(), cpp_output_for_compare.flatten())[0, 1]
    print(f'PyTorch(C++ input) vs C++ output correlation: {corr:.4f}')

    # This is the key test: does GGML produce same output as PyTorch given same input?
    print('\n=== Key Insight ===')
    print('If PyTorch(C++ input) != C++ output, then GGML conv_transpose_1d has a bug.')
    print('If PyTorch(C++ input) == C++ output but != HF output, then the input mismatch is the issue.')


if __name__ == '__main__':
    main()
