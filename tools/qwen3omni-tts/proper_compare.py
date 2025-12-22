#!/usr/bin/env python3
"""Properly compare transconv with correct GGML memory layout handling."""

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
import os

def load_ggml_tensor(path, ne):
    """Load tensor from GGML dump with correct memory layout.

    GGML uses column-major (Fortran) order: ne[0] varies fastest.
    For a tensor with ne = [a, b], element [i, j] is at offset i + j*a.

    After loading, we return in [ne[1], ne[0]] numpy shape for intuitive indexing.
    """
    data = np.fromfile(path, dtype='<f4')
    expected_size = np.prod(ne)
    assert len(data) == expected_size, f"Size mismatch: {len(data)} vs {expected_size}"

    # GGML ne = [fastest, slowest], so reshape as [slowest, fastest] for numpy
    shape_numpy = tuple(reversed(ne))
    return data.reshape(shape_numpy, order='F')

def main():
    # File paths
    cpp_input_path = "/models/debug/cpp_tokens_match/cnxt0_transconv_input.bin"
    cpp_output_path = "/models/debug/cpp_tokens_match/cnxt0_transconv_raw.bin"
    model_path = "/models/Qwen3-Omni-30B-A3B-Instruct"

    # Load C++ tensors with correct GGML layout interpretation
    # cnxt0_transconv_input: ne = [16, 1024] (seq, channels) after first transpose
    cpp_input = load_ggml_tensor(cpp_input_path, [16, 1024])
    print(f"C++ input shape after GGML load: {cpp_input.shape}")  # Should be [1024, 16] in numpy
    # But we want [seq, channels] for PyTorch, so transpose
    cpp_input = cpp_input.T  # [16, 1024] = [seq, channels]
    print(f"C++ input [seq, channels]: {cpp_input.shape}")
    print(f"C++ input[0, :5]: {cpp_input[0, :5]}")

    # cnxt0_transconv_raw: ne = [1024, 32] (channels, seq) after second transpose + bias
    cpp_output = load_ggml_tensor(cpp_output_path, [1024, 32])
    print(f"C++ output shape after GGML load: {cpp_output.shape}")  # Should be [32, 1024] in numpy
    cpp_output = cpp_output.T  # [1024, 32] = [channels, seq]
    print(f"C++ output [channels, seq]: {cpp_output.shape}")
    print(f"C++ output[:5, 0] (first 5 channels at pos 0): {cpp_output[:5, 0]}")

    # Load HuggingFace weights
    hf_weight = None
    hf_bias = None
    for f in os.listdir(model_path):
        if f.endswith('.safetensors'):
            path = os.path.join(model_path, f)
            with safe_open(path, framework='pt') as sf:
                for name in sf.keys():
                    if 'upsample.0.0.conv.weight' in name:
                        hf_weight = sf.get_tensor(name).float()
                    elif 'upsample.0.0.conv.bias' in name:
                        hf_bias = sf.get_tensor(name).float()
            if hf_weight is not None and hf_bias is not None:
                break

    print(f"\nHF weight shape: {list(hf_weight.shape)}")  # [Cin, Cout, K] = [1024, 1024, 2]
    print(f"HF bias shape: {list(hf_bias.shape)}")  # [Cout] = [1024]

    # Run HuggingFace transposed convolution
    # PyTorch expects input [batch, channels, seq]
    input_torch = torch.from_numpy(cpp_input.copy()).unsqueeze(0).permute(0, 2, 1)  # [1, 1024, 16]
    print(f"\nPyTorch input shape: {list(input_torch.shape)}")

    hf_output = F.conv_transpose1d(input_torch, hf_weight, hf_bias, stride=2)
    print(f"HF output shape: {list(hf_output.shape)}")  # [1, 1024, 32]

    # HF output is [batch, channels, seq], convert to [channels, seq]
    hf_output_np = hf_output[0].detach().numpy()  # [1024, 32] = [channels, seq]
    print(f"HF output [channels, seq]: {hf_output_np.shape}")
    print(f"HF output[:5, 0] (first 5 channels at pos 0): {hf_output_np[:5, 0]}")

    # Compare
    print(f"\n=== Comparison ===")
    print(f"C++ shape: {cpp_output.shape}")
    print(f"HF shape:  {hf_output_np.shape}")

    if cpp_output.shape != hf_output_np.shape:
        print("Shape mismatch!")
        return

    diff = np.abs(cpp_output - hf_output_np)
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    corr = np.corrcoef(cpp_output.flatten(), hf_output_np.flatten())[0, 1]
    print(f"Correlation: {corr:.6f}")

    # Check first few positions in detail
    print(f"\n=== Position 0 comparison ===")
    print(f"C++: {cpp_output[:10, 0]}")
    print(f"HF:  {hf_output_np[:10, 0]}")

    print(f"\n=== Position 1 comparison ===")
    print(f"C++: {cpp_output[:10, 1]}")
    print(f"HF:  {hf_output_np[:10, 1]}")

    # Check per-position correlations
    print(f"\n=== Per-position correlations ===")
    for pos in range(min(6, cpp_output.shape[1])):
        c = np.corrcoef(cpp_output[:, pos], hf_output_np[:, pos])[0, 1]
        print(f"Position {pos}: {c:.6f}")

    # Check if the output might be without bias
    hf_output_no_bias = F.conv_transpose1d(input_torch, hf_weight, None, stride=2)
    hf_output_no_bias_np = hf_output_no_bias[0].detach().numpy()
    corr_no_bias = np.corrcoef(cpp_output.flatten(), hf_output_no_bias_np.flatten())[0, 1]
    print(f"\n=== Without bias correlation: {corr_no_bias:.6f} ===")

    # Check per-position without bias
    print(f"\n=== Per-position correlations (no bias) ===")
    for pos in range(min(6, cpp_output.shape[1])):
        c = np.corrcoef(cpp_output[:, pos], hf_output_no_bias_np[:, pos])[0, 1]
        print(f"Position {pos}: {c:.6f}")

if __name__ == "__main__":
    main()
