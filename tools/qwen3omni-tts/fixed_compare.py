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

    Using Fortran-order reshape gives correct mapping.
    """
    data = np.fromfile(path, dtype='<f4')
    expected_size = int(np.prod(ne))
    assert len(data) == expected_size, f"Size mismatch: {len(data)} vs {expected_size}"

    # GGML is column-major, so use Fortran order reshape
    # This gives array[i, j] = data[i + j*ne[0]] = GGML[i, j]
    return data.reshape(ne, order='F')

def main():
    # File paths
    cpp_input_path = "/models/debug/cpp_tokens_match/cnxt0_transconv_input.bin"
    cpp_output_path = "/models/debug/cpp_tokens_match/cnxt0_transconv_raw.bin"
    model_path = "/models/Qwen3-Omni-30B-A3B-Instruct"

    # cnxt0_transconv_input: ne = [16, 1024] (seq, channels) after first transpose
    cpp_input = load_ggml_tensor(cpp_input_path, (16, 1024))
    print(f"C++ input shape: {cpp_input.shape}")  # (16, 1024) = [seq, channels]
    print(f"C++ input[0, :5]: {cpp_input[0, :5]}")  # First seq position, first 5 channels
    print(f"C++ input[:5, 0]: {cpp_input[:5, 0]}")  # First 5 seq positions, first channel

    # cnxt0_transconv_raw: ne = [1024, 32] (channels, seq) after second transpose + bias
    cpp_output = load_ggml_tensor(cpp_output_path, (1024, 32))
    print(f"\nC++ output shape: {cpp_output.shape}")  # (1024, 32) = [channels, seq]
    print(f"C++ output[:5, 0] (first 5 channels at pos 0): {cpp_output[:5, 0]}")
    print(f"C++ output[0, :5] (channel 0 at first 5 positions): {cpp_output[0, :5]}")

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

    # Run HuggingFace transposed convolution
    # PyTorch conv_transpose1d expects input [batch, channels, seq]
    # cpp_input is [seq, channels], so permute to [channels, seq] then add batch
    input_torch = torch.from_numpy(cpp_input.T.copy()).unsqueeze(0)  # [1, channels, seq] = [1, 1024, 16]
    print(f"\nPyTorch input shape: {list(input_torch.shape)}")

    hf_output = F.conv_transpose1d(input_torch, hf_weight, hf_bias, stride=2)
    print(f"HF output shape: {list(hf_output.shape)}")  # [1, 1024, 32]

    # HF output is [batch, channels, seq], extract [channels, seq]
    hf_output_np = hf_output[0].detach().numpy()  # [1024, 32]
    print(f"HF output [channels, seq]: {hf_output_np.shape}")
    print(f"HF output[:5, 0]: {hf_output_np[:5, 0]}")
    print(f"HF output[0, :5]: {hf_output_np[0, :5]}")

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

    # Detailed position comparison
    print(f"\n=== Position 0 (all should match) ===")
    print(f"C++ [:10, 0]: {cpp_output[:10, 0]}")
    print(f"HF  [:10, 0]: {hf_output_np[:10, 0]}")

    print(f"\n=== Position 1 ===")
    print(f"C++ [:10, 1]: {cpp_output[:10, 1]}")
    print(f"HF  [:10, 1]: {hf_output_np[:10, 1]}")

    # Per-position correlations
    print(f"\n=== Per-position correlations ===")
    for pos in range(min(8, cpp_output.shape[1])):
        c = np.corrcoef(cpp_output[:, pos], hf_output_np[:, pos])[0, 1]
        print(f"Position {pos}: {c:.6f}")

    # Check if maybe input interpretation is wrong
    # What if C++ input is actually [channels, seq] not [seq, channels]?
    print(f"\n=== Alternative: C++ input as [channels, seq] ===")
    cpp_input_alt = load_ggml_tensor(cpp_input_path, (1024, 16))  # Try reversed
    input_torch_alt = torch.from_numpy(cpp_input_alt.copy()).unsqueeze(0)  # [1, 1024, 16]
    hf_output_alt = F.conv_transpose1d(input_torch_alt, hf_weight, hf_bias, stride=2)
    hf_output_alt_np = hf_output_alt[0].detach().numpy()
    corr_alt = np.corrcoef(cpp_output.flatten(), hf_output_alt_np.flatten())[0, 1]
    print(f"Alternative correlation: {corr_alt:.6f}")

if __name__ == "__main__":
    main()
