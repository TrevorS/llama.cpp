#!/usr/bin/env python3
"""Compare transposed convolution: C++ vs HuggingFace."""

import sys
sys.path.insert(0, '/app/src/gguf-py')

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
import gguf
import os

def main():
    # Load C++ transconv input
    cpp_input_path = "/models/debug/cpp_tokens_match/cnxt0_transconv_input.bin"
    n_floats = os.path.getsize(cpp_input_path) // 4
    seq_len = n_floats // 1024
    cpp_input = np.fromfile(cpp_input_path, dtype='<f4').reshape(seq_len, 1024)
    print(f"C++ input shape: {cpp_input.shape}")
    print(f"C++ input[0,:5]: {cpp_input[0, :5]}")

    # Load C++ transconv output
    cpp_output_path = "/models/debug/cpp_tokens_match/cnxt0_transconv_raw.bin"
    n_floats_out = os.path.getsize(cpp_output_path) // 4
    out_seq_len = n_floats_out // 1024
    cpp_output = np.fromfile(cpp_output_path, dtype='<f4').reshape(out_seq_len, 1024)
    print(f"C++ output shape: {cpp_output.shape}")
    print(f"C++ output[0,:5]: {cpp_output[0, :5]}")

    # Load HuggingFace weights
    model_path = "/models/Qwen3-Omni-30B-A3B-Instruct"
    hf_weight = None
    hf_bias = None
    for f in os.listdir(model_path):
        if f.endswith('.safetensors'):
            path = os.path.join(model_path, f)
            with safe_open(path, framework='pt') as sf:
                for name in sf.keys():
                    if 'upsample.0.0.conv.weight' in name:
                        hf_weight = sf.get_tensor(name).float()
                        print(f"\nHF weight shape: {list(hf_weight.shape)}")
                    elif 'upsample.0.0.conv.bias' in name:
                        hf_bias = sf.get_tensor(name).float()
                        print(f"HF bias shape: {list(hf_bias.shape)}")
            if hf_weight is not None and hf_bias is not None:
                break

    # Also load GGUF weight to compare
    reader = gguf.GGUFReader("/models/qwen3-omni-30b-talker-f16-v4.gguf")
    for tensor in reader.tensors:
        if tensor.name == "code2wav.up.0.conv.weight":
            gguf_weight = tensor.data.astype(np.float32).reshape(tensor.shape[::-1])
            print(f"GGUF weight shape: {gguf_weight.shape}")
            break

    # Verify weights match
    print("\n=== Weight Comparison ===")
    hf_np = hf_weight.numpy()
    diff = np.abs(hf_np - gguf_weight)
    print(f"Max weight diff: {diff.max():.10f}")
    print(f"Weights match: {np.allclose(hf_np, gguf_weight, rtol=1e-3)}")

    # Run HuggingFace transposed convolution
    print("\n=== Running HuggingFace TransConv ===")
    # PyTorch expects [batch, channels, seq]
    input_torch = torch.from_numpy(cpp_input).unsqueeze(0).permute(0, 2, 1)  # [1, 1024, seq]
    print(f"PyTorch input shape: {list(input_torch.shape)}")

    # Apply conv_transpose1d
    hf_output = F.conv_transpose1d(input_torch, hf_weight, hf_bias, stride=2)
    print(f"HF output shape: {list(hf_output.shape)}")

    # Convert back to [seq, channels]
    hf_output_np = hf_output[0].permute(1, 0).detach().numpy()
    print(f"HF output (seq, channels): {hf_output_np.shape}")

    # Compare outputs
    print("\n=== Output Comparison ===")
    print(f"C++ output shape: {cpp_output.shape}")
    print(f"HF output shape: {hf_output_np.shape}")

    if cpp_output.shape == hf_output_np.shape:
        diff = np.abs(cpp_output - hf_output_np)
        corr = np.corrcoef(cpp_output.flatten(), hf_output_np.flatten())[0, 1]
        print(f"Max diff: {diff.max():.6f}")
        print(f"Mean diff: {diff.mean():.6f}")
        print(f"Correlation: {corr:.6f}")

        print(f"\nFirst row comparison:")
        print(f"  C++: {cpp_output[0, :5]}")
        print(f"  HF:  {hf_output_np[0, :5]}")

        print(f"\nSecond row comparison:")
        print(f"  C++: {cpp_output[1, :5]}")
        print(f"  HF:  {hf_output_np[1, :5]}")

        # Check if it's a permutation issue
        print("\n=== Checking for permutation issues ===")
        # Maybe C++ output is transposed?
        cpp_output_T = cpp_output.T
        if cpp_output_T.shape == hf_output_np.shape:
            corr_T = np.corrcoef(cpp_output_T.flatten(), hf_output_np.flatten())[0, 1]
            print(f"Correlation with C++ transposed: {corr_T:.6f}")

        # Maybe C++ channels are in different order?
        corr_per_seq = []
        for i in range(min(5, cpp_output.shape[0])):
            c = np.corrcoef(cpp_output[i, :], hf_output_np[i, :])[0, 1]
            corr_per_seq.append(c)
        print(f"Per-sequence position correlations: {corr_per_seq}")

        corr_per_ch = []
        for i in range(min(5, cpp_output.shape[1])):
            c = np.corrcoef(cpp_output[:, i], hf_output_np[:, i])[0, 1]
            corr_per_ch.append(c)
        print(f"Per-channel correlations: {corr_per_ch}")

        # Check if even/odd interleaving is wrong
        print("\n=== Checking stride interleaving ===")
        # For stride=2 transconv, output[0], output[2], output[4]... come from kernel[0]
        # and output[1], output[3], output[5]... come from kernel[1]
        cpp_even = cpp_output[0::2, :]  # positions 0, 2, 4, ...
        cpp_odd = cpp_output[1::2, :]   # positions 1, 3, 5, ...
        hf_even = hf_output_np[0::2, :]
        hf_odd = hf_output_np[1::2, :]

        corr_even = np.corrcoef(cpp_even.flatten(), hf_even.flatten())[0, 1]
        corr_odd = np.corrcoef(cpp_odd.flatten(), hf_odd.flatten())[0, 1]
        print(f"Even positions (k=0) correlation: {corr_even:.6f}")
        print(f"Odd positions (k=1) correlation: {corr_odd:.6f}")

        # Maybe even/odd are swapped?
        corr_swap1 = np.corrcoef(cpp_even.flatten(), hf_odd.flatten())[0, 1]
        corr_swap2 = np.corrcoef(cpp_odd.flatten(), hf_even.flatten())[0, 1]
        print(f"C++ even vs HF odd: {corr_swap1:.6f}")
        print(f"C++ odd vs HF even: {corr_swap2:.6f}")

    else:
        print(f"Shape mismatch!")

if __name__ == "__main__":
    main()
