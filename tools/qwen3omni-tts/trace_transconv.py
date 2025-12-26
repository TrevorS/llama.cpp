#!/usr/bin/env python3
"""Trace transposed convolution step by step."""

import numpy as np
import torch
import torch.nn.functional as F

def main():
    # Minimal example: Cin=2, Cout=3, K=2, seq_len=3, stride=2
    Cin, Cout, K = 2, 3, 2
    seq_len = 3
    stride = 2

    # Create input [seq, Cin] = [3, 2]
    np.random.seed(42)
    input_np = np.random.randn(seq_len, Cin).astype(np.float32)
    print(f"Input [seq, Cin] = {input_np.shape}")
    print(f"Input:\n{input_np}")

    # Create kernel - PyTorch shape [Cin, Cout, K] = [2, 3, 2]
    kernel_np = np.random.randn(Cin, Cout, K).astype(np.float32)
    print(f"\nKernel PyTorch [Cin, Cout, K] = {kernel_np.shape}")
    print(f"Kernel:\n{kernel_np}")

    # PyTorch conv_transpose1d
    input_torch = torch.from_numpy(input_np).unsqueeze(0).permute(0, 2, 1)  # [1, Cin, seq]
    kernel_torch = torch.from_numpy(kernel_np)
    output_torch = F.conv_transpose1d(input_torch, kernel_torch, stride=stride)
    output_torch_np = output_torch[0].permute(1, 0).detach().numpy()  # [seq_out, Cout]
    print(f"\nPyTorch output [seq_out, Cout] = {output_torch_np.shape}")
    print(f"PyTorch output:\n{output_torch_np}")

    # Manual computation matching PyTorch
    # output[i*stride + k, cout] = sum over cin of (input[i, cin] * kernel[cin, cout, k])
    seq_out = seq_len * stride
    manual_output = np.zeros((seq_out, Cout), dtype=np.float32)
    for i in range(seq_len):
        for k in range(K):
            o = i * stride + k
            for cout in range(Cout):
                for cin in range(Cin):
                    manual_output[o, cout] += input_np[i, cin] * kernel_np[cin, cout, k]

    print(f"\nManual (PyTorch formula) output:\n{manual_output}")
    print(f"Max diff from PyTorch: {np.abs(manual_output - output_torch_np).max():.10f}")

    # Now simulate GGML's computation
    # GGML stores kernel as [K, Cout, Cin] in ne[] order
    # Memory layout for GGML with ne=[K, Cout, Cin]:
    # kernel_ggml[k, cout, cin] is at offset: k + cout*K + cin*K*Cout
    # But since we're starting from PyTorch's kernel[cin, cout, k], let's see what GGML sees

    # GGUF stores PyTorch's [Cin, Cout, K] with reversed shape [K, Cout, Cin]
    # The memory stays the same - just the shape metadata changes
    # So kernel_ggml[k, cout, cin] = kernel_torch[cin, cout, k] in terms of values

    # GGML permutes from [K, Cout, Cin] to [Cout, K, Cin]
    # wdata[cout, k, cin] = kernel[k, cout, cin] = kernel_torch[cin, cout, k]
    wdata = np.zeros((Cout, K, Cin), dtype=np.float32)
    for cin in range(Cin):
        for cout in range(Cout):
            for k in range(K):
                wdata[cout, k, cin] = kernel_np[cin, cout, k]

    print(f"\nGGML permuted kernel [Cout, K, Cin]:\n{wdata}")

    # Also permute input from [seq, Cin] with ne=[seq, Cin] to wdata_src[seq, Cin]
    # GGML's input permutation:
    # for i11 in range(ne11):  # Cin
    #     for i10 in range(ne10):  # seq
    #         wdata_src[i10*ne11 + i11] = src1[i10, i11]
    # This is just flattening in row-major order
    wdata_src = input_np.flatten()  # [seq * Cin]
    print(f"\nGGML input flattened: {wdata_src}")

    # GGML's main computation:
    # for i1 in range(Cout):  # output channel
    #     for i10 in range(seq_len):  # input position
    #         for i00 in range(K):  # kernel position
    #             v = dot(wdata_src[i10*Cin : i10*Cin+Cin], wdata_kernel[i00, :])
    #             dst[i10*stride + i00, i1] += v
    #
    # where wdata_kernel = wdata[i1, :, :] = wdata[i1]

    ggml_output = np.zeros((seq_out, Cout), dtype=np.float32)
    for cout in range(Cout):
        wdata_kernel = wdata[cout]  # [K, Cin]
        for i10 in range(seq_len):
            input_slice = wdata_src[i10*Cin : (i10+1)*Cin]  # [Cin]
            for k in range(K):
                v = np.dot(input_slice, wdata_kernel[k])  # dot([Cin], [Cin])
                ggml_output[i10*stride + k, cout] += v

    print(f"\nGGML simulation output:\n{ggml_output}")
    print(f"Max diff from PyTorch: {np.abs(ggml_output - output_torch_np).max():.10f}")

    # Check if they match
    if np.allclose(ggml_output, output_torch_np, atol=1e-6):
        print("\n✓ GGML matches PyTorch!")
    else:
        print("\n✗ GGML differs from PyTorch!")
        print("Difference matrix:")
        print(ggml_output - output_torch_np)

if __name__ == "__main__":
    main()
