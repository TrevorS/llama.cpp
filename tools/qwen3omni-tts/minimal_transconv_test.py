#!/usr/bin/env python3
"""Minimal transposed convolution test to understand GGML vs PyTorch."""

import torch
import torch.nn.functional as F
import numpy as np

def main():
    # Small test case: in_channels=2, out_channels=3, kernel_size=2, stride=2
    # Input: seq_len=4
    Cin, Cout, K = 2, 3, 2
    seq_len = 4

    # Create simple weights: identity-ish for first output channel
    weight = torch.zeros(Cin, Cout, K)
    # Set specific values to trace
    weight[0, 0, 0] = 1.0  # in[0] * k[0] -> out[0]
    weight[0, 0, 1] = 0.5  # in[0] * k[1] -> out[0]
    weight[1, 1, 0] = 2.0  # in[1] * k[0] -> out[1]

    print(f"Weight shape: {weight.shape}")
    print(f"Weight tensor:\n{weight}")

    # Input: [seq_len, channels] = [4, 2]
    input_2d = torch.tensor([
        [1.0, 0.0],  # seq 0: only channel 0 active
        [0.0, 1.0],  # seq 1: only channel 1 active
        [1.0, 1.0],  # seq 2: both channels
        [2.0, 0.5],  # seq 3: mixed
    ])
    print(f"\nInput (seq, channels):\n{input_2d}")

    # PyTorch conv_transpose1d expects [batch, channels, seq]
    input_pytorch = input_2d.unsqueeze(0).permute(0, 2, 1)  # [1, 2, 4]
    print(f"\nPyTorch input shape: {input_pytorch.shape}")

    # PyTorch conv_transpose1d with stride=2
    output_pytorch = F.conv_transpose1d(input_pytorch, weight, stride=2)
    print(f"PyTorch output shape: {output_pytorch.shape}")

    # Convert back to [seq, channels] for comparison
    output_2d = output_pytorch[0].permute(1, 0)  # [seq, channels]
    print(f"\nPyTorch output (seq, channels):\n{output_2d}")

    # Explain what should happen:
    print("\n=== Expected behavior ===")
    print("With stride=2, each input position produces 2 output positions")
    print("Input[0] = [1, 0] should produce:")
    print("  Output[0, 0] = weight[0,0,0]*1 = 1.0")
    print("  Output[1, 0] = weight[0,0,1]*1 = 0.5")
    print("  (other channels = 0 since input[0,1]=0)")

    print("\nActual Output[0:2, 0]:", output_2d[0:2, 0].tolist())

    print("\nInput[1] = [0, 1] should produce:")
    print("  Output[2, 1] = weight[1,1,0]*1 = 2.0")
    print("  Output[3, 1] = 0 (weight[1,1,1]=0)")
    print("Actual Output[2:4, 1]:", output_2d[2:4, 1].tolist())

    # Now let's see what GGML would expect
    print("\n=== GGML expectations ===")
    print("GGML conv_transpose_1d takes kernel with ne=[K, Cout, Cin]")
    print("After GGUF writer reversal:")
    print(f"  PyTorch [Cin, Cout, K] = {weight.shape}")
    print(f"  GGUF stores -> GGML reads ne = [{K}, {Cout}, {Cin}]")

    # The internal permutation in GGML ops.cpp:
    # "prepare kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)"
    # This means GGML expects the kernel data in memory order (Cin, K, Cout)
    # after reading ne=[K, Cout, Cin]

    # PyTorch stores weight in memory as:
    # for cin in range(Cin):
    #   for cout in range(Cout):
    #     for k in range(K):
    #       data[cin*Cout*K + cout*K + k]

    # GGML with ne=[K, Cout, Cin] expects:
    # for cin in range(Cin):
    #   for cout in range(Cout):
    #     for k in range(K):
    #       data[k + cout*K + cin*K*Cout]

    # These are the SAME! Both have k varying fastest.

    # Let me verify the actual weight values in memory order
    print("\nPyTorch weight in memory order:")
    flat = weight.flatten()
    for i, v in enumerate(flat):
        cin, cout, k = i // (Cout * K), (i % (Cout * K)) // K, i % K
        print(f"  [{i}] = {v.item():.1f} (cin={cin}, cout={cout}, k={k})")

if __name__ == "__main__":
    main()
