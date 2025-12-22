#!/usr/bin/env python3
"""Debug transposed convolution specifically."""

import torch
import numpy as np
import struct
from pathlib import Path

def save_tensor(path, tensor):
    """Save tensor in GGML-compatible binary format."""
    arr = tensor.detach().cpu().float().numpy()
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', len(arr.shape)))
        for dim in arr.shape:
            f.write(struct.pack('<I', dim))
        f.write(arr.tobytes())

def load_tensor(path, expected_shape=None):
    """Load tensor from raw binary format.

    The C++ code saves raw float data without headers.
    Shape must be inferred from file size or provided explicitly.
    """
    data = np.fromfile(path, dtype='<f4')
    if expected_shape:
        data = data.reshape(expected_shape)
        return data, expected_shape
    return data, (len(data),)

def main():
    from transformers import AutoConfig, AutoModel
    from safetensors import safe_open
    import os

    model_path = "/models/Qwen3-Omni-30B-A3B-Instruct"
    out_dir = Path("/models/debug/transconv")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the ConvNeXt transposed conv weights directly
    print("Loading HF weights...")
    for f in os.listdir(model_path):
        if f.endswith('.safetensors'):
            path = os.path.join(model_path, f)
            with safe_open(path, framework='pt') as sf:
                for name in sf.keys():
                    if 'upsample.0.0.conv' in name:
                        tensor = sf.get_tensor(name)
                        print(f"  {name}: {list(tensor.shape)}, dtype={tensor.dtype}")
                        if 'weight' in name:
                            transconv_weight = tensor.float()
                        elif 'bias' in name:
                            transconv_bias = tensor.float()

    print(f"\nTransconv weight shape: {list(transconv_weight.shape)}")
    # HF shape: [in_channels, out_channels, kernel_size] = [1024, 1024, 2]

    # Create a test input matching what llama.cpp produces
    # Load the pre-transformer output from llama.cpp
    cpp_input_path = Path("/models/debug/cpp_tokens_match/cnxt0_transconv_input.bin")
    if cpp_input_path.exists():
        # File size: 65536 bytes = 16384 floats = 16 * 1024
        n_floats = cpp_input_path.stat().st_size // 4
        print(f"C++ input file: {n_floats} floats")
        # Shape is [seq, channels] = [16, 1024]
        seq_len = n_floats // 1024
        cpp_input, cpp_shape = load_tensor(cpp_input_path, expected_shape=(seq_len, 1024))
        print(f"Loaded C++ transconv input: shape={cpp_shape}")
        test_input = torch.from_numpy(cpp_input).float()
    else:
        print(f"C++ input not found at {cpp_input_path}")
        print("Creating synthetic input for testing...")
        # Create test input: [seq, channels] in C++ = [16, 1024]
        test_input = torch.randn(16, 1024)

    print(f"Test input shape: {list(test_input.shape)}")

    # ======= Apply HuggingFace's ConvTranspose1d =======
    # PyTorch ConvTranspose1d expects:
    #   input: [batch, in_channels, seq_len]
    #   weight: [in_channels, out_channels, kernel_size]
    # GGML conv_transpose_1d expects:
    #   input: [seq_len, in_channels]  (after transpose)
    #   kernel: [kernel_size, out_channels, in_channels]

    # Reshape input from [seq, channels] to [batch, channels, seq]
    hf_input = test_input.unsqueeze(0).permute(0, 2, 1)
    print(f"HF input (batch, channels, seq): {list(hf_input.shape)}")

    # Apply ConvTranspose1d
    import torch.nn.functional as F
    # ConvTranspose1d with stride=2
    hf_output = F.conv_transpose1d(hf_input, transconv_weight, transconv_bias, stride=2)
    print(f"HF output shape: {list(hf_output.shape)}")

    # Convert back to [seq, channels] for comparison with C++
    hf_output_2d = hf_output[0].permute(1, 0)  # [batch, channels, seq] -> [seq, channels]
    print(f"HF output 2D: {list(hf_output_2d.shape)}")

    # Save the HF output
    save_tensor(out_dir / "hf_transconv_output.bin", hf_output_2d)
    print(f"Saved HF transconv output to {out_dir / 'hf_transconv_output.bin'}")

    # Also save the weight in a readable format for debugging
    print("\n=== Weight Analysis ===")
    print(f"Weight shape: {list(transconv_weight.shape)}")
    print(f"Weight[0,0,:] (first in->first out, all kernel): {transconv_weight[0, 0, :].tolist()}")
    print(f"Weight[:2,:2,0] (first kernel pos, first 2x2 channels):")
    print(transconv_weight[:2, :2, 0])

    # Save weight in GGML format for comparison
    # GGML expects [kernel, out, in] but GGUF auto-reverses dimensions
    save_tensor(out_dir / "hf_transconv_weight.bin", transconv_weight)

    # ======= Load and compare C++ output =======
    cpp_output_path = Path("/models/debug/cpp_tokens_match/cnxt0_transconv_raw.bin")
    if cpp_output_path.exists():
        # After transconv with stride=2: output_seq = input_seq * 2 = 32
        # Shape should be [32, 1024] or [1024, 32] depending on C++ layout
        n_floats = cpp_output_path.stat().st_size // 4
        print(f"\n=== Comparison with C++ ===")
        print(f"C++ output file: {n_floats} floats")

        # Try to match HF output shape
        hf_np = hf_output_2d.detach().cpu().numpy()
        expected_shape = tuple(hf_np.shape)
        print(f"HF output shape: {expected_shape}")

        cpp_output, cpp_out_shape = load_tensor(cpp_output_path, expected_shape=expected_shape)
        print(f"C++ output shape: {cpp_out_shape}")

        hf_np = hf_output_2d.detach().cpu().numpy()

        if cpp_out_shape == tuple(hf_np.shape):
            diff = np.abs(cpp_output - hf_np)
            corr = np.corrcoef(cpp_output.flatten(), hf_np.flatten())[0, 1]
            print(f"Max diff: {diff.max():.6f}")
            print(f"Mean diff: {diff.mean():.6f}")
            print(f"Correlation: {corr:.6f}")

            # Find where max diff occurs
            max_idx = np.unravel_index(diff.argmax(), diff.shape)
            print(f"Max diff at {max_idx}: C++={cpp_output[max_idx]:.6f}, HF={hf_np[max_idx]:.6f}")

            # Compare first few values
            print(f"\nFirst row comparison:")
            print(f"  C++: {cpp_output[0, :5].tolist()}")
            print(f"  HF:  {hf_np[0, :5].tolist()}")
        else:
            print("Shape mismatch - comparing first common elements")
            min_shape = tuple(min(c, h) for c, h in zip(cpp_out_shape, hf_np.shape))
            cpp_slice = cpp_output[:min_shape[0], :min_shape[1]]
            hf_slice = hf_np[:min_shape[0], :min_shape[1]]
            diff = np.abs(cpp_slice - hf_slice)
            corr = np.corrcoef(cpp_slice.flatten(), hf_slice.flatten())[0, 1]
            print(f"Comparing first {min_shape} elements:")
            print(f"Max diff: {diff.max():.6f}")
            print(f"Correlation: {corr:.6f}")
    else:
        print(f"C++ output not found at {cpp_output_path}")

    # ======= Debug: Check if weight needs transposition =======
    print("\n=== Weight Transposition Test ===")
    # Try with transposed weight
    weight_permuted = transconv_weight.permute(2, 1, 0)  # [I,O,K] -> [K,O,I]
    print(f"Permuted weight shape: {list(weight_permuted.shape)}")

    # For this test, we need to adjust the conv_transpose1d call
    # Actually, conv_transpose1d's weight layout is fixed [in, out, kernel]
    # So let's check if our GGML implementation matches

if __name__ == "__main__":
    main()
