#!/usr/bin/env python3
"""Check memory layout of C++ tensor dumps."""

import numpy as np
import os

def main():
    # Load C++ transconv output (raw bytes)
    cpp_output_path = "/models/debug/cpp_tokens_match/cnxt0_transconv_raw.bin"
    n_floats = os.path.getsize(cpp_output_path) // 4
    raw_data = np.fromfile(cpp_output_path, dtype='<f4')
    print(f"Total floats: {n_floats}")
    print(f"Expected shape: 32 x 1024 = {32 * 1024}")

    # GGML stores in column-major: ne[0] varies fastest
    # For conv_transpose_1d output: ne[0]=seq_out=32, ne[1]=channels=1024
    # So memory order is: [seq0,ch0], [seq1,ch0], ..., [seq31,ch0], [seq0,ch1], ...

    # In numpy terms (row-major), we need to reshape as [channels, seq] then transpose
    # Or equivalently, use Fortran order
    cpp_fortran = raw_data.reshape((1024, 32), order='F').T  # [seq, channels]
    cpp_c_order = raw_data.reshape((32, 1024))  # [seq, channels] if C-order dump

    print(f"\nFirst 5 raw values: {raw_data[:5]}")
    print(f"\nFortran interpretation [0, :5]: {cpp_fortran[0, :5]}")
    print(f"C-order interpretation [0, :5]: {cpp_c_order[0, :5]}")

    # Let's also check if the values seem more sensible one way
    print(f"\nFortran mean per seq: {cpp_fortran.mean(axis=1)[:5]}")
    print(f"C-order mean per seq: {cpp_c_order.mean(axis=1)[:5]}")

    # Load HuggingFace output for comparison
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    # Load C++ transconv input
    cpp_input_path = "/models/debug/cpp_tokens_match/cnxt0_transconv_input.bin"
    seq_len = os.path.getsize(cpp_input_path) // 4 // 1024
    cpp_input_raw = np.fromfile(cpp_input_path, dtype='<f4')

    # Try both interpretations for input too
    # Input has GGML shape [seq, channels] = [16, 1024]
    # ne[0]=16, ne[1]=1024, so seq varies fastest
    input_fortran = cpp_input_raw.reshape((1024, 16), order='F').T
    input_c_order = cpp_input_raw.reshape((16, 1024))

    print(f"\n=== Input interpretation ===")
    print(f"Input first 5 raw: {cpp_input_raw[:5]}")
    print(f"Fortran [0, :5]: {input_fortran[0, :5]}")
    print(f"C-order [0, :5]: {input_c_order[0, :5]}")

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
                    elif 'upsample.0.0.conv.bias' in name:
                        hf_bias = sf.get_tensor(name).float()
            if hf_weight is not None:
                break

    # Run HF transconv with Fortran-interpreted input
    print(f"\n=== Testing with Fortran-order input ===")
    input_torch_f = torch.from_numpy(input_fortran.copy()).unsqueeze(0).permute(0, 2, 1)
    hf_output_f = F.conv_transpose1d(input_torch_f, hf_weight, hf_bias, stride=2)
    hf_output_f_np = hf_output_f[0].permute(1, 0).detach().numpy()

    corr_f_f = np.corrcoef(cpp_fortran.flatten(), hf_output_f_np.flatten())[0, 1]
    corr_c_f = np.corrcoef(cpp_c_order.flatten(), hf_output_f_np.flatten())[0, 1]
    print(f"Fortran input -> HF output correlation with Fortran output: {corr_f_f:.6f}")
    print(f"Fortran input -> HF output correlation with C-order output: {corr_c_f:.6f}")

    # Run HF transconv with C-order interpreted input
    print(f"\n=== Testing with C-order input ===")
    input_torch_c = torch.from_numpy(input_c_order.copy()).unsqueeze(0).permute(0, 2, 1)
    hf_output_c = F.conv_transpose1d(input_torch_c, hf_weight, hf_bias, stride=2)
    hf_output_c_np = hf_output_c[0].permute(1, 0).detach().numpy()

    corr_f_c = np.corrcoef(cpp_fortran.flatten(), hf_output_c_np.flatten())[0, 1]
    corr_c_c = np.corrcoef(cpp_c_order.flatten(), hf_output_c_np.flatten())[0, 1]
    print(f"C-order input -> HF output correlation with Fortran output: {corr_f_c:.6f}")
    print(f"C-order input -> HF output correlation with C-order output: {corr_c_c:.6f}")

    # Best match?
    correlations = {
        'Fortran-in/Fortran-out': corr_f_f,
        'Fortran-in/C-out': corr_c_f,
        'C-in/Fortran-out': corr_f_c,
        'C-in/C-out': corr_c_c,
    }
    best = max(correlations, key=correlations.get)
    print(f"\n=== Best match: {best} with correlation {correlations[best]:.6f} ===")

if __name__ == "__main__":
    main()
