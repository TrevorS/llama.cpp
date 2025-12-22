#!/usr/bin/env python3
"""Test weight ordering between GGUF and HuggingFace."""

import sys
sys.path.insert(0, '/app/src/gguf-py')

import numpy as np
from safetensors import safe_open
import gguf

# Load HuggingFace weight
hf_weight = None
model_path = "/models/Qwen3-Omni-30B-A3B-Instruct"
import os
for f in os.listdir(model_path):
    if f.endswith('.safetensors'):
        path = os.path.join(model_path, f)
        with safe_open(path, framework='pt') as sf:
            for name in sf.keys():
                if 'upsample.0.0.conv.weight' in name:
                    hf_weight = sf.get_tensor(name).float().numpy()
                    print(f"HF weight shape: {hf_weight.shape}")
                    print(f"HF weight strides: {hf_weight.strides}")
                    break
        if hf_weight is not None:
            break

# Load GGUF weight
reader = gguf.GGUFReader("/models/qwen3-omni-30b-talker-f16-v4.gguf")
for tensor in reader.tensors:
    if tensor.name == "code2wav.up.0.conv.weight":
        # The GGUF reader gives us the tensor shape
        print(f"\nGGUF tensor: {tensor.name}")
        print(f"GGUF shape (ne): {tensor.shape}")
        print(f"GGUF n_elements: {tensor.n_elements}")

        # Read the raw data
        gguf_weight = tensor.data.astype(np.float32)
        print(f"GGUF weight data shape: {gguf_weight.shape}")

        # Reshape to the stored shape
        # GGUF shape is [ne0, ne1, ne2] = [2, 1024, 1024]
        # This means memory is laid out as: innermost=ne0=2, middle=ne1=1024, outermost=ne2=1024
        gguf_weight_3d = gguf_weight.reshape(tensor.shape[::-1])  # reverse for numpy C-order
        print(f"GGUF weight 3D shape: {gguf_weight_3d.shape}")

        break

# Compare first few elements
print("\n=== Value Comparison ===")
print(f"HF weight[0,0,:] (first in-ch, first out-ch, all kernel): {hf_weight[0, 0, :]}")
print(f"GGUF raw first 10: {gguf_weight[:10]}")

# The question is: what's the relationship?
# HF: [Cin, Cout, K] = [1024, 1024, 2]
# GGUF ne: [2, 1024, 1024] = [K, ?, ?]

# If GGUF stores [K, Cout, Cin] in Fortran order (column-major):
# Memory layout: [k0,out0,in0], [k1,out0,in0], [k0,out1,in0], [k1,out1,in0], ...
# In numpy C-order with reversed dims: shape is [Cin, Cout, K]

# Let's check if values match
print(f"\nHF [0,0,0]: {hf_weight[0, 0, 0]}")
print(f"HF [0,0,1]: {hf_weight[0, 0, 1]}")
print(f"HF [0,1,0]: {hf_weight[0, 1, 0]}")
print(f"HF [1,0,0]: {hf_weight[1, 0, 0]}")

# GGUF in numpy with shape [1024, 1024, 2] (Cin, Cout, K if memory is [K, Cout, Cin])
print(f"\nGGUF 3D [0,0,0]: {gguf_weight_3d[0, 0, 0]}")
print(f"GGUF 3D [0,0,1]: {gguf_weight_3d[0, 0, 1]}")
print(f"GGUF 3D [0,1,0]: {gguf_weight_3d[0, 1, 0]}")
print(f"GGUF 3D [1,0,0]: {gguf_weight_3d[1, 0, 0]}")

# Check if they're the same
if np.allclose(hf_weight.flatten()[:100], gguf_weight[:100], rtol=1e-3):
    print("\n✓ HF and GGUF weights match in memory order!")
else:
    print("\n✗ HF and GGUF weights have DIFFERENT memory layout")
    # Find the mapping
    hf_flat = hf_weight.flatten()
    for i in range(10):
        # Find where hf_flat[i] appears in gguf
        matches = np.where(np.isclose(gguf_weight, hf_flat[i], rtol=1e-3))[0]
        if len(matches) > 0:
            print(f"  HF flat[{i}]={hf_flat[i]:.6f} found at GGUF[{matches[0]}]")
