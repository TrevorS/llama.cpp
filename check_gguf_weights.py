#!/usr/bin/env python3
"""Check GGUF model weight magnitudes for Code2Wav."""

import sys
sys.path.insert(0, 'gguf-py')
from gguf import GGUFReader
import numpy as np

# Path to the GGUF file
gguf_path = "/home/trevor/models/qwen3-omni-talker-f16.gguf"

reader = GGUFReader(gguf_path)

print("=== Code2Wav Weights from GGUF ===\n")

# Tensors to check - using actual GGUF names
patterns = [
    "code2wav.up.",      # ConvNeXt upsample
    "code2wav.pre.output_norm",     # Pre-transformer output norm
]

for tensor in reader.tensors:
    name = tensor.name
    # Check if matches any pattern
    if not any(p in name for p in patterns):
        continue

    data = tensor.data
    if hasattr(data, 'numpy'):
        data = data.numpy()
    elif not isinstance(data, np.ndarray):
        data = np.array(data)

    print(f"{name}:")
    print(f"  shape: {tensor.shape}, dtype: {tensor.tensor_type}")
    print(f"  min: {data.min():.6f}, max: {data.max():.6f}")
    print(f"  mean: {data.mean():.6f}, std: {data.std():.6f}")
    print()
