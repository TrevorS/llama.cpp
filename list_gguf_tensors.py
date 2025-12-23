#!/usr/bin/env python3
"""List all tensors in GGUF file."""

import sys
sys.path.insert(0, 'gguf-py')
from gguf import GGUFReader

# Path to the GGUF file
gguf_path = "/home/trevor/models/qwen3-omni-talker-f16.gguf"

reader = GGUFReader(gguf_path)

print(f"=== Tensors in {gguf_path} ===\n")
print(f"Total tensors: {len(reader.tensors)}\n")

for tensor in reader.tensors:
    print(f"{tensor.name}: shape={tensor.shape}, type={tensor.tensor_type}")
