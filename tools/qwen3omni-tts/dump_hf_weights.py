#!/usr/bin/env python3
"""Dump HF Talker layer 0 weights for comparison with GGUF."""

import torch
from safetensors import safe_open
import os

model_dir = "/models/Qwen3-Omni-30B-A3B-Instruct"
files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]

patterns = [
    "talker.model.layers.0.input_layernorm",
    "talker.model.layers.0.post_attention_layernorm",
    "talker.model.layers.0.self_attn.q_proj.weight",
    "talker.model.layers.0.self_attn.k_proj.weight",
    "talker.model.layers.0.self_attn.v_proj.weight",
    "talker.model.layers.0.self_attn.o_proj.weight",
    "talker.model.layers.0.self_attn.q_norm.weight",
    "talker.model.layers.0.self_attn.k_norm.weight",
    "talker.model.norm.weight",  # final norm
    "talker.codec_head.weight",  # lm_head
]

found = {}
for fname in sorted(files):
    path = os.path.join(model_dir, fname)
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        for pattern in patterns:
            matching = [k for k in keys if pattern in k]
            for k in matching:
                if k not in found:
                    tensor = f.get_tensor(k)
                    found[k] = tensor

print("=== HuggingFace Talker Layer 0 Weights ===\n")
for k in sorted(found.keys()):
    tensor = found[k]
    print(f"{k}:")
    print(f"  shape={list(tensor.shape)}")
    print(f"  stats: mean={tensor.float().mean().item():.6f}, std={tensor.float().std().item():.6f}")
    print(f"  first 4: {tensor.flatten()[:4].tolist()}")
    print()
