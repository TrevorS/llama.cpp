#!/usr/bin/env python3
"""Check Expert 93 weights specifically since Token 0 routes 97% to it."""

import sys
sys.path.insert(0, '/app/gguf-py')

import numpy as np
from gguf import GGUFReader
from safetensors import safe_open
import torch
import json


def main():
    gguf_path = "/models/qwen3-omni-30b-talker-f16-v4.gguf"
    hf_path = "/models/Qwen3-Omni-30B-A3B-Instruct"

    print("=" * 80)
    print("Expert 93 Weight Analysis (Layer 2)")
    print("=" * 80)

    # Load GGUF
    print("\nLoading GGUF...")
    reader = GGUFReader(gguf_path)

    gguf_tensors = {}
    for t in reader.tensors:
        if t.name.startswith('blk.2.ffn'):
            gguf_tensors[t.name] = t.data.astype(np.float32)

    # Load HF
    print("Loading HuggingFace...")
    with open(f"{hf_path}/model.safetensors.index.json") as f:
        index = json.load(f)

    # Find which shard has layer 2
    layer2_files = set()
    for name, shard in index["weight_map"].items():
        if "talker.model.layers.2." in name:
            layer2_files.add(shard)

    hf_tensors = {}
    for shard in layer2_files:
        shard_path = f"{hf_path}/{shard}"
        with safe_open(shard_path, framework="pt") as f:
            for name in f.keys():
                if "talker.model.layers.2.mlp.experts.93" in name:
                    hf_tensors[name] = f.get_tensor(name).float().numpy()

    print(f"\nHF Expert 93 tensors: {list(hf_tensors.keys())}")

    # Compare Expert 93 weights
    print("\n=== Expert 93 Comparison ===")

    exp_id = 93

    # Gate projection
    gguf_gate_exps = gguf_tensors.get("blk.2.ffn_gate_exps.weight")
    hf_gate = hf_tensors.get(f"talker.model.layers.2.mlp.experts.{exp_id}.gate_proj.weight")

    if gguf_gate_exps is not None and hf_gate is not None:
        gguf_gate = gguf_gate_exps[exp_id]  # (384, 1024)
        print(f"\ngate_proj:")
        print(f"  GGUF shape: {gguf_gate.shape}, HF shape: {hf_gate.shape}")
        print(f"  GGUF stats: mean={gguf_gate.mean():.6f}, std={gguf_gate.std():.6f}")
        print(f"  HF   stats: mean={hf_gate.mean():.6f}, std={hf_gate.std():.6f}")
        print(f"  GGUF range: [{gguf_gate.min():.4f}, {gguf_gate.max():.4f}]")
        print(f"  HF   range: [{hf_gate.min():.4f}, {hf_gate.max():.4f}]")

        if gguf_gate.shape == hf_gate.shape:
            corr = np.corrcoef(gguf_gate.flatten(), hf_gate.flatten())[0, 1]
            diff = np.abs(gguf_gate - hf_gate)
            print(f"  Correlation: {corr:.10f}")
            print(f"  Max diff: {diff.max():.6e}")
            print(f"  Mean diff: {diff.mean():.6e}")

        # Check for NaN/Inf
        print(f"  GGUF has NaN: {np.isnan(gguf_gate).any()}")
        print(f"  GGUF has Inf: {np.isinf(gguf_gate).any()}")

    # Up projection
    gguf_up_exps = gguf_tensors.get("blk.2.ffn_up_exps.weight")
    hf_up = hf_tensors.get(f"talker.model.layers.2.mlp.experts.{exp_id}.up_proj.weight")

    if gguf_up_exps is not None and hf_up is not None:
        gguf_up = gguf_up_exps[exp_id]
        print(f"\nup_proj:")
        print(f"  GGUF shape: {gguf_up.shape}, HF shape: {hf_up.shape}")
        print(f"  GGUF stats: mean={gguf_up.mean():.6f}, std={gguf_up.std():.6f}")
        print(f"  HF   stats: mean={hf_up.mean():.6f}, std={hf_up.std():.6f}")

        if gguf_up.shape == hf_up.shape:
            corr = np.corrcoef(gguf_up.flatten(), hf_up.flatten())[0, 1]
            diff = np.abs(gguf_up - hf_up)
            print(f"  Correlation: {corr:.10f}")
            print(f"  Max diff: {diff.max():.6e}")

    # Down projection
    gguf_down_exps = gguf_tensors.get("blk.2.ffn_down_exps.weight")
    hf_down = hf_tensors.get(f"talker.model.layers.2.mlp.experts.{exp_id}.down_proj.weight")

    if gguf_down_exps is not None and hf_down is not None:
        gguf_down = gguf_down_exps[exp_id]
        print(f"\ndown_proj:")
        print(f"  GGUF shape: {gguf_down.shape}, HF shape: {hf_down.shape}")
        print(f"  GGUF stats: mean={gguf_down.mean():.6f}, std={gguf_down.std():.6f}")
        print(f"  HF   stats: mean={hf_down.mean():.6f}, std={hf_down.std():.6f}")

        if gguf_down.shape == hf_down.shape:
            corr = np.corrcoef(gguf_down.flatten(), hf_down.flatten())[0, 1]
            diff = np.abs(gguf_down - hf_down)
            print(f"  Correlation: {corr:.10f}")
            print(f"  Max diff: {diff.max():.6e}")

    # Simulate computation with ACTUAL C++ input
    print("\n=== Simulated Expert 93 Computation (with actual C++ input) ===")

    # Load actual ffn_norm from C++ (Layer 2, Token 0)
    import struct

    def load_cpp_tensor(path):
        with open(path, 'rb') as f:
            ndims = struct.unpack('<I', f.read(4))[0]
            shape = [struct.unpack('<I', f.read(4))[0] for _ in range(ndims)]
            data = np.frombuffer(f.read(), dtype='<f4')
            return data.reshape(shape)

    ffn_norm = load_cpp_tensor("/models/debug/cpp_talker/ffn_norm_layer2.bin")
    print(f"\nLoaded ffn_norm Layer 2: shape={ffn_norm.shape}")

    # Token 0
    test_input = ffn_norm[0] if ffn_norm.ndim == 2 else ffn_norm

    print(f"\nTest input: mean={test_input.mean():.6f}, std={test_input.std():.6f}")

    # Gate * input
    gguf_gate = gguf_gate_exps[93]  # (384, 1024)
    gguf_up = gguf_up_exps[93]
    gguf_down = gguf_down_exps[93]

    gate_out = test_input @ gguf_gate.T  # (384,)
    up_out = test_input @ gguf_up.T
    swiglu = gate_out * (1 / (1 + np.exp(-gate_out))) * up_out  # silu(gate) * up
    down_out = swiglu @ gguf_down.T  # (1024,)

    print(f"\nGGUF Expert 93 simulation:")
    print(f"  gate_out: mean={gate_out.mean():.6f}, std={gate_out.std():.6f}")
    print(f"  up_out: mean={up_out.mean():.6f}, std={up_out.std():.6f}")
    print(f"  swiglu: mean={swiglu.mean():.6f}, std={swiglu.std():.6f}")
    print(f"  down_out: mean={down_out.mean():.6f}, std={down_out.std():.6f}")
    print(f"  down_out range: [{down_out.min():.4f}, {down_out.max():.4f}]")

    # Compare with HF
    hf_gate = hf_tensors[f"talker.model.layers.2.mlp.experts.93.gate_proj.weight"]
    hf_up = hf_tensors[f"talker.model.layers.2.mlp.experts.93.up_proj.weight"]
    hf_down = hf_tensors[f"talker.model.layers.2.mlp.experts.93.down_proj.weight"]

    gate_out_hf = test_input @ hf_gate.T
    up_out_hf = test_input @ hf_up.T
    swiglu_hf = gate_out_hf * (1 / (1 + np.exp(-gate_out_hf))) * up_out_hf
    down_out_hf = swiglu_hf @ hf_down.T

    print(f"\nHF Expert 93 simulation:")
    print(f"  gate_out: mean={gate_out_hf.mean():.6f}, std={gate_out_hf.std():.6f}")
    print(f"  swiglu: mean={swiglu_hf.mean():.6f}, std={swiglu_hf.std():.6f}")
    print(f"  down_out: mean={down_out_hf.mean():.6f}, std={down_out_hf.std():.6f}")
    print(f"  down_out range: [{down_out_hf.min():.4f}, {down_out_hf.max():.4f}]")

    print(f"\nComparison:")
    print(f"  down_out correlation: {np.corrcoef(down_out, down_out_hf)[0, 1]:.6f}")
    print(f"  down_out diff mean: {np.abs(down_out - down_out_hf).mean():.6e}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
