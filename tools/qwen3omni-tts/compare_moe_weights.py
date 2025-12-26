#!/usr/bin/env python3
"""Compare MoE expert weights between GGUF and HuggingFace for layer 2."""

import sys
sys.path.insert(0, '/app/gguf-py')

import numpy as np
from gguf import GGUFReader
from safetensors import safe_open


def main():
    gguf_path = "/models/qwen3-omni-30b-talker-f16-v4.gguf"
    hf_path = "/models/Qwen3-Omni-30B-A3B-Instruct"

    print("=" * 80)
    print("Layer 2 MoE Weight Comparison: GGUF vs HuggingFace")
    print("=" * 80)

    # Load GGUF
    print("\nLoading GGUF...")
    reader = GGUFReader(gguf_path)

    # Find layer 2 MoE tensors
    gguf_tensors = {}
    for t in reader.tensors:
        if t.name.startswith('blk.2.ffn'):
            gguf_tensors[t.name] = t.data

    print(f"Found {len(gguf_tensors)} layer 2 FFN tensors in GGUF")
    for name in sorted(gguf_tensors.keys()):
        print(f"  {name}: shape={gguf_tensors[name].shape}")

    # Load HF weights
    print("\nLoading HuggingFace safetensors...")

    # Find which safetensor shard has Talker layer 2
    import json
    with open(f"{hf_path}/model.safetensors.index.json") as f:
        index = json.load(f)

    # Find Talker layer 2 weight files
    layer2_files = set()
    for name, shard in index["weight_map"].items():
        if "talker.model.layers.2." in name:
            layer2_files.add(shard)

    print(f"Talker layer 2 weights in shards: {layer2_files}")

    # Load the relevant shards using torch to handle bfloat16
    import torch
    hf_tensors = {}
    for shard in layer2_files:
        shard_path = f"{hf_path}/{shard}"
        print(f"Loading {shard}...")
        with safe_open(shard_path, framework="pt") as f:
            for name in f.keys():
                if "talker.model.layers.2." in name:
                    hf_tensors[name] = f.get_tensor(name).float().numpy()

    print(f"\nFound {len(hf_tensors)} layer 2 tensors in HuggingFace")
    for name in sorted(hf_tensors.keys())[:10]:
        print(f"  {name}: shape={hf_tensors[name].shape}")
    if len(hf_tensors) > 10:
        print(f"  ... and {len(hf_tensors) - 10} more")

    # Compare key tensors
    print("\n" + "=" * 80)
    print("Weight Comparison")
    print("=" * 80)

    comparisons = [
        ("ffn_gate_inp", "blk.2.ffn_gate_inp.weight", "talker.model.layers.2.mlp.gate.weight"),
        ("ffn_norm", "blk.2.ffn_norm.weight", "talker.model.layers.2.post_attention_layernorm.weight"),
        ("ffn_gate_shexp", "blk.2.ffn_gate_shexp.weight", "talker.model.layers.2.mlp.shared_expert.gate_proj.weight"),
        ("ffn_up_shexp", "blk.2.ffn_up_shexp.weight", "talker.model.layers.2.mlp.shared_expert.up_proj.weight"),
        ("ffn_down_shexp", "blk.2.ffn_down_shexp.weight", "talker.model.layers.2.mlp.shared_expert.down_proj.weight"),
    ]

    for label, gguf_name, hf_name in comparisons:
        print(f"\n--- {label} ---")
        if gguf_name not in gguf_tensors:
            print(f"  GGUF tensor {gguf_name} not found!")
            continue
        if hf_name not in hf_tensors:
            print(f"  HF tensor {hf_name} not found!")
            continue

        gguf_w = gguf_tensors[gguf_name].astype(np.float32)
        hf_w = hf_tensors[hf_name].astype(np.float32)

        print(f"  GGUF shape: {gguf_w.shape}, HF shape: {hf_w.shape}")

        # Handle potential transposition
        if gguf_w.shape != hf_w.shape:
            if gguf_w.T.shape == hf_w.shape:
                gguf_w = gguf_w.T
                print(f"  Transposed GGUF to match")
            elif gguf_w.shape == hf_w.T.shape:
                hf_w = hf_w.T
                print(f"  Transposed HF to match")

        if gguf_w.shape == hf_w.shape:
            diff = np.abs(gguf_w - hf_w)
            corr = np.corrcoef(gguf_w.flatten(), hf_w.flatten())[0, 1]
            print(f"  Correlation: {corr:.6f}")
            print(f"  Mean abs diff: {diff.mean():.6e}")
            print(f"  Max abs diff: {diff.max():.6e}")
            print(f"  GGUF first 5: {gguf_w.flatten()[:5]}")
            print(f"  HF   first 5: {hf_w.flatten()[:5]}")
            if corr < 0.999:
                print(f"  *** POSSIBLE WEIGHT MISMATCH! ***")
        else:
            print(f"  Shape mismatch even after transpose!")

    # Compare expert weights for multiple experts
    print("\n--- Expert weights comparison ---")

    # GGUF shape: (128, 384, 1024) for gate/up, (128, 1024, 384) for down
    # HF shape: experts.{n}.gate_proj.weight = (384, 1024)

    gguf_gate_exps = gguf_tensors.get("blk.2.ffn_gate_exps.weight")
    gguf_up_exps = gguf_tensors.get("blk.2.ffn_up_exps.weight")
    gguf_down_exps = gguf_tensors.get("blk.2.ffn_down_exps.weight")

    print(f"GGUF gate_exps shape: {gguf_gate_exps.shape}")  # (128, 384, 1024)
    print(f"GGUF up_exps shape: {gguf_up_exps.shape}")      # (128, 384, 1024)
    print(f"GGUF down_exps shape: {gguf_down_exps.shape}")  # (128, 1024, 384)

    # Check a few experts
    for exp_id in [0, 1, 63, 127]:
        print(f"\n  Expert {exp_id}:")

        # Gate projection
        hf_gate_name = f"talker.model.layers.2.mlp.experts.{exp_id}.gate_proj.weight"
        if hf_gate_name in hf_tensors:
            hf_gate = hf_tensors[hf_gate_name]  # (384, 1024)
            gguf_gate = gguf_gate_exps[exp_id]  # (384, 1024)

            corr = np.corrcoef(gguf_gate.flatten(), hf_gate.flatten())[0, 1]
            print(f"    gate_proj: corr={corr:.6f}, GGUF first 3={gguf_gate.flatten()[:3]}, HF first 3={hf_gate.flatten()[:3]}")
            if corr < 0.999:
                print(f"    *** MISMATCH! ***")

        # Up projection
        hf_up_name = f"talker.model.layers.2.mlp.experts.{exp_id}.up_proj.weight"
        if hf_up_name in hf_tensors:
            hf_up = hf_tensors[hf_up_name]  # (384, 1024)
            gguf_up = gguf_up_exps[exp_id]  # (384, 1024)

            corr = np.corrcoef(gguf_up.flatten(), hf_up.flatten())[0, 1]
            print(f"    up_proj:   corr={corr:.6f}, GGUF first 3={gguf_up.flatten()[:3]}, HF first 3={hf_up.flatten()[:3]}")
            if corr < 0.999:
                print(f"    *** MISMATCH! ***")

        # Down projection
        hf_down_name = f"talker.model.layers.2.mlp.experts.{exp_id}.down_proj.weight"
        if hf_down_name in hf_tensors:
            hf_down = hf_tensors[hf_down_name]  # (1024, 384)
            gguf_down = gguf_down_exps[exp_id]  # (1024, 384)

            corr = np.corrcoef(gguf_down.flatten(), hf_down.flatten())[0, 1]
            print(f"    down_proj: corr={corr:.6f}, GGUF first 3={gguf_down.flatten()[:3]}, HF first 3={hf_down.flatten()[:3]}")
            if corr < 0.999:
                print(f"    *** MISMATCH! ***")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
