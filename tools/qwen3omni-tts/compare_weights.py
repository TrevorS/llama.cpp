#!/usr/bin/env python3
"""Compare Talker weight tensors between GGUF and HuggingFace safetensors.

Uses safetensors directly to avoid loading the full model (OOM).
"""

import numpy as np
from pathlib import Path
import json
import sys

# Add gguf-py to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gguf-py"))
from gguf import GGUFReader
from safetensors import safe_open


def compare_tensor(name: str, gguf_data: np.ndarray, hf_data: np.ndarray):
    """Compare two tensors and report statistics."""
    # Note: gguf-py reader already returns data in numpy order (reversed from ggml metadata)
    # So no transpose is needed here

    if gguf_data.shape != hf_data.shape:
        # Try reshaping if total elements match
        if gguf_data.size == hf_data.size:
            gguf_data = gguf_data.reshape(hf_data.shape)
        else:
            print(f"  {name}: SHAPE MISMATCH gguf={gguf_data.shape} hf={hf_data.shape}")
            return False

    # Check for exact match or close match
    if np.allclose(gguf_data, hf_data, rtol=1e-4, atol=1e-6):
        print(f"  {name}: ✓ MATCH (shape={hf_data.shape})")
        return True

    # Compute statistics
    diff = np.abs(gguf_data - hf_data)
    max_diff = diff.max()
    mean_diff = diff.mean()
    corr = np.corrcoef(gguf_data.flatten(), hf_data.flatten())[0, 1]

    if corr > 0.9999:
        print(f"  {name}: ✓ CLOSE (shape={hf_data.shape}, corr={corr:.6f})")
        return True

    print(f"  {name}: ✗ DIFFER (shape={hf_data.shape})")
    print(f"    GGUF: mean={gguf_data.mean():.6f}, std={gguf_data.std():.6f}")
    print(f"    HF:   mean={hf_data.mean():.6f}, std={hf_data.std():.6f}")
    print(f"    Diff: max={max_diff:.6e}, mean={mean_diff:.6e}, corr={corr:.6f}")

    return False


def load_safetensor(hf_path: Path, tensor_name: str) -> np.ndarray:
    """Load a single tensor from safetensors files."""
    import torch

    index_path = hf_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        if tensor_name not in index["weight_map"]:
            raise KeyError(f"Tensor {tensor_name} not in index")
        shard_file = hf_path / index["weight_map"][tensor_name]
    else:
        # Single file
        shard_file = hf_path / "model.safetensors"

    # Use torch framework which handles bf16 properly
    with safe_open(shard_file, framework="pt") as f:
        tensor = f.get_tensor(tensor_name)
        return tensor.float().numpy()


def main():
    gguf_path = "/models/qwen3-omni-30b-talker-f16-v3.gguf"
    hf_path = Path("/models/Qwen3-Omni-30B-A3B-Instruct")

    print("=" * 80)
    print("GGUF vs HuggingFace Weight Comparison (safetensors)")
    print("=" * 80)

    # Load GGUF
    print(f"\nLoading GGUF from {gguf_path}...")
    reader = GGUFReader(gguf_path)

    # Build tensor dict from GGUF
    gguf_tensors = {}
    for tensor in reader.tensors:
        gguf_tensors[tensor.name] = tensor.data

    print(f"Loaded {len(gguf_tensors)} tensors from GGUF")

    # Map GGUF names to HF safetensor names
    # Format: (gguf_name, hf_safetensor_name)
    tensor_mappings = [
        # Embedding (Talker uses codec_embedding, not embed_tokens)
        ("token_embd.weight", "talker.model.codec_embedding.weight"),
        # Layer 0 attention
        ("blk.0.attn_norm.weight", "talker.model.layers.0.input_layernorm.weight"),
        ("blk.0.attn_q.weight", "talker.model.layers.0.self_attn.q_proj.weight"),
        ("blk.0.attn_k.weight", "talker.model.layers.0.self_attn.k_proj.weight"),
        ("blk.0.attn_v.weight", "talker.model.layers.0.self_attn.v_proj.weight"),
        ("blk.0.attn_output.weight", "talker.model.layers.0.self_attn.o_proj.weight"),
        ("blk.0.attn_q_norm.weight", "talker.model.layers.0.self_attn.q_norm.weight"),
        ("blk.0.attn_k_norm.weight", "talker.model.layers.0.self_attn.k_norm.weight"),
        # Layer 0 FFN (MoE)
        ("blk.0.ffn_norm.weight", "talker.model.layers.0.post_attention_layernorm.weight"),
        ("blk.0.ffn_gate_inp.weight", "talker.model.layers.0.mlp.gate.weight"),
        # Layer 0 shared expert
        ("blk.0.ffn_gate_shexp.weight", "talker.model.layers.0.mlp.shared_expert.gate_proj.weight"),
        ("blk.0.ffn_up_shexp.weight", "talker.model.layers.0.mlp.shared_expert.up_proj.weight"),
        ("blk.0.ffn_down_shexp.weight", "talker.model.layers.0.mlp.shared_expert.down_proj.weight"),
        # Layer 0 expert 0 (spot check)
        ("blk.0.ffn_gate_exps.weight", "talker.model.layers.0.mlp.experts.0.gate_proj.weight"),
        # Output
        ("output_norm.weight", "talker.model.norm.weight"),
        ("output.weight", "talker.lm_head.weight"),
        # Layer 5 (where degradation accelerates)
        ("blk.5.attn_norm.weight", "talker.model.layers.5.input_layernorm.weight"),
        ("blk.5.attn_q.weight", "talker.model.layers.5.self_attn.q_proj.weight"),
        ("blk.5.attn_k.weight", "talker.model.layers.5.self_attn.k_proj.weight"),
        # Layer 7 (where correlation drops to 0.016)
        ("blk.7.attn_norm.weight", "talker.model.layers.7.input_layernorm.weight"),
        ("blk.7.attn_q.weight", "talker.model.layers.7.self_attn.q_proj.weight"),
    ]

    print("\n" + "=" * 80)
    print("Comparing Key Tensors")
    print("=" * 80)

    all_match = True
    for gguf_name, hf_name in tensor_mappings:
        if gguf_name not in gguf_tensors:
            print(f"  {gguf_name}: NOT IN GGUF")
            all_match = False
            continue

        try:
            hf_tensor = load_safetensor(hf_path, hf_name)
            gguf_tensor = gguf_tensors[gguf_name]

            if not compare_tensor(gguf_name, gguf_tensor, hf_tensor):
                all_match = False
        except KeyError as e:
            print(f"  {gguf_name}: HF tensor not found - {e}")
            all_match = False
        except Exception as e:
            print(f"  {gguf_name}: ERROR - {e}")
            all_match = False

    print("\n" + "=" * 80)
    if all_match:
        print("✓ All checked weights match!")
    else:
        print("✗ Some weights differ!")
    print("=" * 80)

    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
