#!/usr/bin/env python3
"""
Extract Q, K, V tensors from HuggingFace Talker to debug attention.
Uses monkey-patching to capture intermediate values.
"""

import torch
import numpy as np
from pathlib import Path
from transformers import Qwen3OmniMoeForConditionalGeneration
import sys

captured = {}


def patch_attention(model, layer_idx):
    """Monkey-patch attention to capture Q, K, V."""
    layer = model.talker.model.layers[layer_idx]
    original_forward = layer.self_attn.forward

    def patched_forward(hidden_states, position_embeddings, attention_mask=None, **kwargs):
        # Capture input
        captured[f"layer{layer_idx}_attn_input"] = hidden_states.detach().cpu().float().numpy()

        # Get Q, K, V projections
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

        q = layer.self_attn.q_proj(hidden_states)
        k = layer.self_attn.k_proj(hidden_states)
        v = layer.self_attn.v_proj(hidden_states)

        captured[f"layer{layer_idx}_q_proj"] = q.detach().cpu().float().numpy()
        captured[f"layer{layer_idx}_k_proj"] = k.detach().cpu().float().numpy()
        captured[f"layer{layer_idx}_v_proj"] = v.detach().cpu().float().numpy()

        # After Q/K norm
        q_normed = layer.self_attn.q_norm(q.view(hidden_shape))
        k_normed = layer.self_attn.k_norm(k.view(hidden_shape))

        captured[f"layer{layer_idx}_q_norm"] = q_normed.detach().cpu().float().numpy()
        captured[f"layer{layer_idx}_k_norm"] = k_normed.detach().cpu().float().numpy()

        # Call original forward
        return original_forward(hidden_states, position_embeddings, attention_mask, **kwargs)

    layer.self_attn.forward = patched_forward


def main():
    model_path = "/models/Qwen3-Omni-30B-A3B-Instruct"
    output_dir = Path("/models/debug/hf_talker")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prefill embeddings
    prefill_path = output_dir / "prefill_embeds.npy"
    if not prefill_path.exists():
        print(f"Error: {prefill_path} not found.")
        return 1

    prefill_embeds = np.load(prefill_path)
    print(f"Loaded prefill embeddings: {prefill_embeds.shape}")

    # Load model
    print(f"Loading model from {model_path}...")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    model.eval()

    if not model.has_talker:
        model.enable_talker()

    # Patch layer 0 attention
    patch_attention(model, 0)

    # Prepare inputs
    inputs_embeds = torch.tensor(prefill_embeds, dtype=torch.bfloat16, device="cuda").unsqueeze(0)
    batch_size, seq_len, _ = inputs_embeds.shape
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)

    print(f"\nRunning Talker forward pass...")
    with torch.no_grad():
        outputs = model.talker(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )

    # Save captured values
    print("\n=== Saving captured Q/K/V values ===")
    for name, arr in captured.items():
        save_path = output_dir / f"debug_{name}.npy"
        np.save(save_path, arr)
        print(f"  {name}: shape={arr.shape}")
        # Print stats for last token
        if len(arr.shape) >= 2:
            last = arr[0, -1] if len(arr.shape) == 3 else arr[-1]
            print(f"    Last token: mean={last.mean():.6f}, std={last.std():.6f}")

    print(f"\n=== Done! ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
