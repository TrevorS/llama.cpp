#!/usr/bin/env python3
"""
Extract intermediate layer tensors from HuggingFace Talker model.
Focuses on layer 2 internals to debug MoE divergence.
"""

import torch
import numpy as np
from pathlib import Path
from transformers import Qwen3OmniMoeForConditionalGeneration
import sys

def main():
    model_path = "/models/Qwen3-Omni-30B-A3B-Instruct"
    output_dir = Path("/models/debug/hf_talker")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prefill embeddings (same as C++)
    prefill_path = output_dir / "prefill_embeds.npy"
    if not prefill_path.exists():
        print(f"Error: {prefill_path} not found. Run HF debug script first.")
        return 1

    prefill_embeds = np.load(prefill_path)
    print(f"Loaded prefill embeddings: {prefill_embeds.shape}")

    # Load model with trust_remote_code for Qwen3-Omni
    print(f"Loading model from {model_path}...")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    model.eval()

    # Enable talker if needed
    if not model.has_talker:
        model.enable_talker()

    # Access Talker through the model structure
    # Qwen3-Omni structure: model.talker.model.layers
    talker = model.talker.model

    print(f"Talker has {len(talker.layers)} layers")

    # Convert prefill embeddings to tensor (using bfloat16 to match model)
    inputs_embeds = torch.tensor(prefill_embeds, dtype=torch.bfloat16, device="cuda").unsqueeze(0)
    print(f"Input shape: {inputs_embeds.shape}")

    # Manual forward pass to capture intermediates
    hidden_states = inputs_embeds

    # Create attention mask and position ids
    batch_size, seq_len, _ = hidden_states.shape
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)

    # Create causal mask
    causal_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device="cuda"), diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
    attention_mask_4d = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.bfloat16, device="cuda")
    attention_mask_4d.masked_fill_(causal_mask, -65504.0)  # Use -65504 instead of -inf for bfloat16

    for layer_idx, layer in enumerate(talker.layers):
        print(f"\n=== Layer {layer_idx} ===")

        # Save input to layer
        if layer_idx in [1, 2]:
            np.save(output_dir / f"layer{layer_idx}_input.npy",
                   hidden_states[0, -1].detach().cpu().numpy())
            print(f"  Input: mean={hidden_states[0, -1].mean():.6f}, std={hidden_states[0, -1].std():.6f}")

        residual = hidden_states

        # Self attention
        hidden_states_normed = layer.input_layernorm(hidden_states)
        attn_output, _, _ = layer.self_attn(
            hidden_states_normed,
            attention_mask=attention_mask_4d,
            position_ids=position_ids,
        )
        hidden_states = residual + attn_output

        # ffn_inp equivalent (after attention residual)
        if layer_idx in [1, 2]:
            # Save ALL tokens, not just last
            np.save(output_dir / f"ffn_inp_layer{layer_idx}_all.npy",
                   hidden_states[0].detach().cpu().numpy())
            print(f"  ffn_inp all tokens: shape={hidden_states[0].shape}")
            for tok_idx in [0, hidden_states.shape[1]-1]:
                print(f"    Token {tok_idx}: mean={hidden_states[0, tok_idx].mean():.6f}, std={hidden_states[0, tok_idx].std():.6f}")

        residual = hidden_states

        # FFN norm
        hidden_states_normed = layer.post_attention_layernorm(hidden_states)

        if layer_idx in [1, 2]:
            # Save ALL tokens
            np.save(output_dir / f"ffn_norm_layer{layer_idx}_all.npy",
                   hidden_states_normed[0].detach().cpu().numpy())
            print(f"  ffn_norm all tokens: shape={hidden_states_normed[0].shape}")
            for tok_idx in [0, hidden_states_normed.shape[1]-1]:
                print(f"    Token {tok_idx}: mean={hidden_states_normed[0, tok_idx].mean():.6f}, std={hidden_states_normed[0, tok_idx].std():.6f}")

        # MoE forward
        mlp = layer.mlp

        # Get router logits
        router_logits = mlp.gate(hidden_states_normed)
        if layer_idx in [1, 2]:
            np.save(output_dir / f"router_logits_layer{layer_idx}.npy",
                   router_logits[0, -1].detach().cpu().numpy())
            print(f"  router_logits: shape={router_logits.shape}, max={router_logits[0, -1].max():.4f}")

            # Get top-k experts
            routing_weights = torch.softmax(router_logits, dim=-1)
            topk_weights, topk_indices = torch.topk(routing_weights, mlp.top_k, dim=-1)
            print(f"  top-{mlp.top_k} experts for last token: {topk_indices[0, -1].cpu().tolist()}")
            print(f"  top-{mlp.top_k} weights for last token: {topk_weights[0, -1].cpu().tolist()}")

        # Full MoE computation
        moe_output = mlp.moe_block(hidden_states_normed)

        if layer_idx in [1, 2]:
            # Save ALL tokens
            np.save(output_dir / f"ffn_moe_out_layer{layer_idx}_all.npy",
                   moe_output[0].detach().cpu().numpy())
            print(f"  ffn_moe_out all tokens: shape={moe_output[0].shape}")
            for tok_idx in [0, moe_output.shape[1]-1]:
                print(f"    Token {tok_idx}: mean={moe_output[0, tok_idx].mean():.6f}, std={moe_output[0, tok_idx].std():.6f}")

        # Shared expert (if exists)
        if hasattr(mlp, 'shared_expert') and mlp.shared_expert is not None:
            shared_output = mlp.shared_expert(hidden_states_normed)
            if layer_idx in [1, 2]:
                np.save(output_dir / f"ffn_shexp_layer{layer_idx}.npy",
                       shared_output[0, -1].detach().cpu().numpy())
                print(f"  ffn_shexp: mean={shared_output[0, -1].mean():.6f}, std={shared_output[0, -1].std():.6f}")

            # Combined MoE + shared
            combined = moe_output + shared_output
            if layer_idx in [1, 2]:
                np.save(output_dir / f"ffn_moe_shexp_out_layer{layer_idx}.npy",
                       combined[0, -1].detach().cpu().numpy())
                print(f"  ffn_moe_shexp_out: mean={combined[0, -1].mean():.6f}, std={combined[0, -1].std():.6f}")
        else:
            combined = moe_output

        # Final residual
        hidden_states = residual + combined

        # Save layer output
        np.save(output_dir / f"hidden_layer{layer_idx}.npy",
               hidden_states[0, -1].detach().cpu().numpy())
        print(f"  l_out: mean={hidden_states[0, -1].mean():.6f}, std={hidden_states[0, -1].std():.6f}")

    print(f"\n=== Done! Output saved to {output_dir} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
