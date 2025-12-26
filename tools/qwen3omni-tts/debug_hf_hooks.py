#!/usr/bin/env python3
"""
Extract per-token intermediate tensors from HuggingFace Talker using hooks.
This avoids manually reimplementing the forward pass.
"""

import torch
import numpy as np
from pathlib import Path
from transformers import Qwen3OmniMoeForConditionalGeneration
import sys

# Global dict to store hooked outputs
hooked_outputs = {}


def make_hook(name):
    """Create a hook function that captures the output."""
    def hook(module, input, output):
        # output is typically (hidden_states, ...) for decoder layers
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # Store all tokens, not just last
        hooked_outputs[name] = hidden.detach().cpu().float().numpy()
    return hook


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

    # Load model with bfloat16
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
    talker = model.talker.model
    print(f"Talker has {len(talker.layers)} layers")

    # Register hooks on layers of interest
    hooks = []
    for layer_idx in [0, 1, 2, 3]:
        layer = talker.layers[layer_idx]
        # Hook after the full layer (including residual)
        h = layer.register_forward_hook(make_hook(f"layer{layer_idx}_out"))
        hooks.append(h)

        # Hook on input_layernorm (before attention)
        h = layer.input_layernorm.register_forward_hook(
            make_hook(f"layer{layer_idx}_attn_norm_out")
        )
        hooks.append(h)

        # Hook on self_attn (attention output)
        h = layer.self_attn.register_forward_hook(make_hook(f"layer{layer_idx}_attn_out"))
        hooks.append(h)

        # Hook on post_attention_layernorm (ffn_norm input after attention)
        h = layer.post_attention_layernorm.register_forward_hook(
            make_hook(f"layer{layer_idx}_ffn_norm_out")
        )
        hooks.append(h)

        # Hook on MLP/MoE
        h = layer.mlp.register_forward_hook(make_hook(f"layer{layer_idx}_mlp_out"))
        hooks.append(h)

    # Convert prefill embeddings to tensor
    inputs_embeds = torch.tensor(prefill_embeds, dtype=torch.bfloat16, device="cuda").unsqueeze(0)
    print(f"Input shape: {inputs_embeds.shape}")

    # Create attention mask
    batch_size, seq_len, _ = inputs_embeds.shape
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)

    # Run forward pass through talker
    print("\nRunning Talker forward pass...")
    with torch.no_grad():
        outputs = model.talker(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )

    # Check what attributes the output has
    print(f"Output type: {type(outputs)}")
    print(f"Output keys: {outputs.keys() if hasattr(outputs, 'keys') else dir(outputs)}")

    # Get the hidden state (may have different attribute name)
    final_hidden = None
    try:
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
            hs = outputs.hidden_states[-1]
            if hs is not None:
                final_hidden = hs.detach().cpu().float().numpy()
        if final_hidden is None and hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
            final_hidden = outputs.last_hidden_state.detach().cpu().float().numpy()
    except Exception as e:
        print(f"Note: Could not get final hidden from outputs: {e}")
    if final_hidden is None:
        print("Note: Final hidden not available from outputs")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save hooked outputs
    print("\n=== Saving hooked outputs ===")
    for name, arr in hooked_outputs.items():
        save_path = output_dir / f"hooked_{name}.npy"
        np.save(save_path, arr)
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
        # Print per-token stats for first few tokens
        if len(arr.shape) == 3:  # [batch, seq, hidden]
            for tok in range(min(3, arr.shape[1])):
                print(f"    Token {tok}: mean={arr[0, tok].mean():.6f}, std={arr[0, tok].std():.6f}")

    # Also save the final output for all tokens if available
    if final_hidden is not None:
        np.save(output_dir / "hooked_final_hidden.npy", final_hidden)
        print(f"\nFinal hidden: shape={final_hidden.shape}")
        for tok in range(final_hidden.shape[1]):
            print(f"  Token {tok}: mean={final_hidden[0, tok].mean():.6f}, std={final_hidden[0, tok].std():.6f}")

    print(f"\n=== Done! Output saved to {output_dir} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
