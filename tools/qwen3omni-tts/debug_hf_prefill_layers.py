#!/usr/bin/env python3
"""
Debug HuggingFace Talker prefill with per-layer hidden state extraction.

Runs a single forward pass on prefill embeddings and saves:
- hidden_layer{0-19}.npy: Hidden states after each transformer layer
- hidden_after_norm.npy: Final hidden state after output norm
- prefill_logits.npy: Final logits

All tensors saved with shape [n_tokens, n_hidden] for easy comparison with C++.
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch


def load_bin_tensor(path: Path) -> np.ndarray:
    """Load tensor from binary file (C++ format)."""
    with open(path, 'rb') as f:
        ndims = struct.unpack('<I', f.read(4))[0]
        shape = [struct.unpack('<I', f.read(4))[0] for _ in range(ndims)]
        data = np.frombuffer(f.read(), dtype='<f4')
        return data.reshape(shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill", required=True, help="Path to prefill_embeds.bin")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", default="/models/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading prefill embeddings from {args.prefill}...")
    prefill_embeds = load_bin_tensor(Path(args.prefill))
    print(f"  Shape: {prefill_embeds.shape}")
    print(f"  Stats: mean={prefill_embeds.mean():.6f}, std={prefill_embeds.std():.6f}")

    print(f"\nLoading model from {args.model}...")
    from transformers import Qwen3OmniMoeForConditionalGeneration

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    if not model.has_talker:
        print("Enabling talker...")
        model.enable_talker()

    talker = model.talker

    # Convert to tensor
    prefill_tensor = torch.tensor(prefill_embeds, dtype=torch.bfloat16, device=args.device)
    prefill_tensor = prefill_tensor.unsqueeze(0)  # Add batch dim: [1, 9, 1024]
    print(f"Prefill tensor: {prefill_tensor.shape}")

    # Hook to capture layer outputs
    layer_outputs = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            # Output is hidden_states after the layer
            layer_outputs[layer_idx] = output[0].detach().float().cpu()
        return hook

    # Register hooks on each transformer layer
    handles = []
    for il, layer in enumerate(talker.model.layers):
        h = layer.register_forward_hook(make_hook(il))
        handles.append(h)

    print(f"\nRunning forward pass through {len(talker.model.layers)} layers...")

    with torch.no_grad():
        # Use the model's forward method which handles position embeddings properly
        n_tokens = prefill_tensor.shape[1]

        # Create attention mask
        causal_mask = torch.triu(
            torch.full((n_tokens, n_tokens), float('-inf'), device=args.device, dtype=torch.bfloat16),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        # Create position ids
        position_ids = torch.arange(n_tokens, device=args.device).unsqueeze(0)

        # Use the talker.model (the base model without lm_head) forward
        # This calls the full forward with proper RoPE handling
        model_outputs = talker.model(
            inputs_embeds=prefill_tensor,
            attention_mask=None,  # Let it create its own causal mask
            position_ids=position_ids,
            output_hidden_states=True,  # Get per-layer hidden states!
            return_dict=True,
        )

        # hidden_states is a tuple: (embed_output, layer0_out, layer1_out, ..., layerN_out)
        all_hidden_states = model_outputs.hidden_states
        hidden_normed = model_outputs.last_hidden_state  # Already normed

        print(f"Got {len(all_hidden_states)} hidden state tensors (including embeddings)")

        # Get logits - the head might be called codec_head or lm_head
        if hasattr(talker, 'codec_head'):
            logits = talker.codec_head(hidden_normed)
        elif hasattr(talker, 'lm_head'):
            logits = talker.lm_head(hidden_normed)
        else:
            print("Available attributes:", [a for a in dir(talker) if not a.startswith('_')])
            raise AttributeError("Cannot find lm_head or codec_head")

        # Convert to float for saving
        hidden_normed_f32 = hidden_normed.float().cpu()
        logits_f32 = logits.float().cpu()

        # NOTE: Layer outputs were captured by the forward hooks (lines 70-80)
        # DO NOT overwrite with all_hidden_states - that would give us normalized
        # outputs for the last layer, not raw layer outputs!
        # all_hidden_states[20] is AFTER output_norm, not raw layer 19.

    # Remove hooks
    for h in handles:
        h.remove()

    # Save layer outputs
    print(f"\n=== Saving per-layer outputs ===")
    for il in range(len(talker.model.layers)):
        if il in layer_outputs:
            hs = layer_outputs[il].squeeze(0).numpy()  # [n_tokens, hidden]
            out_file = output_path / f"hidden_layer{il}.npy"
            np.save(out_file, hs)
            print(f"  Layer {il:2d}: shape={hs.shape}, mean={hs.mean():.6f}, std={hs.std():.6f}")
        else:
            print(f"  Layer {il:2d}: NOT CAPTURED")

    # Save final norm output
    hs_norm = hidden_normed_f32.squeeze(0).numpy()
    np.save(output_path / "hidden_after_norm.npy", hs_norm)
    print(f"  After norm: shape={hs_norm.shape}, mean={hs_norm.mean():.6f}, std={hs_norm.std():.6f}")

    # Save logits (last token only for comparison)
    logits_np = logits_f32.squeeze(0)[-1].numpy()  # [vocab]
    np.save(output_path / "prefill_logits.npy", logits_np)
    print(f"  Logits (last token): shape={logits_np.shape}")

    # Print top-5 predicted tokens
    top5 = logits_np.argsort()[::-1][:5]
    print(f"\n  Top-5 logits:")
    for i, tok_id in enumerate(top5):
        print(f"    [{i}] token={tok_id}, logit={logits_np[tok_id]:.4f}")

    print(f"\n=== Done ===")
    print(f"Output directory: {output_path}")


if __name__ == "__main__":
    main()
