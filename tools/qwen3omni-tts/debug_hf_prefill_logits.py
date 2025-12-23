#!/usr/bin/env python3
"""
Extract logits from HuggingFace Talker prefill forward pass.
This does a single forward pass (not generate) to get exact logits.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Debug HuggingFace Talker prefill logits")
    parser.add_argument("--model", "-m", default="/models/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--output", "-o", default="/models/debug/hf_talker")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model from {args.model}...")
    from transformers import AutoTokenizer, Qwen3OmniMoeForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    if not model.has_talker:
        print("Enabling talker...")
        model.enable_talker()

    config = model.config

    # Use same input as C++
    text = "The quick brown fox jumps over the lazy dog"
    formatted_text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    print(f"Input: {text}")

    input_ids = tokenizer(formatted_text, return_tensors="pt").input_ids.to(device)
    print(f"Input tokens: {input_ids.shape}")

    with torch.no_grad():
        # Run Thinker
        thinker_outputs = model.thinker.generate(
            input_ids=input_ids,
            max_new_tokens=50,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        thinker_output_ids = thinker_outputs.sequences

        # Get embeddings
        full_embeds = model.thinker.model.embed_tokens(thinker_output_ids)

        # Find assistant segment
        im_start_id = config.im_start_token_id
        tokens_list = thinker_output_ids[0].tolist()
        n_input = input_ids.shape[1]

        assistant_start_idx = None
        for i in range(n_input - 1, -1, -1):
            if tokens_list[i] == im_start_id:
                assistant_start_idx = i
                break
        if assistant_start_idx is None:
            assistant_start_idx = n_input - 1

        im_end_id = config.im_end_token_id
        assistant_end_idx = len(tokens_list)
        if tokens_list[-1] == im_end_id:
            assistant_end_idx = len(tokens_list) - 1

        print(f"Assistant segment: {assistant_start_idx} to {assistant_end_idx}")

        # Project embeddings
        assistant_embeds = full_embeds[:, assistant_start_idx:assistant_end_idx]
        assistant_hidden = model.talker.text_projection(assistant_embeds)
        n_text = assistant_hidden.shape[1]
        print(f"Assistant hidden: {assistant_hidden.shape}")

        # TTS special embeddings
        tts_special_ids = torch.tensor([
            [config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]
        ], device=device)
        tts_special_raw = model.thinker.model.embed_tokens(tts_special_ids)
        tts_special_projected = model.talker.text_projection(tts_special_raw)

        tts_bos_embed = tts_special_projected[:, 0:1]
        tts_eos_embed = tts_special_projected[:, 1:2]
        tts_pad_embed = tts_special_projected[:, 2:3]

        # Trailing text hidden
        if n_text > 4:
            trailing_text_hidden = torch.cat([
                assistant_hidden[:, 4:],
                tts_eos_embed,
            ], dim=1)
        else:
            trailing_text_hidden = tts_eos_embed

        # Codec prefill tokens
        talker_config = config.talker_config if hasattr(config, 'talker_config') else config
        codec_nothink_id = getattr(talker_config, 'codec_nothink_id', 2155)
        codec_think_bos_id = getattr(talker_config, 'codec_think_bos_id', 2156)
        codec_think_eos_id = getattr(talker_config, 'codec_think_eos_id', 2157)
        codec_pad_id = getattr(talker_config, 'codec_pad_id', 2148)
        codec_bos_id = getattr(talker_config, 'codec_bos_id', 2149)
        speaker_id = 2302  # Ethan

        print(f"\nCodec special token IDs:")
        print(f"  nothink={codec_nothink_id}, think_bos={codec_think_bos_id}, think_eos={codec_think_eos_id}")
        print(f"  pad={codec_pad_id}, bos={codec_bos_id}, speaker={speaker_id}")

        # Build prefill: text + codec
        if n_text >= 4:
            text_prefill = torch.cat([
                assistant_hidden[:, :3],
                tts_pad_embed.expand(-1, 4, -1),
                tts_bos_embed,
                assistant_hidden[:, 3:4],
            ], dim=1)
        else:
            text_prefill = torch.cat([
                assistant_hidden,
                tts_pad_embed.expand(-1, 9 - n_text, -1),
            ], dim=1)

        codec_ids = torch.tensor([[
            codec_nothink_id,
            codec_think_bos_id,
            codec_think_eos_id,
            speaker_id,
            codec_pad_id,
            codec_bos_id,
        ]], device=device, dtype=torch.long)
        codec_embeds = model.talker.get_input_embeddings()(codec_ids)

        n_hidden = text_prefill.shape[-1]
        codec_prefill = torch.cat([
            torch.zeros(1, 3, n_hidden, device=device, dtype=codec_embeds.dtype),
            codec_embeds,
        ], dim=1)

        prefill_embeds = text_prefill + codec_prefill
        print(f"Prefill shape: {prefill_embeds.shape}")

        # Save prefill embeddings for comparison with C++
        np.save(output_path / "prefill_embeds.npy", prefill_embeds[0].float().cpu().numpy())
        print(f"Saved prefill embeddings to {output_path / 'prefill_embeds.npy'}")

        # === RUN FORWARD PASS TO GET LOGITS ===
        print("\n=== Running single forward pass for logits ===")

        talker = model.talker

        # Print model structure to understand attribute names
        print(f"Talker type: {type(talker)}")
        print(f"Talker attributes: {[a for a in dir(talker) if not a.startswith('_')][:20]}...")

        # Hook to capture hidden states
        hidden_states_captured = {}
        layer_outputs = {}

        def hook_norm_output(module, input, output):
            hidden_states_captured['norm_output'] = output.detach() if output is not None else None

        def make_layer_hook(layer_idx):
            def hook(module, input, output):
                # output is (hidden, present_kv, ...) or just hidden
                if isinstance(output, tuple):
                    layer_outputs[layer_idx] = output[0].detach()
                else:
                    layer_outputs[layer_idx] = output.detach()
            return hook

        # Register hooks
        handles = []
        # Hook norm for hidden state before lm_head
        handles.append(talker.model.norm.register_forward_hook(hook_norm_output))
        # Hook all layers to see where divergence starts
        for layer_idx in range(len(talker.model.layers)):
            handles.append(talker.model.layers[layer_idx].register_forward_hook(make_layer_hook(layer_idx)))

        # Run forward pass with prefill
        outputs = talker(
            inputs_embeds=prefill_embeds,
            use_cache=True,
            return_dict=True,
        )

        # Clean up hooks
        for h in handles:
            h.remove()

        logits = outputs.logits[0, -1]  # [vocab_size]
        print(f"Logits shape: {logits.shape}")

        # Get hidden state after final RMSNorm (before lm_head) from hook
        if 'norm_output' in hidden_states_captured and hidden_states_captured['norm_output'] is not None:
            last_hidden = hidden_states_captured['norm_output']
            hidden_after_layers = last_hidden[0, -1]
            print(f"Hidden state (norm output) shape: {hidden_after_layers.shape}")
            print(f"Hidden state: mean={hidden_after_layers.mean().item():.6f}, std={hidden_after_layers.std().item():.6f}")
            print(f"  first 8: [{', '.join(f'{x:.4f}' for x in hidden_after_layers[:8].tolist())}]")
            np.save(output_path / "hidden_after_norm.npy", hidden_after_layers.float().cpu().numpy())
        else:
            print("WARNING: Could not capture hidden state before lm_head")
            print(f"  hidden_states_captured keys: {list(hidden_states_captured.keys())}")

        # Dump ALL layer outputs to find where divergence starts
        print("\n--- Layer-by-layer outputs ---")
        for layer_idx in sorted(layer_outputs.keys()):
            layer_out = layer_outputs[layer_idx][0, -1]  # [hidden] at last position
            np.save(output_path / f"hidden_layer{layer_idx}.npy", layer_out.float().cpu().numpy())
            print(f"  Layer {layer_idx}: mean={layer_out.mean().item():.6f}, std={layer_out.std().item():.6f}, first=[{layer_out[0].item():.4f}, {layer_out[1].item():.4f}]")

        # Save logits
        np.save(output_path / "prefill_logits.npy", logits.float().cpu().numpy())

        # Print top-10 logits
        values, indices = torch.topk(logits, 10)
        print("\nTop-10 logits from prefill:")
        for i, (val, idx) in enumerate(zip(values.tolist(), indices.tolist())):
            print(f"  [{i}] token={idx}, logit={val:.4f}")

        # Also print logits for specific tokens
        print("\nLogits for specific tokens:")
        special_tokens = [
            (135, "audio_135"),
            (1049, "audio_1049"),
            (2148, "codec_pad"),
            (2149, "codec_bos"),
            (2150, "codec_eos"),
            (2155, "codec_nothink"),
            (2156, "codec_think_bos"),
            (2157, "codec_think_eos"),
            (2301, "speaker_0"),
            (2302, "speaker_Ethan"),
            (2303, "speaker_Chelsie"),
        ]
        for tok_id, name in special_tokens:
            if tok_id < logits.shape[0]:
                print(f"  {name} ({tok_id}): {logits[tok_id].item():.4f}")

        # Get greedy token
        greedy_token = torch.argmax(logits).item()
        print(f"\nGreedy token: {greedy_token} (logit={logits[greedy_token].item():.4f})")

        # If greedy is audio token, show it
        if greedy_token < 2048:
            print(f"  -> Audio token (normal)")
        else:
            print(f"  -> Special token (abnormal!)")

    print(f"\nLogits saved to: {output_path / 'prefill_logits.npy'}")


if __name__ == "__main__":
    main()
