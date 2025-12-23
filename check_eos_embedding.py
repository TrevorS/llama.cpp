#!/usr/bin/env python3
"""Check EOS token embedding properties in Talker model."""

import torch
from safetensors import safe_open

# Load Talker codec embeddings from the correct shard
# talker.model.codec_embedding.weight is in shard 14
shard_path = "/models/Qwen3-Omni-30B-A3B-Instruct/model-00014-of-00015.safetensors"

with safe_open(shard_path, framework="pt") as f:
    tok_embd = f.get_tensor("talker.model.codec_embedding.weight")
    print(f"Token embeddings shape: {tok_embd.shape}")

    # Check vocab size
    vocab_size = tok_embd.shape[0]
    hidden_dim = tok_embd.shape[1]
    print(f"Vocab size: {vocab_size}, Hidden dim: {hidden_dim}")

    # Key tokens from config
    tokens = {
        "codec_pad_id": 2148,
        "codec_bos_id": 2149,
        "codec_eos_id": 2150,  # The EOS we're looking for
        "codec_nothink_id": 2155,
        "codec_think_bos_id": 2156,
        "codec_think_eos_id": 2157,
    }

    # Compute norms for special tokens
    print("\n=== Special Token Embedding Norms ===")
    for name, idx in tokens.items():
        if idx < vocab_size:
            norm = tok_embd[idx].norm().item()
            print(f"{name} ({idx}): norm = {norm:.4f}")
        else:
            print(f"{name} ({idx}): OUT OF BOUNDS")

    # Compare with codec token statistics (0-2047)
    codec_norms = tok_embd[:2048].norm(dim=1)
    print(f"\n=== Codec Token Stats (0-2047) ===")
    print(f"Mean norm: {codec_norms.mean().item():.4f}")
    print(f"Std norm: {codec_norms.std().item():.4f}")
    print(f"Min norm: {codec_norms.min().item():.4f}")
    print(f"Max norm: {codec_norms.max().item():.4f}")

    # Check if EOS embedding is similar to codec tokens or drastically different
    eos_norm = tok_embd[2150].norm().item()
    codec_mean = codec_norms.mean().item()
    print(f"\n=== EOS Analysis ===")
    print(f"EOS norm: {eos_norm:.4f}")
    print(f"Codec mean norm: {codec_mean:.4f}")
    print(f"Ratio (EOS/codec_mean): {eos_norm/codec_mean:.4f}")

    # Check if EOS is all zeros
    eos_is_zero = torch.allclose(tok_embd[2150], torch.zeros_like(tok_embd[2150]), atol=1e-6)
    print(f"EOS is all zeros: {eos_is_zero}")

    # Check similarity between EOS and some codec tokens
    print(f"\n=== EOS Similarity to Sample Tokens ===")
    for test_idx in [0, 100, 500, 1000, 1500, 2000]:
        cos_sim = torch.nn.functional.cosine_similarity(
            tok_embd[2150].unsqueeze(0),
            tok_embd[test_idx].unsqueeze(0)
        ).item()
        print(f"cos_sim(EOS, token {test_idx}): {cos_sim:.4f}")
