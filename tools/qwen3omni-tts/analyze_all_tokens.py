#!/usr/bin/env python3
"""Analyze C++ MoE output for ALL tokens, not just last."""

import struct
import numpy as np
from pathlib import Path


def load_cpp_tensor(path: Path) -> np.ndarray:
    """Load C++ tensor with dimension header."""
    with open(path, 'rb') as f:
        ndims = struct.unpack('<I', f.read(4))[0]
        shape = []
        for _ in range(ndims):
            shape.append(struct.unpack('<I', f.read(4))[0])
        data = np.frombuffer(f.read(), dtype='<f4')
        return data.reshape(shape)


def main():
    cpp_dir = Path("/models/debug/cpp_talker")

    print("=" * 80)
    print("C++ MoE Output Analysis - All Tokens")
    print("=" * 80)

    for layer_idx in [1, 2]:
        print(f"\n=== Layer {layer_idx} ===")

        # Check router logits for each token
        logits_path = cpp_dir / f"ffn_moe_logits_layer{layer_idx}.bin"
        if logits_path.exists():
            logits = load_cpp_tensor(logits_path)
            print(f"\nRouter logits shape: {logits.shape}")

            if logits.ndim == 2:
                print("\nPer-token router analysis:")
                for tok_idx in range(logits.shape[0]):
                    tok_logits = logits[tok_idx]
                    exp_logits = np.exp(tok_logits - tok_logits.max())
                    probs = exp_logits / exp_logits.sum()

                    top_k = 6
                    top_indices = np.argsort(tok_logits)[-top_k:][::-1]
                    top_probs_normalized = probs[top_indices] / probs[top_indices].sum()

                    top_experts_str = ", ".join([f"{idx}({top_probs_normalized[i]:.2f})" for i, idx in enumerate(top_indices[:3])])
                    print(f"  Token {tok_idx}: logit_mean={tok_logits.mean():.3f}, logit_std={tok_logits.std():.3f}, top3=[{top_experts_str}]")

        # Check ffn_inp (input to post_attention_layernorm)
        ffn_inp_path = cpp_dir / f"ffn_inp_layer{layer_idx}.bin"
        if ffn_inp_path.exists():
            ffn_inp = load_cpp_tensor(ffn_inp_path)
            print(f"\nffn_inp (after attn residual) shape: {ffn_inp.shape}")

            if ffn_inp.ndim == 2:
                print("\nPer-token ffn_inp statistics:")
                for tok_idx in range(ffn_inp.shape[0]):
                    tok = ffn_inp[tok_idx]
                    print(f"  Token {tok_idx}: mean={tok.mean():.6f}, std={tok.std():.6f}")

        moe_out_path = cpp_dir / f"ffn_moe_out_layer{layer_idx}.bin"
        if moe_out_path.exists():
            moe_out = load_cpp_tensor(moe_out_path)
            print(f"\nffn_moe_out shape: {moe_out.shape}")  # Should be (9, 1024)

            if moe_out.ndim == 2:
                print("\nPer-token statistics:")
                for tok_idx in range(moe_out.shape[0]):
                    tok = moe_out[tok_idx]
                    print(f"  Token {tok_idx}: mean={tok.mean():.6f}, std={tok.std():.6f}, min={tok.min():.4f}, max={tok.max():.4f}")

                print(f"\nOverall: mean={moe_out.mean():.6f}, std={moe_out.std():.6f}")
                print(f"  min={moe_out.min():.6f}, max={moe_out.max():.6f}")

        # Check ffn_norm (input to MoE)
        norm_path = cpp_dir / f"ffn_norm_layer{layer_idx}.bin"
        if norm_path.exists():
            norm = load_cpp_tensor(norm_path)
            print(f"\nffn_norm (MoE input) shape: {norm.shape}")

            if norm.ndim == 2:
                print("\nPer-token ffn_norm statistics:")
                for tok_idx in range(norm.shape[0]):
                    tok = norm[tok_idx]
                    print(f"  Token {tok_idx}: mean={tok.mean():.6f}, std={tok.std():.6f}")

        # Compare with hidden layer output
        hidden_path = cpp_dir / f"hidden_layer{layer_idx}.bin"
        if hidden_path.exists():
            hidden = load_cpp_tensor(hidden_path)
            print(f"\nhidden_layer{layer_idx} shape: {hidden.shape}")

            if hidden.ndim == 2:
                print("\nPer-token hidden statistics:")
                for tok_idx in range(hidden.shape[0]):
                    tok = hidden[tok_idx]
                    print(f"  Token {tok_idx}: mean={tok.mean():.6f}, std={tok.std():.6f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
