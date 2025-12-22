#!/usr/bin/env python3
"""Compare per-layer RMSNorm weights between GGUF and HuggingFace."""

import numpy as np
import os
from pathlib import Path
from safetensors import safe_open
from gguf import GGUFReader


def main():
    model_path = "/models/Qwen3-Omni-30B-A3B-Instruct"
    gguf_path = "/models/qwen3-omni-30b-talker-f16-v4.gguf"

    print("=" * 60)
    print("Per-Layer RMSNorm Comparison: GGUF vs HuggingFace")
    print("=" * 60)

    # Load from HuggingFace
    print("\n=== Loading from HuggingFace ===\n")
    hf_norms = {}
    for f in sorted(os.listdir(model_path)):
        if f.endswith('.safetensors'):
            path = os.path.join(model_path, f)
            with safe_open(path, framework='pt') as sf:
                for name in sf.keys():
                    if 'code2wav.pre_transformer.layers' in name and 'layernorm' in name:
                        tensor = sf.get_tensor(name).float().numpy()
                        hf_norms[name] = tensor
                        # Only print first few
                        if len(hf_norms) <= 4:
                            print(f"{name}: mean={tensor.mean():.6f}, std={tensor.std():.6f}")

    print(f"... and {len(hf_norms) - 4} more")

    # Load from GGUF
    print("\n=== Loading from GGUF ===\n")
    reader = GGUFReader(gguf_path)
    gguf_norms = {}
    for tensor in reader.tensors:
        if "code2wav.pre.blk" in tensor.name and ("attn_norm" in tensor.name or "ffn_norm" in tensor.name):
            data = tensor.data.astype(np.float32)
            if len(tensor.shape) == 1:
                data = data.reshape(tensor.shape[0])
            gguf_norms[tensor.name] = data
            if len(gguf_norms) <= 4:
                print(f"{tensor.name}: mean={data.mean():.6f}, std={data.std():.6f}")

    print(f"... and {len(gguf_norms) - 4} more")

    # Compare
    print("\n=== Comparison ===\n")
    print(f"{'Layer':<8} | {'Type':<12} | {'HF mean':<10} | {'GGUF mean':<10} | {'Diff':<10} | {'Corr':<8}")
    print("-" * 75)

    mismatches = []
    for layer in range(8):
        for norm_type in ['attn', 'ffn']:
            # HF name pattern
            if norm_type == 'attn':
                hf_name = f"code2wav.pre_transformer.layers.{layer}.input_layernorm.weight"
            else:
                hf_name = f"code2wav.pre_transformer.layers.{layer}.post_attention_layernorm.weight"

            # GGUF name pattern
            gguf_name = f"code2wav.pre.blk.{layer}.{norm_type}_norm.weight"

            if hf_name in hf_norms and gguf_name in gguf_norms:
                hf_data = hf_norms[hf_name]
                gguf_data = gguf_norms[gguf_name]

                diff = np.abs(hf_data - gguf_data).max()
                corr = np.corrcoef(hf_data.flatten(), gguf_data.flatten())[0, 1]

                status = "✓" if corr > 0.9999 else "✗"
                print(f"{layer:<8} | {norm_type:<12} | {hf_data.mean():.6f}   | {gguf_data.mean():.6f}   | {diff:.6f}   | {corr:.6f} {status}")

                if corr < 0.9999:
                    mismatches.append((layer, norm_type, diff, corr))
            else:
                print(f"{layer:<8} | {norm_type:<12} | {'found' if hf_name in hf_norms else 'MISSING':<10} | {'found' if gguf_name in gguf_norms else 'MISSING':<10}")

    # Summary
    print("\n=== Summary ===\n")
    print(f"Total HF norms: {len(hf_norms)}")
    print(f"Total GGUF norms: {len(gguf_norms)}")
    if mismatches:
        print("\nMISMATCHES FOUND:")
        for layer, norm_type, diff, corr in mismatches:
            print(f"  Layer {layer} {norm_type}: diff={diff:.6f}, corr={corr:.6f}")
    else:
        print("\nAll norms match perfectly!")


if __name__ == "__main__":
    main()
