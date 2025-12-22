#!/usr/bin/env python3
"""Debug LayerScale values and trace where std ratio divergence starts."""

import numpy as np
import os
from pathlib import Path
from safetensors import safe_open


def load_ggml_tensor(path: Path, ne: tuple) -> np.ndarray:
    """Load GGML tensor with Fortran order."""
    data = np.fromfile(path, dtype='<f4')
    expected_size = int(np.prod(ne))
    if len(data) != expected_size:
        raise ValueError(f"Size mismatch: got {len(data)}, expected {expected_size}")
    return data.reshape(ne, order='F')


def load_hf_tensor(path: Path) -> np.ndarray:
    """Load HF .npy tensor, removing batch dim if present."""
    arr = np.load(path)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def main():
    model_path = "/models/Qwen3-Omni-30B-A3B-Instruct"
    cpp_dir = Path("/models/debug/cpp_tokens_match")
    hf_dir = Path("/models/debug/hf_tensors_16f")

    n_frames = 16
    n_embd = 1024
    n_layers = 8

    print("=" * 60)
    print("LayerScale Analysis")
    print("=" * 60)

    # Load LayerScale weights from HuggingFace
    print("\n=== Loading HuggingFace LayerScale weights ===\n")
    layer_scales = {}

    for f in sorted(os.listdir(model_path)):
        if f.endswith('.safetensors'):
            path = os.path.join(model_path, f)
            with safe_open(path, framework='pt') as sf:
                for name in sf.keys():
                    if 'layer_scale' in name and 'code2wav' in name:
                        tensor = sf.get_tensor(name).float().numpy()
                        layer_scales[name] = tensor
                        print(f"{name}")
                        print(f"  shape: {tensor.shape}")
                        print(f"  mean:  {tensor.mean():.6f}")
                        print(f"  std:   {tensor.std():.6f}")
                        print(f"  min:   {tensor.min():.6f}")
                        print(f"  max:   {tensor.max():.6f}")

    if not layer_scales:
        print("No LayerScale weights found in HuggingFace model!")
        print("\nLet me search for any scale-related tensors...")
        for f in sorted(os.listdir(model_path)):
            if f.endswith('.safetensors'):
                path = os.path.join(model_path, f)
                with safe_open(path, framework='pt') as sf:
                    for name in sf.keys():
                        if 'code2wav' in name and ('scale' in name.lower() or 'ls' in name):
                            tensor = sf.get_tensor(name).float().numpy()
                            print(f"{name}: shape {tensor.shape}")

    # Check if we have pre-transformer layer debug dumps
    print("\n=== Checking C++ intermediate tensor dumps ===\n")

    cpp_tensors = sorted(cpp_dir.glob("pretrans_*.bin"))
    for t in cpp_tensors:
        size = t.stat().st_size // 4
        print(f"{t.name}: {size} floats")

    # Load and analyze layer 0 tensors
    print("\n=== Layer 0 Analysis ===\n")

    tensor_analysis = [
        ("pretrans_layer0_input", (n_embd, n_frames), "Input to layer 0"),
        ("pretrans_layer0_after_attn_norm", (n_embd, n_frames), "After attention RMSNorm"),
        ("pretrans_layer0_after_attn_res", (n_embd, n_frames), "After attention residual"),
        ("pretrans_before_output_norm", (n_embd, n_frames), "After all 8 layers"),
        ("after_pretrans", (n_embd, n_frames), "After output norm"),
    ]

    cpp_tensors_data = {}
    for name, shape, desc in tensor_analysis:
        path = cpp_dir / f"{name}.bin"
        if path.exists():
            try:
                arr = load_ggml_tensor(path, shape)
                cpp_tensors_data[name] = arr
                print(f"{desc} ({name}):")
                print(f"  shape: {arr.shape}")
                print(f"  mean:  {arr.mean():.6f}")
                print(f"  std:   {arr.std():.6f}")
                print(f"  min:   {arr.min():.6f}")
                print(f"  max:   {arr.max():.6f}")
            except Exception as e:
                print(f"{name}: Error loading - {e}")
        else:
            print(f"{name}: NOT FOUND")

    # Compare with HuggingFace
    print("\n=== Comparing with HuggingFace ===\n")

    # The HF tensor for pre-transformer output
    if (hf_dir / "03_pre_xfmr_out.npy").exists():
        hf_pretrans = load_hf_tensor(hf_dir / "03_pre_xfmr_out.npy")
        print(f"HF pre_xfmr_out shape: {hf_pretrans.shape}")
        print(f"HF pre_xfmr_out mean:  {hf_pretrans.mean():.6f}")
        print(f"HF pre_xfmr_out std:   {hf_pretrans.std():.6f}")

        if "after_pretrans" in cpp_tensors_data:
            cpp_pretrans = cpp_tensors_data["after_pretrans"]
            # Check if transpose needed
            if cpp_pretrans.shape != hf_pretrans.shape:
                if cpp_pretrans.T.shape == hf_pretrans.shape:
                    hf_pretrans = hf_pretrans.T
                    print("Transposed HF to match C++")

            print(f"\nC++ after_pretrans std: {cpp_pretrans.std():.6f}")
            print(f"HF pre_xfmr_out std:    {hf_pretrans.std():.6f}")
            print(f"Std ratio (C++/HF):     {cpp_pretrans.std() / hf_pretrans.std():.4f}")

            corr = np.corrcoef(cpp_pretrans.flatten(), hf_pretrans.flatten())[0, 1]
            print(f"Correlation:            {corr:.6f}")

    # Compute std ratio at each stage
    print("\n=== Std Ratio Progression Through Layers ===\n")

    # We need to know what HF tensors we have for intermediate stages
    print("Available HF tensors:")
    for t in sorted(hf_dir.glob("*.npy")):
        arr = np.load(t)
        print(f"  {t.name}: shape {arr.shape}, std {arr.std():.4f}")


if __name__ == "__main__":
    main()
