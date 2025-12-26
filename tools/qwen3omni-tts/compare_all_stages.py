#!/usr/bin/env python3
"""Compare ALL intermediate tensors between C++ and HuggingFace.

Uses correct GGML memory layout (Fortran order) for all comparisons.
"""

import numpy as np
from pathlib import Path
import sys


def load_ggml_tensor(path: Path, ne: tuple) -> np.ndarray:
    """Load GGML tensor dump with correct memory layout.

    GGML uses column-major (Fortran) order where ne[0] varies fastest.
    This function returns array with shape (ne[0], ne[1], ...) in numpy.
    """
    data = np.fromfile(path, dtype='<f4')
    expected_size = int(np.prod(ne))
    if len(data) != expected_size:
        raise ValueError(f"Size mismatch: got {len(data)}, expected {expected_size}")
    return data.reshape(ne, order='F')


def load_hf_tensor(path: Path) -> np.ndarray:
    """Load HuggingFace .npy tensor, removing batch dimension if present."""
    arr = np.load(path)
    if arr.ndim >= 3 and arr.shape[0] == 1:
        arr = arr[0]  # Remove batch dimension
    return arr


def compare_tensors(name: str, cpp: np.ndarray, hf: np.ndarray, transpose: bool = False) -> dict:
    """Compare two tensors and return statistics."""
    if transpose:
        cpp = cpp.T

    result = {
        'name': name,
        'cpp_shape': cpp.shape,
        'hf_shape': hf.shape,
    }

    if cpp.shape != hf.shape:
        # Try to find a permutation that matches
        if cpp.T.shape == hf.shape:
            cpp = cpp.T
            result['note'] = 'transposed'
            result['cpp_shape'] = cpp.shape
        elif len(cpp.shape) == 2 and cpp.shape[0] == hf.shape[1] and cpp.shape[1] == hf.shape[0]:
            cpp = cpp.T
            result['note'] = 'transposed'
            result['cpp_shape'] = cpp.shape
        else:
            result['status'] = 'shape_mismatch'
            return result

    # Flatten for correlation
    cpp_flat = cpp.flatten()
    hf_flat = hf.flatten()

    # Compute statistics
    diff = np.abs(cpp_flat - hf_flat)
    result['max_diff'] = float(diff.max())
    result['mean_diff'] = float(diff.mean())

    # Correlation (handle constant arrays)
    if np.std(cpp_flat) > 1e-10 and np.std(hf_flat) > 1e-10:
        result['correlation'] = float(np.corrcoef(cpp_flat, hf_flat)[0, 1])
    else:
        result['correlation'] = 1.0 if np.allclose(cpp_flat, hf_flat) else 0.0

    # Status based on correlation
    corr = result['correlation']
    if corr > 0.999:
        result['status'] = 'excellent'
    elif corr > 0.99:
        result['status'] = 'good'
    elif corr > 0.9:
        result['status'] = 'warning'
    else:
        result['status'] = 'diverged'

    return result


def get_tensor_mappings(n_frames: int) -> list:
    """Get tensor name/shape mappings for given number of frames.

    Returns list of (cpp_name, hf_name, cpp_ne) tuples.
    cpp_ne is the GGML shape [ne0, ne1] where ne0 is fastest varying.
    """
    seq = n_frames
    seq2 = seq * 2
    seq4 = seq * 4

    mappings = [
        # Pre-transformer output
        # C++: ne=[1024, seq] → [channels, seq]
        # HF: [channels, seq] after removing batch
        ('after_pretrans', '03_pre_xfmr_out', (1024, seq)),

        # ConvNeXt Block 0
        ('cnxt0_transconv_raw', '05_up0_transconv', (1024, seq2)),
        ('cnxt0_dwconv', '06_cnxt0_dwconv', (1024, seq2)),
        ('cnxt0_norm', '07_cnxt0_norm', (1024, seq2)),
        ('cnxt0_pw1', '08_cnxt0_pw1', (4096, seq2)),
        ('cnxt0_gelu', '09_cnxt0_gelu', (4096, seq2)),
        ('cnxt0_pw2', '10_cnxt0_pw2', (1024, seq2)),
        ('cnxt0_scale', '11_cnxt0_out', (1024, seq2)),  # After residual

        # ConvNeXt Block 1
        ('cnxt1_transconv_raw', '05_up1_transconv', (1024, seq4)),
        ('cnxt1_dwconv', '06_cnxt1_dwconv', (1024, seq4)),
        ('cnxt1_norm', '07_cnxt1_norm', (1024, seq4)),
        ('cnxt1_pw1', '08_cnxt1_pw1', (4096, seq4)),
        ('cnxt1_gelu', '09_cnxt1_gelu', (4096, seq4)),
        ('cnxt1_pw2', '10_cnxt1_pw2', (1024, seq4)),
        ('after_convnext', '11_cnxt1_out', (1024, seq4)),
    ]

    return mappings


def get_hifi_mappings(n_frames: int) -> list:
    """Get HiFi-GAN tensor mappings.

    HiFi-GAN has variable sequence lengths due to upsampling.
    We'll infer shapes from file sizes.
    """
    # HiFi-GAN upsamples: 64 → 512 → 2560 → 10240 → 30720 (for 16 frames)
    # But actual HF sizes differ slightly due to padding
    # We'll use file-size inference for these

    mappings = [
        # Stage 0 input (after pre-ups projection 1024→1536)
        ('hifi_stage0_before_snake', '12_dec0', None),  # Shape inferred
    ]
    return mappings


def infer_shape_from_size(n_floats: int, known_dim: int = None) -> tuple:
    """Infer 2D shape from total size."""
    if known_dim:
        other = n_floats // known_dim
        if known_dim * other == n_floats:
            return (known_dim, other)
    # Common channel sizes
    for ch in [1536, 1024, 768, 384, 192, 96]:
        if n_floats % ch == 0:
            return (ch, n_floats // ch)
    return None


def main():
    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage: compare_all_stages.py <cpp_dir> <hf_dir> [n_frames]")
        print("Example: compare_all_stages.py /models/debug/cpp_tokens_match /models/debug/hf_tensors_16f 16")
        sys.exit(1)

    cpp_dir = Path(sys.argv[1])
    hf_dir = Path(sys.argv[2])
    n_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 16

    if not cpp_dir.exists():
        print(f"Error: C++ directory not found: {cpp_dir}")
        sys.exit(1)
    if not hf_dir.exists():
        print(f"Error: HF directory not found: {hf_dir}")
        sys.exit(1)

    print(f"=== Comparing C++ vs HuggingFace ({n_frames} frames) ===\n")

    # Get mappings
    mappings = get_tensor_mappings(n_frames)

    # Track first divergence
    first_divergence = None
    results = []

    print("=== Pre-Transformer & ConvNeXt Stages ===\n")

    for cpp_name, hf_name, cpp_ne in mappings:
        cpp_path = cpp_dir / f"{cpp_name}.bin"
        hf_path = hf_dir / f"{hf_name}.npy"

        if not cpp_path.exists():
            print(f"  ⊘ {cpp_name}: C++ file not found")
            continue
        if not hf_path.exists():
            print(f"  ⊘ {cpp_name}: HF file not found ({hf_name})")
            continue

        try:
            cpp_tensor = load_ggml_tensor(cpp_path, cpp_ne)
            hf_tensor = load_hf_tensor(hf_path)

            result = compare_tensors(cpp_name, cpp_tensor, hf_tensor)
            results.append(result)

            # Format output
            status = result['status']
            if status == 'excellent':
                icon = '✓'
            elif status == 'good':
                icon = '○'
            elif status == 'warning':
                icon = '⚠'
            elif status == 'diverged':
                icon = '✗'
                if first_divergence is None:
                    first_divergence = result
            else:
                icon = '?'

            corr = result.get('correlation', 0)
            max_diff = result.get('max_diff', 0)
            note = f" ({result['note']})" if 'note' in result else ''

            print(f"  {icon} {cpp_name}: corr={corr:.6f}, max_diff={max_diff:.6f}{note}")

        except Exception as e:
            print(f"  ✗ {cpp_name}: Error - {e}")

    # HiFi-GAN comparisons (more complex due to varying sizes)
    print("\n=== HiFi-GAN Decoder Stages ===\n")

    hifi_stages = [
        ('hifi_stage0_before_snake', '12_dec0'),
        ('hifi_stage0_after_upsample', '12_dec1'),
        ('hifi_stage1_after_upsample', '12_dec2'),
        ('hifi_stage2_after_upsample', '12_dec3'),
        ('hifi_stage3_after_upsample', '12_dec4'),
        ('hifi_stage3_after_resblk', '12_dec5'),
        ('before_clamp', '12_dec6'),
    ]

    for cpp_name, hf_name in hifi_stages:
        cpp_path = cpp_dir / f"{cpp_name}.bin"
        hf_path = hf_dir / f"{hf_name}.npy"

        if not cpp_path.exists():
            print(f"  ⊘ {cpp_name}: C++ file not found")
            continue
        if not hf_path.exists():
            print(f"  ⊘ {cpp_name}: HF file not found ({hf_name})")
            continue

        try:
            # Load HF first to get shape
            hf_tensor = load_hf_tensor(hf_path)
            hf_shape = hf_tensor.shape

            # Infer C++ shape from HF shape (GGML is [channels, seq])
            cpp_size = cpp_path.stat().st_size // 4
            if len(hf_shape) == 2:
                # HF is [channels, seq], try that for GGML
                cpp_ne = (hf_shape[0], hf_shape[1])
            else:
                # Infer from size
                cpp_ne = infer_shape_from_size(cpp_size)

            if cpp_ne is None:
                print(f"  ? {cpp_name}: Cannot infer shape (size={cpp_size})")
                continue

            # Check if sizes match
            if cpp_size != int(np.prod(cpp_ne)):
                # Size mismatch - shapes are different
                print(f"  ⚠ {cpp_name}: Size mismatch - C++ has {cpp_size} floats, HF shape {hf_shape}")
                # Try to load what we can
                if cpp_size != np.prod(hf_shape):
                    # Find C++ shape
                    cpp_ne_inferred = infer_shape_from_size(cpp_size)
                    if cpp_ne_inferred:
                        print(f"      C++ shape likely: {cpp_ne_inferred}, HF shape: {hf_shape}")
                continue

            cpp_tensor = load_ggml_tensor(cpp_path, cpp_ne)
            result = compare_tensors(cpp_name, cpp_tensor, hf_tensor)
            results.append(result)

            status = result['status']
            if status == 'excellent':
                icon = '✓'
            elif status == 'good':
                icon = '○'
            elif status == 'warning':
                icon = '⚠'
            elif status == 'diverged':
                icon = '✗'
                if first_divergence is None:
                    first_divergence = result
            else:
                icon = '?'

            corr = result.get('correlation', 0)
            max_diff = result.get('max_diff', 0)
            note = f" ({result['note']})" if 'note' in result else ''

            print(f"  {icon} {cpp_name}: corr={corr:.6f}, max_diff={max_diff:.6f}{note}")

        except Exception as e:
            print(f"  ✗ {cpp_name}: Error - {e}")

    # Summary
    print("\n" + "=" * 60)
    print("=== Summary ===\n")

    excellent = sum(1 for r in results if r.get('status') == 'excellent')
    good = sum(1 for r in results if r.get('status') == 'good')
    warning = sum(1 for r in results if r.get('status') == 'warning')
    diverged = sum(1 for r in results if r.get('status') == 'diverged')

    print(f"  Excellent (corr > 0.999): {excellent}")
    print(f"  Good (corr > 0.99):       {good}")
    print(f"  Warning (corr > 0.9):     {warning}")
    print(f"  Diverged (corr < 0.9):    {diverged}")

    if first_divergence:
        print(f"\n>>> FIRST DIVERGENCE: {first_divergence['name']}")
        print(f"    Correlation: {first_divergence.get('correlation', 0):.6f}")
        print(f"    This is where debugging should focus.")


if __name__ == "__main__":
    main()
