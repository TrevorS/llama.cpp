#!/usr/bin/env python3
"""Quick tensor comparison for debugging.

IMPORTANT: GGML uses column-major (Fortran) memory layout!
For a tensor with ne[0]=A, ne[1]=B, the memory is laid out with
ne[0] being contiguous. To read into numpy correctly:
  arr = np.fromfile(...).reshape(B, A).T  # gives [A, B] in numpy
"""

import numpy as np
from pathlib import Path

hf_dir = Path('/models/debug/hf_tensors')
cpp_dir = Path('/models/debug/cpp_tensors_new')

# Mapping from HF names to (C++ name, expected GGML shape [ne0, ne1])
# HF shapes are [batch, channels, seq] = [1, C, S]
# C++ shapes are [ne0, ne1] in GGML notation, memory is column-major
name_map = {
    # After transconv: HF [1, 1024, 40], GGML [1024, 40] (20 input frames -> 40)
    '05_up0_transconv': ('cnxt0_transconv_raw', (1024, 40)),
    '05_up1_transconv': ('cnxt1_transconv_raw', (1024, 80)),
    # After dwconv: HF [1, 1024, 40], GGML [1024, 40]
    '06_cnxt0_dwconv': ('cnxt0_dwconv', (1024, 40)),
    '06_cnxt1_dwconv': ('cnxt1_dwconv', (1024, 80)),
    # After norm: HF [1, 1024, 40], GGML [1024, 40]
    '07_cnxt0_norm': ('cnxt0_norm', (1024, 40)),
    '07_cnxt1_norm': ('cnxt1_norm', (1024, 80)),
    # After pw1: HF [1, 4096, 40], GGML [4096, 40]
    '08_cnxt0_pw1': ('cnxt0_pw1', (4096, 40)),
    '08_cnxt1_pw1': ('cnxt1_pw1', (4096, 80)),
    # After gelu: same as pw1
    '09_cnxt0_gelu': ('cnxt0_gelu', (4096, 40)),
    '09_cnxt1_gelu': ('cnxt1_gelu', (4096, 80)),
    # After pw2: HF [1, 1024, 40], GGML [1024, 40]
    '10_cnxt0_pw2': ('cnxt0_pw2', (1024, 40)),
    '10_cnxt1_pw2': ('cnxt1_pw2', (1024, 80)),
}


def load_ggml_tensor(path, shape):
    """Load GGML tensor with correct memory layout.

    GGML stores tensors in column-major order with ne[0] contiguous.
    For shape (ne0, ne1), we read as reshape(ne1, ne0).T
    """
    arr = np.fromfile(str(path), dtype=np.float32)
    ne0, ne1 = shape
    expected_size = ne0 * ne1
    if len(arr) != expected_size:
        return None, f"Size mismatch: expected {expected_size}, got {len(arr)}"
    # Read column-major: reshape to (ne1, ne0) then transpose to get (ne0, ne1)
    arr_2d = arr.reshape(ne1, ne0).T
    return arr_2d, None


print('Tensor Comparison: HuggingFace vs GGML (C++)\n')
print('{:<25} {:<15} {:<15} {:>8} {}'.format('Name', 'HF Shape', 'CPP Shape', 'Corr', 'Status'))
print('=' * 80)

for hf_name, (cpp_name, ggml_shape) in name_map.items():
    hf_file = hf_dir / f'{hf_name}.npy'
    cpp_file = cpp_dir / f'{cpp_name}.bin'

    if not hf_file.exists():
        print(f'{hf_name:<25} HF FILE MISSING')
        continue
    if not cpp_file.exists():
        print(f'{hf_name:<25} CPP FILE MISSING')
        continue

    hf_arr = np.load(str(hf_file))
    cpp_arr, err = load_ggml_tensor(cpp_file, ggml_shape)

    if err:
        print(f'{hf_name:<25} {err}')
        continue

    # HF is [batch, channels, seq], C++ is [channels, seq]
    # Need to reshape HF to match for comparison
    hf_flat = hf_arr.flatten()

    # For C++, flatten in row-major (C order) after correct reshape
    cpp_flat = cpp_arr.flatten()

    # Handle size mismatch
    if len(hf_flat) != len(cpp_flat):
        print(f'{hf_name:<25} SIZE MISMATCH: HF={len(hf_flat)}, CPP={len(cpp_flat)}')
        continue

    # Correlation
    if np.std(hf_flat) > 1e-10 and np.std(cpp_flat) > 1e-10:
        corr = float(np.corrcoef(hf_flat, cpp_flat)[0, 1])
    else:
        corr = 0.0

    match = 'OK' if corr > 0.9 else 'FAIL'

    print('{:<25} {:<15} {:<15} {:>8.4f} {}'.format(
        hf_name, str(hf_arr.shape), str(cpp_arr.shape), corr, match))

print()
