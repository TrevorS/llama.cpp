#!/usr/bin/env python3
"""
Utility functions for loading tensor dumps from debug_hf_pipeline.py

The binary format is:
    [num_dims: uint32] [dim0: uint32] [dim1: uint32] ... [data: float32 array]
"""

import struct
from pathlib import Path
from typing import Tuple

import numpy as np


def load_tensor_bin(filepath: str) -> Tuple[np.ndarray, list]:
    """
    Load a tensor from binary file created by debug_hf_pipeline.py.

    Args:
        filepath: Path to .bin file

    Returns:
        (tensor as numpy array, shape as list)
    """
    filepath = Path(filepath)

    with open(filepath, 'rb') as f:
        # Read number of dimensions
        num_dims = struct.unpack('<I', f.read(4))[0]

        # Read each dimension
        shape = []
        for _ in range(num_dims):
            dim = struct.unpack('<I', f.read(4))[0]
            shape.append(dim)

        # Read data
        num_elements = int(np.prod(shape))
        data = np.frombuffer(f.read(num_elements * 4), dtype='<f4')

        # Reshape
        tensor = data.reshape(shape)

    return tensor, shape


def print_tensor_info(filepath: str) -> None:
    """
    Print information about a tensor file.

    Args:
        filepath: Path to .bin file
    """
    tensor, shape = load_tensor_bin(filepath)

    print(f"File: {filepath}")
    print(f"  Shape: {shape}")
    print(f"  DType: {tensor.dtype}")
    print(f"  Size: {tensor.size} elements ({tensor.nbytes} bytes)")
    print(f"  Range: [{tensor.min():.6f}, {tensor.max():.6f}]")
    print(f"  Mean: {tensor.mean():.6f}")
    print(f"  Std: {tensor.std():.6f}")


def convert_bin_to_npy(bin_path: str, npy_path: str = None) -> None:
    """
    Convert a .bin tensor to .npy format for compatibility with existing tools.

    Args:
        bin_path: Input .bin file
        npy_path: Output .npy file (defaults to same name with .npy extension)
    """
    bin_path = Path(bin_path)

    if npy_path is None:
        npy_path = bin_path.with_suffix('.npy')
    else:
        npy_path = Path(npy_path)

    tensor, _ = load_tensor_bin(bin_path)
    np.save(npy_path, tensor)

    print(f"Converted {bin_path} -> {npy_path}")


def convert_directory(input_dir: str, output_dir: str = None) -> None:
    """
    Convert all .bin files in a directory to .npy format.

    Args:
        input_dir: Directory containing .bin files
        output_dir: Output directory (defaults to same directory)
    """
    input_dir = Path(input_dir)

    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    bin_files = list(input_dir.glob('*.bin'))
    print(f"Found {len(bin_files)} .bin files in {input_dir}")

    for bin_file in bin_files:
        npy_file = output_dir / bin_file.with_suffix('.npy').name
        convert_bin_to_npy(bin_file, npy_file)

    print(f"\nConverted {len(bin_files)} files")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and inspect tensor files from debug_hf_pipeline.py"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Info command
    info_parser = subparsers.add_parser('info', help='Print tensor information')
    info_parser.add_argument('file', help='Path to .bin file')

    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert .bin to .npy')
    convert_parser.add_argument('input', help='Input .bin file or directory')
    convert_parser.add_argument('--output', '-o', help='Output .npy file or directory')

    args = parser.parse_args()

    if args.command == 'info':
        print_tensor_info(args.file)

    elif args.command == 'convert':
        input_path = Path(args.input)

        if input_path.is_dir():
            convert_directory(args.input, args.output)
        else:
            convert_bin_to_npy(args.input, args.output)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
