#!/usr/bin/env python3
"""
Extract Thinker embeddings from Qwen3-Omni for Talker testing.

This script generates the hidden state embeddings that the Talker expects as input.
The embeddings are saved as a binary file that can be loaded by the C++ test tool.

Usage:
    python extract_embeddings.py --text "Hello, this is a test." --output test_embeddings.bin
    python extract_embeddings.py --text "Hello" --output test.bin --model /path/to/model

Requirements:
    pip install torch transformers
"""

import argparse
import struct
import sys
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: Required packages not found. Install with:")
    print("  pip install torch transformers")
    sys.exit(1)


def extract_embeddings(
    text: str,
    output_path: str,
    model_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    accept_hidden_layer: int = 6,
    device: str = "auto",
) -> None:
    """
    Extract Thinker embeddings and save to binary file.

    Args:
        text: Input text to convert to speech
        output_path: Path to save embeddings binary
        model_path: HuggingFace model path or local directory
        accept_hidden_layer: Which hidden layer to extract (default: 6)
        device: Device to use ("auto", "cuda", "cpu")
    """
    print(f"Loading model from {model_path}...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    print(f"Tokenizing text: {text!r}")
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    print(f"Token count: {inputs['input_ids'].shape[1]}")

    print(f"Extracting embeddings from layer {accept_hidden_layer}...")
    with torch.no_grad():
        # Get hidden states from Thinker
        outputs = model.model.thinker(
            **inputs,
            output_hidden_states=True,
        )

        # Get the specified hidden layer
        hidden = outputs.hidden_states[accept_hidden_layer]

    # Convert to float32 for compatibility
    hidden = hidden.float().cpu()
    print(f"Embedding shape: {hidden.shape}")  # [batch, seq_len, hidden_dim]

    # Save as binary file
    # Format: [seq_len: u32] [hidden_dim: u32] [data: f32 array]
    seq_len = hidden.shape[1]
    hidden_dim = hidden.shape[2]
    data = hidden[0].numpy()  # Remove batch dim

    with open(output_path, "wb") as f:
        f.write(struct.pack("<II", seq_len, hidden_dim))
        f.write(data.astype("<f4").tobytes())

    print(f"Saved embeddings to {output_path}")
    print(f"  Shape: [{seq_len}, {hidden_dim}]")
    print(f"  Size: {Path(output_path).stat().st_size} bytes")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Thinker embeddings for Talker testing"
    )
    parser.add_argument(
        "--text", "-t", required=True, help="Input text to convert to speech"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output path for embeddings binary"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Model path (HuggingFace or local)",
    )
    parser.add_argument(
        "--layer",
        "-l",
        type=int,
        default=6,
        help="Hidden layer to extract (default: 6)",
    )
    parser.add_argument(
        "--device", "-d", default="auto", help="Device (auto, cuda, cpu)"
    )

    args = parser.parse_args()

    extract_embeddings(
        text=args.text,
        output_path=args.output,
        model_path=args.model,
        accept_hidden_layer=args.layer,
        device=args.device,
    )


if __name__ == "__main__":
    main()
