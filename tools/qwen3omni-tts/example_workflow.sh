#!/bin/bash
# Example workflow for debugging Qwen3-Omni TTS pipeline

set -e

# Configuration
TEXT="Hello world"
HF_OUTPUT="/tmp/hf_tensors"
CPP_OUTPUT="/tmp/cpp_tensors"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3-Omni-30B-A3B-Instruct}"

echo "=== Qwen3-Omni TTS Debug Workflow ==="
echo

# Step 1: Extract HuggingFace reference tensors
echo "Step 1: Extracting HuggingFace reference tensors..."
python3 debug_hf_pipeline.py \
    --text "$TEXT" \
    --output "$HF_OUTPUT" \
    --model "$MODEL_PATH" \
    --max-codec-tokens 20

echo
echo "Step 2: Inspecting extracted tensors..."

# Show a few key tensors
echo "  - Thinker token embeddings:"
python3 load_tensor.py info "$HF_OUTPUT/02_thinker_tok_embd.bin"

echo
echo "  - Talker prefill embeddings:"
python3 load_tensor.py info "$HF_OUTPUT/15_talker_prefill_embeds.bin"

echo
echo "  - Code predictor output:"
python3 load_tensor.py info "$HF_OUTPUT/18_talker_codes.bin"

echo
echo "Step 3: Run llama.cpp implementation..."
echo "  (This is where you would run your C++ implementation)"
echo "  Example:"
echo "    ./build/bin/llama-tts --model model.gguf --text \"$TEXT\" --dump \"$CPP_OUTPUT\""

echo
echo "Step 4: Compare tensors..."
echo "  After running llama.cpp and saving tensors to $CPP_OUTPUT:"
echo "    python3 compare_tensors.py \"$HF_OUTPUT\" \"$CPP_OUTPUT\""

echo
echo "=== Alternative: Convert to .npy for other tools ==="
echo "python3 load_tensor.py convert \"$HF_OUTPUT\""

echo
echo "=== Listen to generated audio ==="
if [ -f "$HF_OUTPUT/output.wav" ]; then
    echo "HuggingFace audio saved to: $HF_OUTPUT/output.wav"
    echo "Play with: aplay $HF_OUTPUT/output.wav"
else
    echo "No audio file found (scipy may not be installed)"
fi

echo
echo "Done!"
