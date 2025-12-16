#!/usr/bin/env python3
"""Unit tests for QWEN3OMNIMOE GGUF constants."""

import sys
import os

# Support both container (/app/src) and host (repo root) paths
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repo_root, 'gguf-py'))
sys.path.insert(0, '/app/src/gguf-py')  # Container path

from gguf.constants import MODEL_ARCH, MODEL_TENSORS, MODEL_TENSOR, MODEL_ARCH_NAMES, Keys


def test_qwen3omnimoe_enum_exists():
    """Test that QWEN3OMNIMOE enum exists."""
    assert hasattr(MODEL_ARCH, 'QWEN3OMNIMOE'), 'QWEN3OMNIMOE not found in MODEL_ARCH'
    print(f'Test 1 PASSED: QWEN3OMNIMOE = {MODEL_ARCH.QWEN3OMNIMOE}')


def test_architecture_name_mapping():
    """Test that architecture name is mapped correctly."""
    assert MODEL_ARCH.QWEN3OMNIMOE in MODEL_ARCH_NAMES, 'QWEN3OMNIMOE not in MODEL_ARCH_NAMES'
    assert MODEL_ARCH_NAMES[MODEL_ARCH.QWEN3OMNIMOE] == 'qwen3omnimoe', 'Wrong name mapping'
    print(f'Test 2 PASSED: Name = {MODEL_ARCH_NAMES[MODEL_ARCH.QWEN3OMNIMOE]}')


def test_tensor_definitions_exist():
    """Test that tensor definitions exist."""
    assert MODEL_ARCH.QWEN3OMNIMOE in MODEL_TENSORS, 'QWEN3OMNIMOE not in MODEL_TENSORS'
    tensors = MODEL_TENSORS[MODEL_ARCH.QWEN3OMNIMOE]
    print(f'Test 3 PASSED: Tensor count = {len(tensors)}')


def test_required_tensors_present():
    """Test that all required tensors are present."""
    tensors = MODEL_TENSORS[MODEL_ARCH.QWEN3OMNIMOE]
    required_tensors = [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
    ]
    for t in required_tensors:
        assert t in tensors, f'{t} not in tensors'
    print(f'Test 4 PASSED: All {len(required_tensors)} required tensors present')


def test_mrope_key_exists():
    """Test that M-RoPE sections key exists."""
    assert hasattr(Keys.Rope, 'DIMENSION_SECTIONS'), 'DIMENSION_SECTIONS not in Keys.Rope'
    print(f'Test 5 PASSED: Key = {Keys.Rope.DIMENSION_SECTIONS}')


def test_talker_enum_exists():
    """Test that QWEN3OMNI_TALKER enum exists."""
    assert hasattr(MODEL_ARCH, 'QWEN3OMNI_TALKER'), 'QWEN3OMNI_TALKER not found in MODEL_ARCH'
    print(f'Test 6 PASSED: QWEN3OMNI_TALKER = {MODEL_ARCH.QWEN3OMNI_TALKER}')


def test_talker_name_mapping():
    """Test that QWEN3OMNI_TALKER name is mapped correctly."""
    assert MODEL_ARCH.QWEN3OMNI_TALKER in MODEL_ARCH_NAMES, 'QWEN3OMNI_TALKER not in MODEL_ARCH_NAMES'
    assert MODEL_ARCH_NAMES[MODEL_ARCH.QWEN3OMNI_TALKER] == 'qwen3omni-talker', 'Wrong name mapping'
    print(f'Test 7 PASSED: Name = {MODEL_ARCH_NAMES[MODEL_ARCH.QWEN3OMNI_TALKER]}')


def test_talker_tensor_definitions():
    """Test that QWEN3OMNI_TALKER tensor definitions exist."""
    assert MODEL_ARCH.QWEN3OMNI_TALKER in MODEL_TENSORS, 'QWEN3OMNI_TALKER not in MODEL_TENSORS'
    tensors = MODEL_TENSORS[MODEL_ARCH.QWEN3OMNI_TALKER]
    print(f'Test 8 PASSED: Talker tensor count = {len(tensors)}')


def test_talker_specific_tensors():
    """Test that Talker-specific tensors are present."""
    tensors = MODEL_TENSORS[MODEL_ARCH.QWEN3OMNI_TALKER]
    required_talker_tensors = [
        MODEL_TENSOR.TALKER_TEXT_PROJ_FC1,
        MODEL_TENSOR.TALKER_TEXT_PROJ_FC2,
        MODEL_TENSOR.TALKER_CODEC_HEAD,
        MODEL_TENSOR.TALKER_CODEC_EMBD,
    ]
    for t in required_talker_tensors:
        assert t in tensors, f'{t} not in Talker tensors'
    print(f'Test 9 PASSED: All {len(required_talker_tensors)} Talker-specific tensors present')


def test_talker_metadata_keys():
    """Test that Talker metadata keys exist."""
    assert hasattr(Keys, 'Talker'), 'Keys.Talker class not found'
    assert hasattr(Keys.Talker, 'ACCEPT_HIDDEN_LAYER'), 'ACCEPT_HIDDEN_LAYER key not found'
    assert hasattr(Keys.Talker, 'CODEC_VOCAB_SIZE'), 'CODEC_VOCAB_SIZE key not found'
    assert hasattr(Keys.Talker, 'NUM_CODEBOOKS'), 'NUM_CODEBOOKS key not found'
    print('Test 10 PASSED: Talker metadata keys exist')


# Code2Wav tests
def test_code2wav_metadata_keys():
    """Test that Code2Wav metadata keys exist."""
    assert hasattr(Keys, 'Code2Wav'), 'Keys.Code2Wav class not found'
    assert hasattr(Keys.Code2Wav, 'CODEBOOK_SIZE'), 'CODEBOOK_SIZE key not found'
    assert hasattr(Keys.Code2Wav, 'CODEBOOK_DIM'), 'CODEBOOK_DIM key not found'
    assert hasattr(Keys.Code2Wav, 'NUM_QUANTIZERS'), 'NUM_QUANTIZERS key not found'
    assert hasattr(Keys.Code2Wav, 'SEMANTIC_CODEBOOK_SIZE'), 'SEMANTIC_CODEBOOK_SIZE key not found'
    assert hasattr(Keys.Code2Wav, 'HIDDEN_SIZE'), 'HIDDEN_SIZE key not found'
    assert hasattr(Keys.Code2Wav, 'DECODER_DIM'), 'DECODER_DIM key not found'
    assert hasattr(Keys.Code2Wav, 'UPSAMPLE_RATES'), 'UPSAMPLE_RATES key not found'
    assert hasattr(Keys.Code2Wav, 'SLIDING_WINDOW'), 'SLIDING_WINDOW key not found'
    print('Test 11 PASSED: Code2Wav metadata keys exist')


def test_code2wav_tensor_enums():
    """Test that Code2Wav tensor enums exist."""
    c2w_tensors = [
        # Input embedding
        'C2W_CODE_EMBD',
        # Pre-transformer
        'C2W_PRE_ATTN_Q',
        'C2W_PRE_ATTN_K',
        'C2W_PRE_ATTN_V',
        'C2W_PRE_ATTN_OUT',
        'C2W_PRE_ATTN_NORM',
        'C2W_PRE_FFN_GATE',
        'C2W_PRE_FFN_UP',
        'C2W_PRE_FFN_DOWN',
        'C2W_PRE_FFN_NORM',
        'C2W_PRE_ATTN_SCALE',
        'C2W_PRE_FFN_SCALE',
        'C2W_PRE_OUTPUT_NORM',
        # Upsample blocks
        'C2W_UP_CONV',
        'C2W_UP_DWCONV',
        'C2W_UP_NORM',
        'C2W_UP_PWCONV1',
        'C2W_UP_PWCONV2',
        'C2W_UP_GAMMA',
        # HiFi-GAN decoder
        'C2W_DEC_CONV_IN',
        'C2W_DEC_SNAKE_ALPHA',
        'C2W_DEC_SNAKE_BETA',
        'C2W_DEC_BLK_SNAKE_A',
        'C2W_DEC_BLK_SNAKE_B',
        'C2W_DEC_BLK_CONV',
        'C2W_DEC_BLK_CONV1',
        'C2W_DEC_BLK_CONV2',
        'C2W_DEC_BLK_ACT1_A',
        'C2W_DEC_BLK_ACT1_B',
        'C2W_DEC_BLK_ACT2_A',
        'C2W_DEC_BLK_ACT2_B',
    ]
    for tensor_name in c2w_tensors:
        assert hasattr(MODEL_TENSOR, tensor_name), f'{tensor_name} not found in MODEL_TENSOR'
    print(f'Test 12 PASSED: All {len(c2w_tensors)} Code2Wav tensor enums exist')


def test_code2wav_tensor_names_mapping():
    """Test that Code2Wav tensors have string name mappings in TENSOR_NAMES."""
    from gguf.constants import TENSOR_NAMES

    c2w_tensors = [
        MODEL_TENSOR.C2W_CODE_EMBD,
        MODEL_TENSOR.C2W_PRE_ATTN_Q,
        MODEL_TENSOR.C2W_PRE_ATTN_K,
        MODEL_TENSOR.C2W_PRE_ATTN_V,
        MODEL_TENSOR.C2W_PRE_ATTN_OUT,
        MODEL_TENSOR.C2W_PRE_ATTN_NORM,
        MODEL_TENSOR.C2W_PRE_FFN_GATE,
        MODEL_TENSOR.C2W_PRE_FFN_UP,
        MODEL_TENSOR.C2W_PRE_FFN_DOWN,
        MODEL_TENSOR.C2W_PRE_FFN_NORM,
        MODEL_TENSOR.C2W_PRE_ATTN_SCALE,
        MODEL_TENSOR.C2W_PRE_FFN_SCALE,
        MODEL_TENSOR.C2W_PRE_OUTPUT_NORM,
        MODEL_TENSOR.C2W_UP_CONV,
        MODEL_TENSOR.C2W_UP_DWCONV,
        MODEL_TENSOR.C2W_UP_NORM,
        MODEL_TENSOR.C2W_UP_PWCONV1,
        MODEL_TENSOR.C2W_UP_PWCONV2,
        MODEL_TENSOR.C2W_UP_GAMMA,
        MODEL_TENSOR.C2W_DEC_CONV_IN,
        MODEL_TENSOR.C2W_DEC_SNAKE_ALPHA,
        MODEL_TENSOR.C2W_DEC_SNAKE_BETA,
        MODEL_TENSOR.C2W_DEC_BLK_SNAKE_A,
        MODEL_TENSOR.C2W_DEC_BLK_SNAKE_B,
        MODEL_TENSOR.C2W_DEC_BLK_CONV,
        MODEL_TENSOR.C2W_DEC_BLK_CONV1,
        MODEL_TENSOR.C2W_DEC_BLK_CONV2,
        MODEL_TENSOR.C2W_DEC_BLK_ACT1_A,
        MODEL_TENSOR.C2W_DEC_BLK_ACT1_B,
        MODEL_TENSOR.C2W_DEC_BLK_ACT2_A,
        MODEL_TENSOR.C2W_DEC_BLK_ACT2_B,
    ]
    for tensor in c2w_tensors:
        assert tensor in TENSOR_NAMES, f'{tensor} not in TENSOR_NAMES'
        assert TENSOR_NAMES[tensor].startswith('code2wav.'), f'{tensor} name should start with code2wav.'
    print('Test 13 PASSED: All Code2Wav tensor name mappings exist and have correct prefix')


def test_talker_tensor_mappings():
    """Test that Talker tensors have mappings in TensorNameMap."""
    from gguf.tensor_mapping import TensorNameMap

    # Create a TensorNameMap for QWEN3OMNI_TALKER
    tensor_map = TensorNameMap(MODEL_ARCH.QWEN3OMNI_TALKER, n_blocks=20)

    # Check that Talker-specific HF tensor names map correctly
    # Using try_suffixes as done in converter
    talker_hf_tensors = [
        "talker.text_projection.linear_fc1.weight",  # MLP fc1
        "talker.text_projection.linear_fc2.weight",  # MLP fc2
        "talker.codec_head.weight",
        "talker.codec_embeddings.0.weight",
        "talker.codec_embeddings.15.weight",  # Test last codebook
    ]
    for hf_name in talker_hf_tensors:
        mapped = tensor_map.get_name(hf_name, try_suffixes=(".weight", ".bias"))
        assert mapped is not None, f'{hf_name} not found in tensor mapping'
    print(f'Test 14 PASSED: All {len(talker_hf_tensors)} Talker HF tensors map correctly')


def test_code2wav_tensor_mappings():
    """Test that Code2Wav tensors have mappings in TensorNameMap."""
    from gguf.tensor_mapping import TensorNameMap

    # Create a TensorNameMap for QWEN3OMNI_TALKER (Code2Wav is bundled with Talker)
    tensor_map = TensorNameMap(MODEL_ARCH.QWEN3OMNI_TALKER, n_blocks=20)

    # Check Code2Wav HF tensor names that can be mapped via TensorNameMap
    # Note: decoder.{bid}.block.{xid}.* tensors are handled specially in converter
    c2w_hf_tensors = [
        # Input embedding
        "code2wav.code_embedding.weight",
        # Pre-transformer (8 layers)
        "code2wav.pre_transformer.layers.0.self_attn.q_proj.weight",
        "code2wav.pre_transformer.layers.0.self_attn.k_proj.weight",
        "code2wav.pre_transformer.layers.0.self_attn.v_proj.weight",
        "code2wav.pre_transformer.layers.0.self_attn.o_proj.weight",
        "code2wav.pre_transformer.layers.0.input_layernorm.weight",
        "code2wav.pre_transformer.layers.0.mlp.gate_proj.weight",
        "code2wav.pre_transformer.layers.0.mlp.up_proj.weight",
        "code2wav.pre_transformer.layers.0.mlp.down_proj.weight",
        "code2wav.pre_transformer.layers.0.post_attention_layernorm.weight",
        "code2wav.pre_transformer.layers.0.self_attn_layer_scale.scale",
        "code2wav.pre_transformer.layers.0.mlp_layer_scale.scale",
        "code2wav.pre_transformer.norm.weight",
        # Upsample blocks (ConvNeXt style)
        "code2wav.upsample.0.0.conv.weight",
        "code2wav.upsample.0.1.dwconv.conv.weight",
        "code2wav.upsample.0.1.norm.weight",
        "code2wav.upsample.0.1.pwconv1.weight",
        "code2wav.upsample.0.1.pwconv2.weight",
        "code2wav.upsample.0.1.gamma",
        # HiFi-GAN decoder (non-nested tensors only)
        "code2wav.decoder.0.conv.weight",  # Initial conv
        "code2wav.decoder.1.alpha",  # Snake activation
        "code2wav.decoder.1.beta",
    ]
    for hf_name in c2w_hf_tensors:
        mapped = tensor_map.get_name(hf_name, try_suffixes=(".weight", ".bias", ".scale", ""))
        assert mapped is not None, f'{hf_name} not found in tensor mapping'
    print(f'Test 15 PASSED: All {len(c2w_hf_tensors)} Code2Wav HF tensors map correctly')
    # Note: decoder block tensors (decoder.{bid}.block.{xid}.*) are handled by converter special case


def test_code_predictor_tensor_enums():
    """Test that code_predictor tensor enums exist."""
    cp_tensors = [
        'TALKER_CP_CODEC_EMBD',
        'TALKER_CP_ATTN_Q',
        'TALKER_CP_ATTN_K',
        'TALKER_CP_ATTN_V',
        'TALKER_CP_ATTN_OUT',
        'TALKER_CP_ATTN_Q_NORM',
        'TALKER_CP_ATTN_K_NORM',
        'TALKER_CP_ATTN_NORM',
        'TALKER_CP_FFN_GATE',
        'TALKER_CP_FFN_UP',
        'TALKER_CP_FFN_DOWN',
        'TALKER_CP_FFN_NORM',
        'TALKER_CP_OUTPUT_NORM',
        'TALKER_CP_LM_HEAD',
    ]
    for tensor_name in cp_tensors:
        assert hasattr(MODEL_TENSOR, tensor_name), f'{tensor_name} not found in MODEL_TENSOR'
    print(f'Test 16 PASSED: All {len(cp_tensors)} code_predictor tensor enums exist')


def test_code_predictor_tensor_mappings():
    """Test that code_predictor tensors have mappings in TensorNameMap."""
    from gguf.tensor_mapping import TensorNameMap

    # Create a TensorNameMap for QWEN3OMNI_TALKER
    tensor_map = TensorNameMap(MODEL_ARCH.QWEN3OMNI_TALKER, n_blocks=20)

    # Check code_predictor HF tensor names map correctly
    cp_hf_tensors = [
        "talker.code_predictor.model.codec_embedding.0.weight",  # First codec embedding
        "talker.code_predictor.model.codec_embedding.14.weight", # Last codec embedding
        "talker.code_predictor.model.layers.0.self_attn.q_proj.weight",
        "talker.code_predictor.model.layers.0.self_attn.k_proj.weight",
        "talker.code_predictor.model.layers.0.self_attn.v_proj.weight",
        "talker.code_predictor.model.layers.0.self_attn.o_proj.weight",
        "talker.code_predictor.model.layers.0.self_attn.q_norm.weight",
        "talker.code_predictor.model.layers.0.self_attn.k_norm.weight",
        "talker.code_predictor.model.layers.0.input_layernorm.weight",
        "talker.code_predictor.model.layers.0.mlp.gate_proj.weight",
        "talker.code_predictor.model.layers.0.mlp.up_proj.weight",
        "talker.code_predictor.model.layers.0.mlp.down_proj.weight",
        "talker.code_predictor.model.layers.0.post_attention_layernorm.weight",
        "talker.code_predictor.model.layers.4.self_attn.q_proj.weight", # Last layer (5 layers)
        "talker.code_predictor.model.norm.weight",  # Output norm
        "talker.code_predictor.lm_head.0.weight",   # First lm head
        "talker.code_predictor.lm_head.14.weight",  # Last lm head
    ]
    for hf_name in cp_hf_tensors:
        mapped = tensor_map.get_name(hf_name, try_suffixes=(".weight", ".bias"))
        assert mapped is not None, f'{hf_name} not found in tensor mapping'
    print(f'Test 17 PASSED: All {len(cp_hf_tensors)} code_predictor HF tensors map correctly')


if __name__ == '__main__':
    # QWEN3OMNIMOE (Thinker) tests
    test_qwen3omnimoe_enum_exists()
    test_architecture_name_mapping()
    test_tensor_definitions_exist()
    test_required_tensors_present()
    test_mrope_key_exists()

    # QWEN3OMNI_TALKER tests
    test_talker_enum_exists()
    test_talker_name_mapping()
    test_talker_tensor_definitions()
    test_talker_specific_tensors()
    test_talker_metadata_keys()

    # Code2Wav tests
    test_code2wav_metadata_keys()
    test_code2wav_tensor_enums()
    test_code2wav_tensor_names_mapping()

    # Tensor mapping tests
    test_talker_tensor_mappings()
    test_code2wav_tensor_mappings()

    # Code_predictor tests
    test_code_predictor_tensor_enums()
    test_code_predictor_tensor_mappings()

    print('\nAll tests passed!')
