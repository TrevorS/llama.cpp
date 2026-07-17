#!/usr/bin/env bash
# DS4-Flash serving: 64k ctx + MTP spec-decode + refusal steering, LAN-exposed.
set -u
cd /home/trevor/Projects/llama.cpp
MODEL=~/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf
DRAFT=~/models/ds4/DeepSeek-V4-Flash-MTP-mxfp4.gguf
CVEC=~/Projects/ds4-refusal/llamacpp-iq3/weights/native_nothink.gguf
SCALEFILE=~/Projects/ds4-refusal/llamacpp-iq3/cvec_scale.txt
LOG=~/.cache/llama-server-refusal.log

exec env \
  LLAMA_DSV4_LID_CACHE_MXFP4=1 \
  GGML_CUDA_POWER_ADAPT=1 \
  LLAMA_CVEC_ABLATE=1 LLAMA_CVEC_AT_FFN=1 LLAMA_CVEC_FFN_ONLY=1 \
  LLAMA_CVEC_SCALE_FILE=$SCALEFILE \
  ./build/bin/llama-server -m $MODEL \
    -md $DRAFT --spec-type draft-mtp --spec-draft-n-max 2 -ngld 999 \
    --control-vector-scaled $CVEC:1.0 \
    -c 65536 -fa on --no-mmap -ub 2048 -b 2048 -ngl 999 \
    -np 1 --kv-unified --jinja \
    --cache-ram 2048 -ctxcp 2 \
    --host 0.0.0.0 --port 8080 \
    2>&1 | tee $LOG
