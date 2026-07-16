#!/usr/bin/env bash
set -uo pipefail
MODEL=~/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf
BIN=~/Projects/llama.cpp/build/bin
D=~/Projects/llama.cpp/experiments/profiles/fp4mma-ab
snap() { { echo "--- $1 $(date +%T)"; nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader; nvidia-smi -q -d PERFORMANCE 2>/dev/null | grep -iE "sw power cap$|power brake|thermal slowdown +:"; } >> $D/d131k-clocks.log; }
wait_mem() {
  for i in $(seq 1 80); do
    a1=$(free -g | awk '/^Mem:/{print $7}'); sleep 6
    a2=$(free -g | awk '/^Mem:/{print $7}')
    [[ $a1 -ge 100 && $a2 -ge 100 ]] && return 0
  done
  return 1
}
COMMON="GGML_CUDA_POWER=85 LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_LID_CACHE_MXFP4=1"
echo "== fp4-mma d131k A/B (both arms CACHE_MXFP4=1, POWER=85) ==" > $D/d131k.log
snap pre-armA
echo "== arm A: int8 (FP4_MMA off) start $(date +%T) ==" >> $D/d131k.log
env GGML_CUDA_POWER=85 LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_LID_CACHE_MXFP4=1 \
  $BIN/llama-bench -m $MODEL -fa on -ub 2048 -b 2048 -mmp 0 -r 1 -p 2048 -n 0 -d 131072 >> $D/d131k.log 2>&1
echo "== arm A exit=$? $(date +%T) ==" >> $D/d131k.log
sync; snap post-armA
wait_mem || echo "== mem recovery timeout ==" >> $D/d131k.log
snap pre-armB
echo "== arm B: LLAMA_DSV4_LID_FP4_MMA=1 start $(date +%T) ==" >> $D/d131k.log
env GGML_CUDA_POWER=85 LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_LID_CACHE_MXFP4=1 LLAMA_DSV4_LID_FP4_MMA=1 \
  $BIN/llama-bench -m $MODEL -fa on -ub 2048 -b 2048 -mmp 0 -r 1 -p 2048 -n 0 -d 131072 >> $D/d131k.log 2>&1
echo "== arm B exit=$? $(date +%T) ==" >> $D/d131k.log
sync; snap post-armB
echo "== AB COMPLETE ==" >> $D/d131k.log
sync
