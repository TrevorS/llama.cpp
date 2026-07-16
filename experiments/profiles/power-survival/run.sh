#!/usr/bin/env bash
set -uo pipefail
MODEL=~/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf
BIN=~/Projects/llama.cpp/build/bin
D=~/Projects/llama.cpp/experiments/profiles/power-survival
snap() {
  { echo "--- $1 $(date +%T)";
    nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader 2>/dev/null;
    nvidia-smi -q -d PERFORMANCE 2>/dev/null | grep -iE "sw power cap|hw slowdown|sw thermal|hw thermal|power brake" ;
  } >> $D/clocks.log
}
wait_mem() {
  for i in $(seq 1 80); do
    a1=$(free -g | awk '/^Mem:/{print $7}'); sleep 6
    a2=$(free -g | awk '/^Mem:/{print $7}')
    [[ $a1 -ge 100 && $a2 -ge 100 ]] && return 0
  done
  return 1
}
echo "== survival test: 2x pp2048@d131072, GGML_CUDA_POWER=85 ==" > $D/result.log
sync
snap pre-leg1
echo "== leg 1 start $(date +%T) ==" >> $D/result.log
env GGML_CUDA_POWER=85 LLAMA_DSV4_FUSED_LID=1 $BIN/llama-bench -m $MODEL \
  -fa on -ub 2048 -b 2048 -mmp 0 -r 1 -p 2048 -n 0 -d 131072 >> $D/result.log 2>&1
echo "== leg 1 exit=$? $(date +%T) ==" >> $D/result.log
sync
snap post-leg1
wait_mem || echo "== mem recovery timeout ==" >> $D/result.log
snap pre-leg2
echo "== leg 2 start $(date +%T) ==" >> $D/result.log
env GGML_CUDA_POWER=85 LLAMA_DSV4_FUSED_LID=1 $BIN/llama-bench -m $MODEL \
  -fa on -ub 2048 -b 2048 -mmp 0 -r 1 -p 2048 -n 0 -d 131072 >> $D/result.log 2>&1
echo "== leg 2 exit=$? $(date +%T) ==" >> $D/result.log
sync
snap post-leg2
echo "== SURVIVED BOTH LEGS ==" >> $D/result.log
sync
