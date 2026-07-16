#!/usr/bin/env bash
set -uo pipefail
MODEL=~/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf
BIN=~/Projects/llama.cpp/build/bin
D=~/Projects/llama.cpp/experiments/profiles/ovf-bf16-gates
wait_mem() { for i in $(seq 1 80); do a1=$(free -g|awk '/^Mem:/{print $7}'); sleep 6; a2=$(free -g|awk '/^Mem:/{print $7}'); [[ $a1 -ge 100 && $a2 -ge 100 ]] && return 0; done; return 1; }
log() { echo "== $1 $(date +%T) ==" >> $D/probe2.log; sync; }
echo "== post-fix: identity + honest cost ==" > $D/probe2.log

log "identC2 OVF=0 fullpower"
env LLAMA_DSV4_FUSED_LID=1 \
  $BIN/llama-completion -m $MODEL -f $D/ident-prompt.txt -n 32 --temp 0 -s 42 \
  -c 69632 -ub 2048 -b 2048 -ngl 999 -fa on --no-mmap \
  -no-cnv --simple-io --no-display-prompt > $D/identC2.out 2> $D/identC2.log
log "identC2 exit=$?"
grep -E "prompt eval" $D/identC2.log | tail -1 >> $D/probe2.log
wait_mem

log "identD2 OVF=2048 fullpower"
env LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_CSA_TILE_OVF=2048 \
  $BIN/llama-completion -m $MODEL -f $D/ident-prompt.txt -n 32 --temp 0 -s 42 \
  -c 69632 -ub 2048 -b 2048 -ngl 999 -fa on --no-mmap \
  -no-cnv --simple-io --no-display-prompt > $D/identD2.out 2> $D/identD2.log
log "identD2 exit=$?"
grep -E "prompt eval" $D/identD2.log | tail -1 >> $D/probe2.log
if cmp -s $D/identC2.out $D/identD2.out; then log "IDENTITY: BYTE-IDENTICAL"; else
  log "IDENTITY: DIFFER"; { echo "--- C2:"; head -c 300 $D/identC2.out; echo; echo "--- D2:"; head -c 300 $D/identD2.out; echo; } >> $D/probe2.log
fi
wait_mem

log "costE OVF=2048 POWER=85 DEBUG pp2048@d65536"
env GGML_CUDA_POWER=85 GGML_CUDA_POWER_DEBUG=1 LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_CSA_TILE_OVF=2048 \
  $BIN/llama-bench -m $MODEL -fa on -ub 2048 -b 2048 -mmp 0 -r 1 -p 2048 -n 0 -d 65536 > $D/costE.log 2>&1
log "costE exit=$?"
grep -E "pp2048|cuda-power" $D/costE.log >> $D/probe2.log
log "ALL DONE"
