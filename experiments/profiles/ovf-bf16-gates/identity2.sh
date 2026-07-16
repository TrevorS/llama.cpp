#!/usr/bin/env bash
set -uo pipefail
MODEL=~/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf
BIN=~/Projects/llama.cpp/build/bin
D=~/Projects/llama.cpp/experiments/profiles/ovf-bf16-gates
WIKI=~/models/datasets/wiki.test.raw
wait_mem() { for i in $(seq 1 80); do a1=$(free -g|awk '/^Mem:/{print $7}'); sleep 6; a2=$(free -g|awk '/^Mem:/{print $7}'); [[ $a1 -ge 100 && $a2 -ge 100 ]] && return 0; done; return 1; }
log() { echo "== $1 $(date +%T) ==" >> $D/identity2.log; sync; }
echo "== round 2: full logs ==" > $D/identity2.log

log "legA2 ppl c65536 OVF=0"
env GGML_CUDA_POWER=85 LLAMA_DSV4_FUSED_LID=1 \
  $BIN/llama-perplexity -m $MODEL -f $WIKI -c 65536 --chunks 1 -fa on -ub 2048 -b 2048 --no-mmap > $D/legA2.log 2>&1
log "legA2 exit=$?"
tail -5 $D/legA2.log >> $D/identity2.log
wait_mem

log "legB2 ppl c65536 OVF=2048"
env GGML_CUDA_POWER=85 LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_CSA_TILE_OVF=2048 \
  $BIN/llama-perplexity -m $MODEL -f $WIKI -c 65536 --chunks 1 -fa on -ub 2048 -b 2048 --no-mmap > $D/legB2.log 2>&1
log "legB2 exit=$?"
tail -5 $D/legB2.log >> $D/identity2.log
wait_mem

log "legD repeat OVF=2048 pp2048@d65536"
env GGML_CUDA_POWER=85 LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_CSA_TILE_OVF=2048 \
  $BIN/llama-bench -m $MODEL -fa on -ub 2048 -b 2048 -mmp 0 -r 1 -p 2048 -n 0 -d 65536 > $D/legD.log 2>&1
log "legD exit=$?"
grep "pp2048" $D/legD.log >> $D/identity2.log
log "ALL DONE"
