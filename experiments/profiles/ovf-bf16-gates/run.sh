#!/usr/bin/env bash
set -uo pipefail
MODEL=~/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf
BIN=~/Projects/llama.cpp/build/bin
D=~/Projects/llama.cpp/experiments/profiles/ovf-bf16-gates
WIKI=~/models/datasets/wiki.test.raw
wait_mem() {
  for i in $(seq 1 80); do
    a1=$(free -g | awk '/^Mem:/{print $7}'); sleep 6
    a2=$(free -g | awk '/^Mem:/{print $7}')
    [[ $a1 -ge 100 && $a2 -ge 100 ]] && return 0
  done
  return 1
}
BASE="GGML_CUDA_POWER=85 LLAMA_DSV4_FUSED_LID=1"
log() { echo "== $1 $(date +%T) ==" >> $D/gates.log; sync; }
echo "== OVF + bf16 gate run ==" > $D/gates.log

log "leg1 OVF=0 pp2048@d65536 start"
env GGML_CUDA_POWER=85 LLAMA_DSV4_FUSED_LID=1 \
  $BIN/llama-bench -m $MODEL -fa on -ub 2048 -b 2048 -mmp 0 -r 1 -p 2048 -n 0 -d 65536 >> $D/gates.log 2>&1
log "leg1 exit=$?"; wait_mem

log "leg2 OVF=2048 pp2048@d65536 start"
env GGML_CUDA_POWER=85 LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_CSA_TILE_OVF=2048 \
  $BIN/llama-bench -m $MODEL -fa on -ub 2048 -b 2048 -mmp 0 -r 1 -p 2048 -n 0 -d 65536 >> $D/gates.log 2>&1
log "leg2 exit=$?"; wait_mem

log "leg3 OVF=2048+UNION_STATS audit start"
env GGML_CUDA_POWER=85 LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_CSA_TILE_OVF=2048 LLAMA_DSV4_UNION_STATS=1 \
  $BIN/llama-bench -m $MODEL -fa on -ub 2048 -b 2048 -mmp 0 -r 1 -p 2048 -n 0 -d 65536 > $D/stats-raw.log 2>&1
log "leg3 exit=$?"
grep -h "^US " $D/stats-raw.log | awk '{for(i=1;i<=NF;i++){if($i~/^max=/){sub("max=","",$i); if($i+0>mx)mx=$i+0} if($i~/^over=/){split($i,a,"="); split(a[2],b,"/"); ov+=b[1]; tot+=b[2]}}} END{print "US-SUMMARY max_union="mx" over_total="ov" of "tot}' >> $D/gates.log
wait_mem

log "leg4 ppl base c4096 start"
env GGML_CUDA_POWER=85 LLAMA_DSV4_FUSED_LID=1 \
  $BIN/llama-perplexity -m $MODEL -f $WIKI -c 4096 --chunks 8 -fa on -ub 2048 -b 2048 --no-mmap >> $D/gates.log 2>&1
log "leg4 exit=$?"; wait_mem

log "leg5 ppl BF16_RT c4096 start"
env GGML_CUDA_POWER=85 LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_KV_BF16_RT=1 \
  $BIN/llama-perplexity -m $MODEL -f $WIKI -c 4096 --chunks 8 -fa on -ub 2048 -b 2048 --no-mmap >> $D/gates.log 2>&1
log "leg5 exit=$?"
log "ALL GATES DONE"
