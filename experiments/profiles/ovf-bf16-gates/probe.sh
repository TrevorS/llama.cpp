#!/usr/bin/env bash
set -uo pipefail
MODEL=~/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf
BIN=~/Projects/llama.cpp/build/bin
D=~/Projects/llama.cpp/experiments/profiles/ovf-bf16-gates
wait_mem() { for i in $(seq 1 80); do a1=$(free -g|awk '/^Mem:/{print $7}'); sleep 6; a2=$(free -g|awk '/^Mem:/{print $7}'); [[ $a1 -ge 100 && $a2 -ge 100 ]] && return 0; done; return 1; }
log() { echo "== $1 $(date +%T) ==" >> $D/probe.log; sync; }
echo "== governor probe + identity ==" > $D/probe.log

log "probeA OVF=0 POWER=85 DEBUG pp2048@d65536"
env GGML_CUDA_POWER=85 GGML_CUDA_POWER_DEBUG=1 LLAMA_DSV4_FUSED_LID=1 \
  $BIN/llama-bench -m $MODEL -fa on -ub 2048 -b 2048 -mmp 0 -r 1 -p 2048 -n 0 -d 65536 > $D/probeA.log 2>&1
log "probeA exit=$?"
grep -E "pp2048|cuda-power" $D/probeA.log >> $D/probe.log
wait_mem

log "probeB OVF=2048 POWER=85 DEBUG pp2048@d65536"
env GGML_CUDA_POWER=85 GGML_CUDA_POWER_DEBUG=1 LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_CSA_TILE_OVF=2048 \
  $BIN/llama-bench -m $MODEL -fa on -ub 2048 -b 2048 -mmp 0 -r 1 -p 2048 -n 0 -d 65536 > $D/probeB.log 2>&1
log "probeB exit=$?"
grep -E "pp2048|cuda-power" $D/probeB.log >> $D/probe.log
wait_mem

python3 - $D/ident-prompt.txt <<'PYEOF'
import sys
junk = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
with open(sys.argv[1], "w") as f:
    f.write("Below is a long passage. Read it carefully and then continue it naturally.\n")
    f.write(junk*2900)
    f.write("\nContinue the passage:")
PYEOF

log "probeC identity OVF=0 fullpower c69632"
env LLAMA_DSV4_FUSED_LID=1 \
  $BIN/llama-completion -m $MODEL -f $D/ident-prompt.txt -n 32 --temp 0 -s 42 \
  -c 69632 -ub 2048 -b 2048 -ngl 999 -fa on --no-mmap \
  -no-cnv --simple-io --no-display-prompt > $D/identC.out 2> $D/identC.log
log "probeC exit=$?"
grep -E "prompt eval|eval time" $D/identC.log | tail -3 >> $D/probe.log
wait_mem

log "probeD identity OVF=2048 fullpower c69632"
env LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_CSA_TILE_OVF=2048 \
  $BIN/llama-completion -m $MODEL -f $D/ident-prompt.txt -n 32 --temp 0 -s 42 \
  -c 69632 -ub 2048 -b 2048 -ngl 999 -fa on --no-mmap \
  -no-cnv --simple-io --no-display-prompt > $D/identD.out 2> $D/identD.log
log "probeD exit=$?"
grep -E "prompt eval|eval time" $D/identD.log | tail -3 >> $D/probe.log

if cmp -s $D/identC.out $D/identD.out; then
  log "IDENTITY: BYTE-IDENTICAL"
else
  log "IDENTITY: DIFFER"
  diff <(head -c 400 $D/identC.out) <(head -c 400 $D/identD.out) >> $D/probe.log 2>&1 || true
fi
log "ALL PROBES DONE"
