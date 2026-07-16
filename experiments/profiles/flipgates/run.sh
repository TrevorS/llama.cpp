#!/usr/bin/env bash
set -uo pipefail
MODEL=~/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf
BIN=~/Projects/llama.cpp/build/bin
D=~/Projects/llama.cpp/experiments/profiles/flipgates
WIKI=~/models/datasets/wiki.test.raw
KEY=71432
wait_mem() { for i in $(seq 1 80); do a1=$(free -g|awk '/^Mem:/{print $7}'); sleep 6; a2=$(free -g|awk '/^Mem:/{print $7}'); [[ $a1 -ge 100 && $a2 -ge 100 ]] && return 0; done; return 1; }
log() { echo "== $1 $(date +%T) ==" >> $D/results.log; sync; }
mkpk() { # mkpk <out> <pos%> <n_repeats>
python3 - "$1" "$2" "$3" <<'PYEOF'
import sys
out, pos, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
junk = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
needle = " The pass key is 71432. Remember it. 71432 is the pass key. "
k = max(1, n*pos//100)
with open(out, "w") as f:
    f.write("There is a pass key hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about it.\n")
    f.write(junk*k + needle + junk*(n-k))
    f.write("\nWhat is the pass key? The pass key is")
PYEOF
}
run_completion() { # run_completion <envstr...> -- <promptfile> <ctx> <out> <errlog>
  local envs=(); while [[ "$1" != "--" ]]; do envs+=("$1"); shift; done; shift
  env "${envs[@]}" $BIN/llama-completion -m $MODEL -f "$1" -n 12 --temp 0 -s 42 \
    -c "$2" -ub 2048 -b 2048 -ngl 999 -fa on --no-mmap \
    -no-cnv --simple-io --no-display-prompt > "$3" 2> "$4"
}
echo "== flip gates: fp4-mma (item1) + OVF passkey@131k (item2) ==" > $D/results.log

# ---------------- item 1: fp4-mma PPL/coherence/passkey/determinism ----------------
log "1a ppl c4096 container+int8"
env LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_LID_CACHE_MXFP4=1 \
  $BIN/llama-perplexity -m $MODEL -f $WIKI -c 4096 --chunks 8 -fa on -ub 2048 -b 2048 --no-mmap > $D/1a.log 2>&1
log "1a exit=$?"; grep "Final estimate" $D/1a.log >> $D/results.log; wait_mem

log "1b ppl c4096 container+fp4mma"
env LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_LID_CACHE_MXFP4=1 LLAMA_DSV4_LID_FP4_MMA=1 \
  $BIN/llama-perplexity -m $MODEL -f $WIKI -c 4096 --chunks 8 -fa on -ub 2048 -b 2048 --no-mmap > $D/1b.log 2>&1
log "1b exit=$?"; grep "Final estimate" $D/1b.log >> $D/results.log; wait_mem

mkpk $D/pk42-10.txt 10 1900
mkpk $D/pk42-97.txt 97 1900
for leg in "1c pk42-10" "1d pk42-97" "1e pk42-97"; do
  set -- $leg
  log "$1 passkey $2 fp4mma @42k"
  run_completion LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_LID_CACHE_MXFP4=1 LLAMA_DSV4_LID_FP4_MMA=1 LLAMA_DSV4_CSA_TILE_MIN=2048 \
    -- $D/$2.txt 49152 $D/$1.out $D/$1.log
  log "$1 exit=$?"
  grep -q $KEY $D/$1.out && echo "$1: PASS ($(head -c 40 $D/$1.out|tr '\n' ' '))" >> $D/results.log \
                         || echo "$1: FAIL ($(head -c 60 $D/$1.out|tr '\n' ' '))" >> $D/results.log
  wait_mem
done
cmp -s $D/1d.out $D/1e.out && echo "determinism 1d==1e: PASS" >> $D/results.log || echo "determinism 1d!=1e: FAIL" >> $D/results.log
sync

# ---------------- item 2: OVF overflow-active passkey @131k ----------------
mkpk $D/pk131-85.txt 85 5800
mkpk $D/pk131-95.txt 95 5800
for leg in "2a pk131-85 OVF0" "2b pk131-85 OVF2048" "2c pk131-95 OVF0" "2d pk131-95 OVF2048"; do
  set -- $leg
  log "$1 passkey $2 $3 @131k POWER=85"
  if [[ $3 == OVF2048 ]]; then
    run_completion GGML_CUDA_POWER=85 LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_CSA_TILE_OVF=2048 \
      -- $D/$2.txt 139264 $D/$1.out $D/$1.log
  else
    run_completion GGML_CUDA_POWER=85 LLAMA_DSV4_FUSED_LID=1 \
      -- $D/$2.txt 139264 $D/$1.out $D/$1.log
  fi
  log "$1 exit=$?"
  grep -q $KEY $D/$1.out && echo "$1 $3: PASS ($(head -c 40 $D/$1.out|tr '\n' ' '))" >> $D/results.log \
                         || echo "$1 $3: MISS ($(head -c 60 $D/$1.out|tr '\n' ' '))" >> $D/results.log
  wait_mem
done
log "ALL FLIP GATES DONE"
