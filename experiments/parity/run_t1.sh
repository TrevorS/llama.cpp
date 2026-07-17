#!/usr/bin/env bash
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); BIN=~/Projects/llama.cpp/build/bin
IQ3=~/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf
IQ2=$(ls ~/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ2_XXS/*00001*.gguf | head -1)
wait_mem() { for i in $(seq 1 80); do a1=$(free -g|awk '/^Mem:/{print $7}'); sleep 6; a2=$(free -g|awk '/^Mem:/{print $7}'); [[ $a1 -ge 100 && $a2 -ge 100 ]] && return 0; done; return 1; }
ppl() { local m=$1 out=$2; shift 2; env LLAMA_DSV4_FUSED_LID=1 "$@" $BIN/llama-perplexity -m $m -f $HERE/corpus-t1.txt -c 1024 --chunks 20 --parse-special --save-all-logits $out -fa on -ub 2048 -b 2048 --no-mmap > ${out%.kld}.log 2>&1; }
cd $HERE
for spec in "arm1-fast-t1:$IQ3:LLAMA_DSV4_LID_CACHE_MXFP4=1" \
            "arm2-exact-t1:$IQ3:LLAMA_DSV4_LID_CACHE_MXFP4=1 LLAMA_DSV4_LID_EXACT=1 LLAMA_DSV4_CSA_TILE_OVF=2048" \
            "arm3-iq2-t1:$IQ2:LLAMA_DSV4_LID_CACHE_MXFP4=1" \
            "arm4-bf16-t1:$IQ3:LLAMA_DSV4_LID_CACHE_MXFP4=1 LLAMA_DSV4_KV_BF16_RT=1"; do
  name=${spec%%:*}; rest=${spec#*:}; model=${rest%%:*}; envs=${rest#*:}
  wait_mem; echo "== $name =="
  ppl $model $name.kld $envs
  python3 analyze.py $name.kld --meta meta-t1.json | tee $name.report
done
echo "== T1 LADDER DONE =="
