#!/usr/bin/env bash
# Run the 4 parity arms over corpus.txt, dump logits, analyze each vs the API,
# and compute native full-vocab KLD of arms 2-4 against arm 1.
# One model process at a time; waits for memory between legs.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
BIN=$REPO/build/bin
MODEL_IQ3=~/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf
MODEL_IQ2=~/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ2_XXS/*00001*.gguf
N_CHUNKS=$(python3 -c "import json;print(len(json.load(open('$HERE/meta.json'))['units']))")

wait_mem() { for i in $(seq 1 80); do a1=$(free -g|awk '/^Mem:/{print $7}'); sleep 6; a2=$(free -g|awk '/^Mem:/{print $7}'); [[ $a1 -ge 100 && $a2 -ge 100 ]] && return 0; done; return 1; }

ppl() { # ppl <model> <outfile.kld> [extra env as VAR=VAL ...]
  local model=$1 out=$2; shift 2
  env LLAMA_DSV4_FUSED_LID=1 "$@" \
    $BIN/llama-perplexity -m $model -f $HERE/corpus.txt -c 2048 --chunks $N_CHUNKS \
    --parse-special --save-all-logits $out -fa on -ub 2048 -b 2048 --no-mmap \
    > ${out%.kld}.log 2>&1
}

cd $HERE
echo "== arm1: fast profile (defaults + container) =="
ppl $MODEL_IQ3 arm1-fast.kld LLAMA_DSV4_LID_CACHE_MXFP4=1
python3 analyze.py arm1-fast.kld | tee arm1-fast.report
wait_mem

echo "== arm2: EXACT profile =="
ppl $MODEL_IQ3 arm2-exact.kld LLAMA_DSV4_LID_CACHE_MXFP4=1 LLAMA_DSV4_LID_EXACT=1 LLAMA_DSV4_CSA_TILE_OVF=2048
python3 analyze.py arm2-exact.kld | tee arm2-exact.report
wait_mem

echo "== arm3: IQ2_XXS quant ladder =="
ppl $(ls $MODEL_IQ2 | head -1) arm3-iq2.kld LLAMA_DSV4_LID_CACHE_MXFP4=1
python3 analyze.py arm3-iq2.kld | tee arm3-iq2.report
wait_mem

echo "== arm4: bf16-RT caches =="
ppl $MODEL_IQ3 arm4-bf16.kld LLAMA_DSV4_LID_CACHE_MXFP4=1 LLAMA_DSV4_KV_BF16_RT=1
python3 analyze.py arm4-bf16.kld | tee arm4-bf16.report
wait_mem

echo "== native full-vocab KLD vs arm1 (llama-side, exact) =="
for arm in arm2-exact "arm3-iq2" arm4-bf16; do
  case $arm in
    arm2-exact) envs=(LLAMA_DSV4_LID_CACHE_MXFP4=1 LLAMA_DSV4_LID_EXACT=1 LLAMA_DSV4_CSA_TILE_OVF=2048); m=$MODEL_IQ3;;
    arm3-iq2)   envs=(LLAMA_DSV4_LID_CACHE_MXFP4=1); m=$(ls $MODEL_IQ2 | head -1);;
    arm4-bf16)  envs=(LLAMA_DSV4_LID_CACHE_MXFP4=1 LLAMA_DSV4_KV_BF16_RT=1); m=$MODEL_IQ3;;
  esac
  env LLAMA_DSV4_FUSED_LID=1 "${envs[@]}" \
    $BIN/llama-perplexity -m $m -f $HERE/corpus.txt -c 2048 --chunks $N_CHUNKS \
    --parse-special --kl-divergence-base arm1-fast.kld --kl-divergence \
    -fa on -ub 2048 -b 2048 --no-mmap > kld-$arm-vs-arm1.log 2>&1
  grep -E "Average|Maximum|KLD|Same top" kld-$arm-vs-arm1.log | tail -8
  wait_mem
done
echo "== ALL ARMS DONE =="
