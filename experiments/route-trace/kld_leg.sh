#!/usr/bin/env bash
# One hashification leg measured against a stored base-logits file.
#
# PPL over a handful of chunks (+/- 0.07) cannot resolve a single-layer intervention.
# KL-divergence against the unmodified model's own logits can: it is paired per token,
# so it reports mean KLD, RMS dp and same-top-1 agreement, and the ln(PPL) ratio comes
# with an error bar that is an order of magnitude tighter than two independent PPL runs.
#
# Usage: kld_leg.sh <name> <hsfy|none> <layers|-> [corpus] [base_kld] [chunks] [duty]
set -u
NAME=${1:?usage: kld_leg.sh <name> <hsfy|none> <layers|-> [corpus] [base_kld] [chunks] [duty]}
HSFY=${2:?missing hsfy or 'none'}
LAYERS=${3:--}
CORPUS=${4:-/home/trevor/models/datasets/wiki.test.raw}
BASE=${5:-/home/trevor/models/ds4/kld/base-udiq3-c512-n100.kld}
CHUNKS=${6:-100}
DUTY=${7:-85}

MODEL=/home/trevor/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf
BIN=$(dirname "$0")/../../build/bin/llama-perplexity
OUT=/home/trevor/.cache/hashify-kld
mkdir -p "$OUT"
LOG="$OUT/$NAME.log"
CSV="$OUT/$NAME.thermal.csv"

echo "ts,watts,clock,gpu_c,soc_c" > "$CSV"
(
    hot=0
    while true; do
        smi=$(nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        soc=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)
        soc=$((soc/1000))
        echo "$(date +%H:%M:%S),${smi},${soc}" >> "$CSV"
        # runaway backstop only. 92-97 C is the NORMAL prefill band on this box, so a
        # trigger inside it just truncates good legs - back-to-back legs pushed a 97x2
        # rule into firing at chunk 84. Accumulation is controlled by duty and cooldown;
        # this only catches something genuinely running away.
        if [ "$soc" -ge 99 ]; then hot=$((hot+1)); else hot=0; fi
        if [ "$hot" -ge 3 ]; then
            echo "SOC ABORT: >=99 C x3" >> "$CSV"
            pkill -f llama-perplexity
            break
        fi
        sleep 1
    done
) &
SAMPLER=$!

ENVV=(GGML_CUDA_POWER="$DUTY" GGML_CUDA_POWER_GRANULARITY=layer)
case "$HSFY" in
    none) ;;
    # pass "roll" instead of a table to rotate the live routing at those layers - the
    # same-layer perturbation floor for a hashification leg, no table involved
    roll) ENVV+=(LLAMA_MOE_ORDER_ROLL="$LAYERS") ;;
    *)    ENVV+=(LLAMA_DSV4_HASHIFY="$HSFY")
          [ "$LAYERS" != "-" ] && ENVV+=(LLAMA_DSV4_HASHIFY_LAYERS="$LAYERS") ;;
esac

env "${ENVV[@]}" \
    "$BIN" -m "$MODEL" --no-mmap -fa on -c 512 -ub 2048 -b 2048 -ngl 999 \
    --chunks "$CHUNKS" -f "$CORPUS" \
    --kl-divergence-base "$BASE" --kl-divergence > "$LOG" 2>&1
RC=$?

kill $SAMPLER 2>/dev/null
wait $SAMPLER 2>/dev/null

echo "=== $NAME rc=$RC (hsfy=$(basename "$HSFY") layers=$LAYERS chunks=$CHUNKS duty=$DUTY) ==="
grep -a 'ABORT' "$CSV" || echo "no abort"
awk -F, 'NR>1 && $2+0>0 {n++; if ($2>mw) mw=$2; if ($5>ms) ms=$5}
    END {printf "samples=%d max_watts=%s max_soc=%s\n", n, mw, ms}' "$CSV"
grep -aE 'Mean PPL\(Q\) |Mean ln|Mean +KLD|Median +KLD|^RMS |Same top p' "$LOG" \
    || tail -c 300 "$LOG"
