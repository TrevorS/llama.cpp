#!/usr/bin/env bash
# Calm route-trace corpus leg. Usage: collect_leg.sh <corpus_file> <out_prefix> [duty] [ub]
#
# Protocol after wedge #12 (which happened at only 81.3 W while SoC rode 92-97 C):
# watts did not discriminate, and SoC 92-97 C turns out to be the NORMAL prefill state
# on this box (a 92 C abort fires within ~3 min and makes no progress). So neither is a
# tripwire: the real control is DURATION. Keep each leg short via --chunks, cool down
# between legs, and treat the thermal watch as a runaway backstop only (2 samples >= 97 C).
set -u
CORPUS=${1:?usage: collect_leg.sh <corpus_file> <out_prefix> [duty] [ub] [chunks]}
PREFIX=${2:?missing out_prefix}
DUTY=${3:-70}
UB=${4:-512}
CHUNKS=${5:-12}

MODEL=/home/trevor/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf
BIN=$(dirname "$0")/../../build/bin/llama-perplexity
CSV="${PREFIX}.thermal.csv"
LOG="${PREFIX}.ppl.log"

echo "ts,watts,clock,gpu_c,soc_c" > "$CSV"
(
    hot=0
    while true; do
        smi=$(nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        soc=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)
        soc=$((soc/1000))
        echo "$(date +%H:%M:%S),${smi},${soc}" >> "$CSV"
        # runaway backstop only (SoC 92-97 C is normal here); duration is the real control
        if [ "$soc" -ge 97 ]; then hot=$((hot+1)); else hot=0; fi
        if [ "$hot" -ge 2 ]; then
            echo "SOC ABORT: >=97 C x2" >> "$CSV"
            pkill -f llama-perplexity
            break
        fi
        sleep 1
    done
) &
SAMPLER=$!

env GGML_CUDA_POWER="$DUTY" GGML_CUDA_POWER_GRANULARITY=layer \
    LLAMA_ROUTE_TRACE="$PREFIX" \
    "$BIN" -m "$MODEL" --no-mmap -fa on -c 2048 -ub "$UB" -b 2048 -ngl 999 \
    --chunks "$CHUNKS" -f "$CORPUS" > "$LOG" 2>&1
RC=$?

kill $SAMPLER 2>/dev/null
wait $SAMPLER 2>/dev/null

echo "=== rc=$RC (duty=$DUTY ub=$UB chunks=$CHUNKS) ==="
grep -a 'ABORT' "$CSV" || echo "no abort"
awk -F, 'NR>1 && $2+0>0 {n++; if ($2>mw) mw=$2; if ($5>ms) ms=$5}
    END {printf "samples=%d max_watts=%s max_soc=%s\n", n, mw, ms}' "$CSV"
grep -aoE 'Final estimate: PPL = [0-9.]+' "$LOG" || tail -c 200 "$LOG"
ls -la "$(dirname "$PREFIX")" | grep "$(basename "$PREFIX")" | awk '{print $5, $9}'
