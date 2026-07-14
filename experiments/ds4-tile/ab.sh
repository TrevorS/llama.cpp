#!/usr/bin/env zsh
# Greedy A/B for DSV4 CSA-path variants vs dense, exercising the PREFILL path
# (full prompt reprocessed every run — prompt-cache restore would skip the
# prefill graph and only test decode).
#
# Usage:
#   ab.sh baseline                 # run dense, save reference output
#   ab.sh tile16 LLAMA_DSV4_CSA_TILE=16 LLAMA_DSV4_CSA_TILE_MIN=2048
#   ab.sh <label> [ENV=VAL ...]    # run variant, diff vs baseline
#
# Env knobs: AB_NGEN (default 64), AB_PROMPT (default auto-built ~12k tokens),
#            AB_CTX (default 16384), AB_UB (default 2048)
set -e
cd "$(dirname "$0")/../.."

# IQ3 is the serving quant of record (override with AB_MODEL for IQ2 comparisons)
MODEL=${AB_MODEL:-~/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf}
OUTDIR=experiments/ds4-tile/ab-runs
mkdir -p $OUTDIR

NGEN=${AB_NGEN:-64}
CTX=${AB_CTX:-16384}
UB=${AB_UB:-2048}
PROMPT=${AB_PROMPT:-$OUTDIR/prompt.txt}

# deterministic ~12k-token prompt from repo sources (n_csa ~3k at the end)
if [[ ! -f $PROMPT ]]; then
    head -c 48000 ggml/src/ggml.c > $PROMPT
fi

label=$1; shift || true
[[ -n $label ]] || { echo "usage: ab.sh <label> [ENV=VAL ...]"; exit 1; }

out=$OUTDIR/$label.out
log=$OUTDIR/$label.log

env "$@" ./build/bin/llama-completion -m $MODEL \
    -f $PROMPT -n $NGEN --temp 0 -s 42 \
    -c $CTX -ub $UB -b $UB -ngl 999 -fa on --no-mmap \
    -no-cnv --simple-io --no-display-prompt \
    > $out 2> $log

echo "--- $label (last 3 lines) ---"
tail -3 $out
if [[ $label != baseline && -f $OUTDIR/baseline.out ]]; then
    if diff -q $OUTDIR/baseline.out $out > /dev/null; then
        echo "TOKEN-IDENTICAL vs baseline"
    else
        echo "DIVERGES vs baseline:"
        diff <(head -c 400 $OUTDIR/baseline.out) <(head -c 400 $out) | head -10 || true
    fi
fi
grep -E "prompt eval time|eval time" $log | tail -2
