#!/usr/bin/env zsh
# DS4-Flash standing bench matrix: llama-bench pp2048/tg128 at 5 depths +
# llama-completion tg (n=256, temp 0) at 4 prompt depths x {base, mtp}.
# One llama process at a time; cooldown (mem settle + temp) between legs;
# clocksnap around every leg. Usage: bench-matrix.sh <tag>
set -u
cd /home/trevor/Projects/llama.cpp

TAG=${1:-run}
MODEL=~/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf
DRAFT=~/models/ds4/DeepSeek-V4-Flash-MTP-mxfp4.gguf
BIN=./build/bin
RUN=experiments/profiles/matrix-$TAG-$(date +%Y%m%d-%H%M%S)
mkdir -p $RUN
STATUS=$RUN/STATUS
SNAP=experiments/profiles/clocksnap.sh
log() { print -r -- "$(date +%H:%M:%S) $1" >> $STATUS; }

cooldown() {
    # mem avail >= 100G twice, then temp <= 55C (cap 6 min on the temp wait)
    local ok=0 t=0
    while (( ok < 2 )); do
        local avail=$(free -g | awk '/^Mem:/{print $7}')
        if (( avail >= 100 )); then ok=$((ok+1)); else ok=0; fi
        sleep 6
    done
    while (( t < 36 )); do
        local temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        [[ -n "$temp" ]] && (( temp <= 55 )) && break
        sleep 10; t=$((t+1))
    done
}

# deterministic prompt text from immutable blobs (no working-tree drift)
mkprompt() {
    local out=$1 bytes=$2
    { git show e3546c794:ggml/src/ggml.c
      git show e3546c794:ggml/src/ggml-cuda/ggml-cuda.cu
      git show e3546c794:ggml/src/ggml-cpu/ggml-cpu.c
    } | head -c $bytes > $out
}

log "== bench matrix $TAG start (build $(git rev-parse --short HEAD)) =="

# ---- llama-bench legs: pp2048 and tg128 at 5 depths --------------------
for d in 0 16384 32768 65536 131072; do
    for mode in pp tg; do
        if [[ $mode == pp ]]; then args=(-p 2048 -n 0); else args=(-p 0 -n 128); fi
        cooldown
        log "== bench $mode d$d =="
        $SNAP pre-$mode-d$d >> $RUN/clocks.log
        $BIN/llama-bench -m $MODEL -fa on -ub 2048 -b 2048 -mmp 0 -r 3 \
            $args -d $d > $RUN/bench-$mode-d$d.log 2>&1
        $SNAP post-$mode-d$d >> $RUN/clocks.log
        tps=$(grep -oE '[0-9]+\.[0-9]+ ± [0-9]+\.[0-9]+' $RUN/bench-$mode-d$d.log | head -1)
        log "$mode@d$d = ${tps:-FAIL}"
    done
done

# ---- completion tg legs: {base, mtp} x 4 prompt depths ------------------
# bytes ~ 3.1 chars/token for source text
typeset -A PB CTX
PB=(short 4000 32k 101000 65k 203000 131k 406000)
CTX=(short 4096 32k 36864 65k 69632 131k 139264)
for depth in short 32k 65k 131k; do
    mkprompt $RUN/prompt-$depth.txt ${PB[$depth]}
    for arm in base mtp; do
        extra=()
        [[ $arm == mtp ]] && extra=(-md $DRAFT --spec-type draft-mtp)
        cooldown
        log "== completion $arm $depth (ctx ${CTX[$depth]}) =="
        $SNAP pre-$arm-$depth >> $RUN/clocks.log
        $BIN/llama-completion -m $MODEL -f $RUN/prompt-$depth.txt -n 256 \
            --temp 0 -s 42 -fa on --no-mmap -ub 2048 -b 2048 -c ${CTX[$depth]} \
            $extra -no-cnv --simple-io --no-display-prompt \
            > $RUN/gen-$arm-$depth.out 2> $RUN/gen-$arm-$depth.log
        $SNAP post-$arm-$depth >> $RUN/clocks.log
        eval_line=$(grep -E "eval time" $RUN/gen-$arm-$depth.log | tail -1)
        log "$arm@$depth: ${eval_line:-FAIL}"
    done
done

log "== DONE =="
