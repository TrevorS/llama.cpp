#!/usr/bin/env zsh
# Post-wedge-12 safe remainder: completion legs short/32k/65k only (NO 131k —
# one-deep-fill-per-boot rule is now absolute), then parity arms.
set -u
cd /home/trevor/Projects/llama.cpp
MODEL=~/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf
DRAFT=~/models/ds4/DeepSeek-V4-Flash-MTP-mxfp4.gguf
BIN=./build/bin
RUN=experiments/profiles/matrix-safe-remainder-$(date +%Y%m%d-%H%M%S)
mkdir -p $RUN
STATUS=$RUN/STATUS
log() { print -r -- "$(date +%H:%M:%S) $1" >> $STATUS; }
cooldown() {
    local ok=0
    while (( ok < 2 )); do
        local avail=$(free -g | awk '/^Mem:/{print $7}')
        if (( avail >= 100 )); then ok=$((ok+1)); else ok=0; fi
        sleep 6
    done
    sleep 90
}
mkprompt() {
    local out=$1 bytes=$2
    { git show e3546c794:ggml/src/ggml.c
      git show e3546c794:ggml/src/ggml-cuda/ggml-cuda.cu
      git show e3546c794:ggml/src/ggml-cpu/ggml-cpu.c
    } | head -c $bytes > $out
}
log "== safe remainder start (build $(git rev-parse --short HEAD)) =="
typeset -A PB CTX
PB=(short 4000 32k 101000 65k 203000)
CTX=(short 4096 32k 36864 65k 69632)
for depth in short 32k 65k; do
    mkprompt $RUN/prompt-$depth.txt ${PB[$depth]}
    for arm in base mtp; do
        extra=()
        [[ $arm == mtp ]] && extra=(-md $DRAFT --spec-type draft-mtp)
        envs=(LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_LID_CACHE_MXFP4=1)
        [[ $depth == 65k ]] && envs+=(GGML_CUDA_POWER=85 GGML_CUDA_POWER_DEBUG=1)
        cooldown
        log "== completion $arm $depth =="
        env $envs $BIN/llama-completion -m $MODEL -f $RUN/prompt-$depth.txt -n 256 \
            --temp 0 -s 42 -fa on --no-mmap -ub 2048 -b 2048 -c ${CTX[$depth]} \
            $extra -no-cnv --simple-io --no-display-prompt \
            > $RUN/gen-$arm-$depth.out 2> $RUN/gen-$arm-$depth.log
        eval_line=$(grep -E "eval time" $RUN/gen-$arm-$depth.log | tail -1)
        log "$arm@$depth: ${eval_line:-FAIL}"
    done
done
log "== completions done, starting parity arms =="
experiments/parity/run_arms.sh >> $RUN/parity.log 2>&1
log "== ALL DONE =="
