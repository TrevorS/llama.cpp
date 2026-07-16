#!/usr/bin/env zsh
# Resume of matrix-newdefaults after the 11:45 wedge: remaining legs only,
# d131k first (fresh boot), POWER=75 on deep legs, power-settle cooldown.
set -u
cd /home/trevor/Projects/llama.cpp
MODEL=~/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf
DRAFT=~/models/ds4/DeepSeek-V4-Flash-MTP-mxfp4.gguf
BIN=./build/bin
RUN=experiments/profiles/matrix-newdefaults-resume-$(date +%Y%m%d-%H%M%S)
mkdir -p $RUN
STATUS=$RUN/STATUS
SNAP=experiments/profiles/clocksnap.sh
log() { print -r -- "$(date +%H:%M:%S) $1" >> $STATUS; }

powercap_us() { nvidia-smi -q -d PERFORMANCE 2>/dev/null | grep -i "SW Power Capping" | grep -oE '[0-9]+' | head -1; }
cooldown() {
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
    # NOTE: no power-settle gate — the cumulative SW-power-cap counter
    # advances at IDLE by design (parked clocks count as capping), so there
    # is no usable settle signal. Fixed 90s grace instead.
    sleep 90
}

log "== matrix resume start (build $(git rev-parse --short HEAD)) =="

for d in 131072; do
    for mode in pp tg; do
        if [[ $mode == pp ]]; then args=(-p 2048 -n 0); else args=(-p 0 -n 128); fi
        envs=(LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_LID_CACHE_MXFP4=1 GGML_CUDA_POWER=75 GGML_CUDA_POWER_DEBUG=1)
        cooldown
        log "== bench $mode d$d (POWER=75) =="
        $SNAP pre-$mode-d$d >> $RUN/clocks.log
        env $envs $BIN/llama-bench -m $MODEL -fa on -ub 2048 -b 2048 -mmp 0 -r 3 \
            $args -d $d > $RUN/bench-$mode-d$d.log 2>&1
        $SNAP post-$mode-d$d >> $RUN/clocks.log
        tps=$(grep -oE '[0-9]+\.[0-9]+ ± [0-9]+\.[0-9]+' $RUN/bench-$mode-d$d.log | head -1)
        log "$mode@d$d = ${tps:-FAIL}"
    done
done

mkprompt() {
    local out=$1 bytes=$2
    { git show e3546c794:ggml/src/ggml.c
      git show e3546c794:ggml/src/ggml-cuda/ggml-cuda.cu
      git show e3546c794:ggml/src/ggml-cpu/ggml-cpu.c
    } | head -c $bytes > $out
}
typeset -A PB CTX
PB=(short 4000 32k 101000 65k 203000 131k 406000)
CTX=(short 4096 32k 36864 65k 69632 131k 139264)
for depth in short 32k 65k 131k; do
    mkprompt $RUN/prompt-$depth.txt ${PB[$depth]}
    for arm in base mtp; do
        extra=()
        [[ $arm == mtp ]] && extra=(-md $DRAFT --spec-type draft-mtp)
        envs=(LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_LID_CACHE_MXFP4=1)
        [[ $depth == 65k ]] && envs+=(GGML_CUDA_POWER=85 GGML_CUDA_POWER_DEBUG=1)
        [[ $depth == 131k ]] && envs+=(GGML_CUDA_POWER=75 GGML_CUDA_POWER_DEBUG=1)
        cooldown
        log "== completion $arm $depth (ctx ${CTX[$depth]}) =="
        $SNAP pre-$arm-$depth >> $RUN/clocks.log
        env $envs $BIN/llama-completion -m $MODEL -f $RUN/prompt-$depth.txt -n 256 \
            --temp 0 -s 42 -fa on --no-mmap -ub 2048 -b 2048 -c ${CTX[$depth]} \
            $extra -no-cnv --simple-io --no-display-prompt \
            > $RUN/gen-$arm-$depth.out 2> $RUN/gen-$arm-$depth.log
        $SNAP post-$arm-$depth >> $RUN/clocks.log
        eval_line=$(grep -E "eval time" $RUN/gen-$arm-$depth.log | tail -1)
        log "$arm@$depth: ${eval_line:-FAIL}"
    done
done
log "== RESUME DONE =="
