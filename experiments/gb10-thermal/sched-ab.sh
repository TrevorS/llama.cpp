#!/usr/bin/env bash
# A/B the CUDA host-sync policy on GB10: spin (upstream default) vs yield vs blocking.
#
# llama.cpp sets cudaDeviceScheduleSpin for cc121 specifically, to cut sync latency on
# iGPUs. On a discrete GPU the spinning core is free. On GB10 the CPU and GPU share one
# package and one thermal budget, so that core -- up to 3.9 GHz on an X925 -- is pure
# heat taken out of the GPU's headroom.
#
# The measurement that matters is CORES BURNED DURING DECODE, taken from the server
# process's own utime+stime so it cannot be confused with background noise. Throughput
# and latency are measured alongside, because upstream chose spin for a reason and the
# trade only makes sense if the latency cost is small.
set -uo pipefail
cd /home/trevor/Projects/llama.cpp

POLICIES=${POLICIES:-"spin yield blocking"}
PORT=8234
OUT=${OUT:-$HOME/thermal/sched-ab-$(date +%Y%m%d-%H%M)}
M=~/models/ds4/0731/unsloth-iq3xxs-v2/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf
D=~/models/ds4/0731/DeepSeek-V4-Flash-0731-DSpark-mxfp4.gguf
mkdir -p "$OUT"

soc(){ cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1 | awk '{printf "%.1f",$1/1000}'; }
CLK=$(getconf CLK_TCK)

echo "════ CUDA sync-policy A/B: $POLICIES ════"
echo "  governor $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor) | clock cap $(systemctl is-active gb10-clock-cap.service 2>/dev/null)"
echo "policy,tg_tps,acc,cores_decode,soc_start,soc_end,ttft_ms" > "$OUT/results.csv"

for POL in $POLICIES; do
    # cool a little so legs start comparably
    while [ "$(awk -v a="$(soc)" 'BEGIN{print (a>56)?1:0}')" = 1 ]; do sleep 10; done
    S0=$(soc)
    echo "── $POL (start ${S0}C) ──"

    env GGML_CUDA_SCHED="$POL" LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_HC_FUSED=1 \
    build/bin/llama-server -m "$M" --alias ab -lm none -fa on -ngl 999 \
      -c 16384 -ub 2048 -np 1 --kv-unified \
      --spec-type draft-dspark -md "$D" -ngld 999 --spec-draft-n-max 2 \
      --jinja --temp 0 --host 127.0.0.1 --port $PORT > "$OUT/$POL.log" 2>&1 &
    PID=$!
    for i in $(seq 1 240); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; kill -0 $PID 2>/dev/null || break; sleep 4; done
    if ! curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
        echo "    FAILED TO LOAD"; tail -5 "$OUT/$POL.log"; kill $PID 2>/dev/null; continue
    fi
    grep -a 'GGML_CUDA_SCHED' "$OUT/$POL.log" | head -1 | sed 's/^/    /'

    # warm-up so graph capture and first-touch costs land outside the measured window
    curl -s --max-time 600 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":32,"temperature":0}' >/dev/null

    read -r U0 S0T < <(awk '{print $14, $15}' /proc/$PID/stat)
    T0=$(date +%s.%N)
    RESP=$(curl -s --max-time 900 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Explain how a B-tree index speeds up range queries in a relational database, then contrast it with a hash index."}],"max_tokens":320,"temperature":0}')
    T1=$(date +%s.%N)
    read -r U1 S1T < <(awk '{print $14, $15}' /proc/$PID/stat)

    CORES=$(python3 -c "print(f'{(($U1-$U0)+($S1T-$S0T))/$CLK/($T1-$T0):.2f}')")
    S1=$(soc)
    # parse in python, arithmetic in bash: passing shell vars AFTER `python3 -c` makes
    # them argv, not environ, which silently produced empty rows on the first attempt
    METRICS=$(printf '%s' "$RESP" | python3 -c "
import json,sys
d=json.load(sys.stdin,strict=False); t=d.get('timings') or {}
dn=t.get('draft_n') or 0; da=t.get('draft_n_accepted') or 0
print('%.2f,%.4f,%.0f' % (t.get('predicted_per_second') or 0, (da/dn) if dn else 0, t.get('prompt_ms') or 0))")
    TG=$(echo "$METRICS" | cut -d, -f1)
    ACC=$(echo "$METRICS" | cut -d, -f2)
    TTFT=$(echo "$METRICS" | cut -d, -f3)
    echo "    tg=${TG} t/s  acc=${ACC}  cores_during_decode=${CORES}  SoC ${S0}->${S1}C  ttft=${TTFT}ms"
    echo "$POL,$TG,$ACC,$CORES,$S0,$S1,$TTFT" >> "$OUT/results.csv"

    kill $PID 2>/dev/null; while kill -0 $PID 2>/dev/null; do sleep 2; done
done

echo "════ done ════"; column -t -s, "$OUT/results.csv"
