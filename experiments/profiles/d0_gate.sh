#!/usr/bin/env bash
# D0 post-wedge health gate (gb10 playbook). Usage: d0_gate.sh <outdir> [load_seconds]
# PASS: >=1400 MHz and >=40 W under load, no 611 MHz / 13-16 W pin.
# Records SoC temp (max thermal zone) per the 07-27 amendment.
set -u
D0=${1:?usage: d0_gate.sh <outdir> [load_seconds]}
SECS=${2:-75}
mkdir -p "$D0"

nvidia-smi -q > "$D0/nvidia-smi-q-pre.txt" 2>&1
for z in /sys/class/thermal/thermal_zone*; do
    echo "$(cat $z/type 2>/dev/null): $(cat $z/temp 2>/dev/null)"
done > "$D0/soc-pre.txt"

echo "ts,clocks_sm_mhz,power_w,gpu_temp_c,soc_temp_max_c" > "$D0/load-samples.csv"
(
    while true; do
        smi=$(nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        soc=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)
        echo "$(date +%H:%M:%S),${smi},$((soc/1000))" >> "$D0/load-samples.csv"
        sleep 1
    done
) &
SAMPLER=$!

timeout "$SECS" env GGML_CUDA_POWER=70 \
    "$(dirname "$0")/../../build/bin/test-backend-ops" perf -o MUL_MAT -b CUDA0 > "$D0/load-run.log" 2>&1
RC=$?

sleep 2
kill $SAMPLER 2>/dev/null
wait $SAMPLER 2>/dev/null

echo "=== load rc=$RC (124 = timeout cap, expected) ==="
awk -F, 'NR>1 {n++; if ($2>maxc) maxc=$2; if ($3>maxw) maxw=$3; if ($4>maxt) maxt=$4; if ($5>maxs) maxs=$5;
    if ($2>=1400) hi++; if ($2<=700 && $3<=20) pin++}
    END {printf "samples=%d max_clock=%s max_watts=%s max_gpu_temp=%s max_soc_temp=%s hi_clock_samples=%d pin_samples=%d\n", n, maxc, maxw, maxt, maxs, hi, pin}' "$D0/load-samples.csv"
