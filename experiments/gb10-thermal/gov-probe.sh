#!/usr/bin/env bash
# Measure what the cpufreq governor costs thermally, and whether it costs throughput.
#
# Two phases so the answer is honest in both directions:
#   IDLE  — what the box burns doing nothing. `performance` pins all 20 cores at max
#           (10 x 3.9 GHz X925 + 10 x 2.8 GHz A725) regardless of load; schedutil should
#           let them fall toward their floors (1378 / 338 MHz).
#   LOAD  — fixed CPU work on 4 cores. If schedutil ramps correctly this should take the
#           same wall-clock as performance. If it is slower, that is the real cost.
#
# Run once per governor and diff. Nothing here touches the GPU, so it is safe to run
# with a model resident (though the numbers are cleaner without).
set -uo pipefail
TAG=${1:-$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor)}
OUT=${OUT:-$HOME/thermal/gov-$TAG-$(date +%H%M%S)}
IDLE_S=${IDLE_S:-90}
ITERS=${ITERS:-120000000}
mkdir -p "$OUT"

soc(){ cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1 | awk '{printf "%.1f",$1/1000}'; }
freqs(){ cat /sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq 2>/dev/null | paste -sd,; }

echo "════ governor probe: $TAG ════"
echo "  governor: $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor) / $(cat /sys/devices/system/cpu/cpufreq/policy5/scaling_governor)"

echo "── phase 1: idle, ${IDLE_S}s ──"
: > "$OUT/idle.csv"
for i in $(seq 1 $((IDLE_S/3))); do
    echo "$(soc),$(freqs)" >> "$OUT/idle.csv"
    sleep 3
done

echo "── phase 2: fixed CPU work on 4 cores ──"
: > "$OUT/load.csv"
( while :; do echo "$(soc),$(freqs)" >> "$OUT/load.csv"; sleep 2; done ) & SAMP=$!

LOAD_START=$(date +%s.%N)
WPIDS=""
for c in 1 2 3 4; do
    python3 -c "
x=0
for i in range($ITERS): x=(x*1103515245+12345)&0x7fffffff
" &
    WPIDS="$WPIDS $!"
done
# explicit PIDs: `jobs` is unreliable in a non-interactive shell, which made the first
# version return in 1.3 s without ever waiting for the workers
for p in $WPIDS; do wait "$p" 2>/dev/null; done
LOAD_S=$(python3 -c "print(f'{$(date +%s.%N)-$LOAD_START:.1f}')")
kill $SAMP 2>/dev/null

python3 - "$OUT" "$LOAD_S" "$TAG" <<'PY'
import sys, statistics as st, pathlib
out, load_s, tag = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
def parse(p):
    soc, fr = [], []
    for line in (out/p).read_text().splitlines():
        if not line.strip(): continue
        parts = line.split(',')
        soc.append(float(parts[0]))
        fr.append([int(x) for x in parts[1:] if x])
    return soc, fr
isoc, ifr = parse("idle.csv")
lsoc, lfr = parse("load.csv")
def mhz(fr):
    if not fr: return (0,0)
    big  = st.mean(sorted(f)[-10:] for f in ()) if False else st.mean([st.mean(sorted(f)[-10:]) for f in fr])/1000
    lit  = st.mean([st.mean(sorted(f)[:10]) for f in fr])/1000
    return big, lit
ib, il = mhz(ifr); lb, ll = mhz(lfr)
print()
print(f"  RESULT [{tag}]")
print(f"    idle  SoC mean {st.mean(isoc):.1f} C  max {max(isoc):.1f}   big-cluster {ib:.0f} MHz  little {il:.0f} MHz")
if lsoc:
    print(f"    load  SoC mean {st.mean(lsoc):.1f} C  max {max(lsoc):.1f}   big-cluster {lb:.0f} MHz  little {ll:.0f} MHz")
print(f"    fixed CPU work completed in {load_s} s")
PY
