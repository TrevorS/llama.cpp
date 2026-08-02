#!/usr/bin/env bash
# GB10 thermal soak: does the -lgc clock cap actually prevent the wedge?
#
# The short two-leg test showed the cap is +18% FASTER and ~21 C cooler than P85 duty
# cycling. That establishes throughput and short-run thermals -- it does NOT establish
# wedge immunity, because every wedge on this box has come from SUSTAINED load:
#   wedge #11  38 min at P85          wedge #13  ~21 min at P85, 11 h uptime
# So the target is >45 min of continuous agent-shaped traffic, past both marks.
#
# ROUND 1 DELIBERATELY OMITS CPU CO-LOAD. CPU is a first-class contributor to SoC heat
# (cpu 20-60% is thermally comparable to gpu 80-100%), and wedge #13 had a container on
# 4 cores. Isolating the GPU mitigation first is better experimental design and lower
# risk; if this survives, round 2 adds the CPU co-load and re-runs.
#
# Everything is appended and flushed line-by-line so a hard lockup still leaves the trace
# on disk -- processes keep writing for ~60 s after journald goes quiet.
#
# Usage: soak.sh [minutes]   (default 45)
set -uo pipefail
cd /home/trevor/Projects/llama.cpp

MINUTES=${1:-45}
OUT=${OUT:-$HOME/thermal/soak-$(date +%Y%m%d-%H%M)}
PORT=8232
SOC_ABORT=${SOC_ABORT:-93}   # 93, not 90: both wedges lived at 94-98 C (README 3.0),
                             # and aborting at 90 killed two soaks before they could plateau
MEM_FLOOR=3000

M=~/models/ds4/0731/unsloth-iq3xxs-v2/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf
D=~/models/ds4/0731/DeepSeek-V4-Flash-0731-DSpark-mxfp4.gguf
CACHE=~/.cache/llama.cpp/ds4-soak-cache

mkdir -p "$OUT" "$CACHE"
soc(){ cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1 | awk '{printf "%.1f",$1/1000}'; }
avail(){ echo $(( $(grep MemAvailable /proc/meminfo | tr -dc '0-9') / 1024 )); }

echo "════ GB10 soak: ${MINUTES} min, clock-capped, NO duty cycling ════"
echo "  out: $OUT"
echo "  clock cap: $(systemctl is-active gb10-clock-cap.service 2>/dev/null)  governor: $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor)"
echo "  POWER=${POWER:-none}  GRAN=${GRAN:-none}  PIN=${PIN:-none}  abort=${SOC_ABORT}C"
echo "  SoC start $(soc) C   avail $(avail) MiB"

# ---- telemetry ----
spark-monitor --log "$OUT/spark.csv" --interval 1 > /dev/null 2>&1 &
SPARK=$!

# ---- server: production config, minus GGML_CUDA_POWER ----
${PIN:+taskset -c $PIN} \
env LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_HC_FUSED=1 \
    ${POWER:+GGML_CUDA_POWER=$POWER} ${GRAN:+GGML_CUDA_GRANULARITY=$GRAN} \
build/bin/llama-server -m "$M" --alias soak -lm none \
  -fa on -ngl 999 -c 262144 -ub 1024 -np 1 --kv-unified \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-type draft-dspark -md "$D" -ngld 999 --spec-draft-n-max 3 --spec-draft-ubatch 256 \
  --cache-disk "$CACHE" --cache-disk-mb 65536 \
  --jinja --temp 0 --host 127.0.0.1 --port $PORT \
  > "$OUT/server.log" 2>&1 &
PID=$!

for i in $(seq 1 240); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; kill -0 $PID 2>/dev/null || break; sleep 5; done
if ! curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
    echo "  FAILED TO LOAD"; tail -20 "$OUT/server.log"; kill $SPARK 2>/dev/null; exit 1
fi
echo "  loaded. SoC $(soc) C  avail $(avail) MiB"

# ---- watchdog ----
( while :; do
    t=$(soc); a=$(avail)
    if [ "$(awk -v x="$t" -v y=$SOC_ABORT 'BEGIN{print (x>y)?1:0}')" = 1 ]; then
        echo "ABORT soc=$t" >> "$OUT/abort.txt"; pkill -x llama-server; break
    fi
    if [ "$a" -lt "$MEM_FLOOR" ]; then
        echo "ABORT mem=$a" >> "$OUT/abort.txt"; pkill -x llama-server; break
    fi
    sleep 3
  done ) & WD=$!

# ---- agent-shaped traffic: long prefill, then short cache-reusing follow-ups ----
OUT="$OUT" PORT=$PORT MINUTES=$MINUTES python3 - <<'PY'
import json, os, time, urllib.request, pathlib, glob, itertools

OUT   = pathlib.Path(os.environ["OUT"])
PORT  = os.environ["PORT"]
DEADLINE = time.time() + float(os.environ["MINUTES"]) * 60

srcs = sorted(glob.glob("/home/trevor/Projects/llama.cpp/src/models/*.cpp"))
srcs = [p for p in srcs if os.path.getsize(p) > 20000][:12]
assert srcs, "no source files found"

FOLLOWUPS = ["Name the first function defined.",
             "How many #include lines are there?",
             "Summarise the control flow in two sentences."]

def soc():
    return max(int(open(z).read()) for z in glob.glob("/sys/class/thermal/thermal_zone*/temp")) / 1000.0

def ask(messages, max_tokens):
    body = json.dumps({"messages": messages, "max_tokens": max_tokens, "temperature": 0}).encode()
    req  = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                  {"Content-Type": "application/json"})
    t0 = time.time()
    # 300 s, not 1800: the longest real request here is a ~27k-token prefill at ~90 s.
    # The first run used 1800 and, when the watchdog killed the server mid-prefill, the
    # driver sat on a dead socket for 30 minutes looking like a box hang. Fail fast instead.
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read().decode(), strict=False)
    return d, time.time() - t0


def aborted():
    return (OUT / "abort.txt").exists()


def server_alive():
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=10).read()
        return True
    except Exception:
        return False

# append-and-flush: a hard lockup must still leave the trace behind
f = open(OUT / "requests.csv", "w", buffering=1)
f.write("elapsed_s,cycle,kind,prompt_n,pp_tps,tg_tps,acc,soc_c,avail_mib\n")

t_start = time.time()
print("  traffic started; long prefill then 3 cache-reusing follow-ups per cycle", flush=True)
stop = None
for cycle in itertools.count(1):
    if time.time() > DEADLINE or stop: break
    src = srcs[(cycle - 1) % len(srcs)]
    body = open(src, encoding="utf-8", errors="replace").read()[:120000]
    convo = [{"role": "user", "content": "Read this C++ file.\n\n" + body}]
    plan = [("prefill", 200)] + [("followup", 320)] * len(FOLLOWUPS)
    for i, (kind, mt) in enumerate(plan):
        if time.time() > DEADLINE: break
        if aborted():
            stop = "watchdog abort"; break
        if kind == "followup":
            convo.append({"role": "user", "content": FOLLOWUPS[i - 1]})
        try:
            d, wall = ask(convo, mt)
        except Exception as e:
            f.write(f"{time.time()-t_start:.0f},{cycle},{kind},ERR,,,,{soc():.1f},\n")
            print(f"  [{time.time()-t_start:.0f}s] request failed: {str(e)[:80]}", flush=True)
            # don't keep hammering a dead server -- that is what masked the abort last time
            if aborted():
                stop = "watchdog abort"; break
            if not server_alive():
                stop = "server gone"; break
            time.sleep(5); continue
        msg = d["choices"][0]["message"]
        convo.append({"role": "assistant", "content": msg.get("content") or ""})
        t  = d.get("timings") or {}
        dn = t.get("draft_n") or 0; da = t.get("draft_n_accepted") or 0
        avail = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split("kB")[0]) // 1024
        el = time.time() - t_start
        f.write(f"{el:.0f},{cycle},{kind},{t.get('prompt_n')},{t.get('prompt_per_second') or 0:.1f},"
                f"{t.get('predicted_per_second') or 0:.2f},{(da/dn) if dn else ''},{soc():.1f},{avail}\n")
        print(f"  [{el:>5.0f}s] c{cycle} {kind:<8} prompt_n={str(t.get('prompt_n')):>6} "
              f"tg={t.get('predicted_per_second') or 0:>6.2f} acc={(f'{da/dn:.3f}' if dn else '  n/a')} "
              f"SoC={soc():>5.1f} avail={avail}", flush=True)
f.close()
if stop:
    print(f"  driver stopped early: {stop}", flush=True)
PY

echo "  SoC after $(soc) C   avail $(avail) MiB"
kill $PID $WD $SPARK 2>/dev/null
while kill -0 $PID 2>/dev/null; do sleep 2; done
[ -f "$OUT/abort.txt" ] && echo "  !!! ABORTED: $(cat "$OUT/abort.txt")"
echo "════ soak finished — artifacts in $OUT ════"
