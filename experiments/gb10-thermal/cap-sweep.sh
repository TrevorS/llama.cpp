#!/usr/bin/env bash
# Find the highest GPU clock cap that survives sustained prefill without reaching 88 C.
#
# Why the cap and not duty: two soaks showed layer-granularity duty only DOUBLES
# time-to-90 (182 s -> 361 s) for ~9 % throughput, and still fails. Duty throttles by
# discarding wall-clock; a clock cap lowers the V/f point, and voltage scaling is
# superlinear -- so it should buy more thermal per unit throughput lost.
#
# ONE model load for the whole sweep: -lgc is a driver-level setting and can be changed
# live between legs. Requires /etc/sudoers.d/nvidia-clocks so the cap can be set without
# a password.
#
# Each leg: set cap -> cool to a common start -> hammer prefill -> record. A leg PASSES
# if it holds under HOT_C for the full duration. Cooling to a common start matters:
# idle time predicts thermal capacity on this box, so legs from different starts are
# not comparable.
set -uo pipefail
cd /home/trevor/Projects/llama.cpp

CAPS=${CAPS:-"2200 2000 1800 1600"}
LEG_S=${LEG_S:-360}          # sustained load per leg
COOL_C=${COOL_C:-56}         # cool to this before each leg
COOL_MAX_S=${COOL_MAX_S:-600}
HOT_C=${HOT_C:-88}           # leg fails on reaching this
ABORT_C=${ABORT_C:-91}
PORT=8233
OUT=${OUT:-$HOME/thermal/capsweep-$(date +%Y%m%d-%H%M)}

M=~/models/ds4/0731/unsloth-iq3xxs-v2/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf
D=~/models/ds4/0731/DeepSeek-V4-Flash-0731-DSpark-mxfp4.gguf

mkdir -p "$OUT"
soc(){ cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1 | awk '{printf "%.1f",$1/1000}'; }
hotter(){ [ "$(awk -v a="$1" -v b="$2" 'BEGIN{print (a>=b)?1:0}')" = 1 ]; }

sudo -n nvidia-smi -lgc 300,2200 >/dev/null 2>&1 || {
  echo "need passwordless nvidia-smi. Run:"
  echo "  echo 'trevor ALL=(root) NOPASSWD: /usr/bin/nvidia-smi -lgc *, /usr/bin/nvidia-smi -rgc' | sudo tee /etc/sudoers.d/nvidia-clocks > /dev/null && sudo visudo -c"
  exit 1
}

echo "════ clock-cap sweep: ${CAPS} ════"
echo "  leg ${LEG_S}s | cool to ${COOL_C}C | fail at ${HOT_C}C | abort ${ABORT_C}C"

spark-monitor --log "$OUT/spark.csv" --interval 1 >/dev/null 2>&1 & SPARK=$!

env LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_HC_FUSED=1 \
build/bin/llama-server -m "$M" --alias sweep -lm none \
  -fa on -ngl 999 -c 262144 -ub 1024 -np 1 --kv-unified \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-type draft-dspark -md "$D" -ngld 999 --spec-draft-n-max 3 --spec-draft-ubatch 256 \
  --jinja --temp 0 --host 127.0.0.1 --port $PORT > "$OUT/server.log" 2>&1 &
PID=$!
for i in $(seq 1 240); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; kill -0 $PID 2>/dev/null || break; sleep 5; done
curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 || { echo "  FAILED TO LOAD"; tail -20 "$OUT/server.log"; kill $SPARK 2>/dev/null; exit 1; }
echo "  server up. SoC $(soc) C"
echo "cap,start_c,peak_c,t_to_85_s,t_to_88_s,pp_tps,tg_tps,verdict" > "$OUT/results.csv"

for CAP in $CAPS; do
  sudo -n nvidia-smi -lgc 300,$CAP >/dev/null 2>&1
  echo "── cap $CAP: cooling to ${COOL_C}C ──"
  c0=$(date +%s)
  while hotter "$(soc)" "$COOL_C"; do
      [ $(( $(date +%s) - c0 )) -gt $COOL_MAX_S ] && { echo "    cooldown timeout at $(soc)C"; break; }
      sleep 10
  done
  START=$(soc); echo "    start $START C — loading for ${LEG_S}s"

  T85=""; T88=""; PEAK=$START; VERDICT=PASS
  ( LEG_S=$LEG_S PORT=$PORT python3 - <<'PY'
import json,os,time,urllib.request,glob
PORT=os.environ["PORT"]; END=time.time()+float(os.environ["LEG_S"])
srcs=[p for p in sorted(glob.glob("/home/trevor/Projects/llama.cpp/src/models/*.cpp")) if os.path.getsize(p)>20000][:12]
i=0
while time.time()<END:
    body=open(srcs[i%len(srcs)],encoding="utf-8",errors="replace").read()[:120000]; i+=1
    try:
        req=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json.dumps({"messages":[{"role":"user","content":"Read this file.\n\n"+body}],
                        "max_tokens":80,"temperature":0}).encode(),
            {"Content-Type":"application/json"})
        d=json.loads(urllib.request.urlopen(req,timeout=300).read().decode(),strict=False)
        t=d.get("timings") or {}
        print(f"{t.get('prompt_per_second') or 0:.1f},{t.get('predicted_per_second') or 0:.2f}",flush=True)
    except Exception:
        break
PY
  ) > "$OUT/leg-$CAP.txt" 2>&1 & LEG=$!

  t0=$(date +%s)
  while kill -0 $LEG 2>/dev/null; do
      s=$(soc); el=$(( $(date +%s) - t0 ))
      hotter "$s" "$PEAK" && PEAK=$s
      [ -z "$T85" ] && hotter "$s" 85 && T85=$el
      [ -z "$T88" ] && hotter "$s" "$HOT_C" && { T88=$el; VERDICT=FAIL; }
      hotter "$s" "$ABORT_C" && { echo "    !!! abort ${s}C"; kill -9 $LEG 2>/dev/null; VERDICT=ABORT; break; }
      sleep 2
  done
  wait $LEG 2>/dev/null

  PP=$(awk -F, '{s+=$1;n++} END{if(n)printf "%.1f",s/n}' "$OUT/leg-$CAP.txt" 2>/dev/null)
  TG=$(awk -F, '{s+=$2;n++} END{if(n)printf "%.2f",s/n}' "$OUT/leg-$CAP.txt" 2>/dev/null)
  echo "$CAP,$START,$PEAK,${T85:-},${T88:-},${PP:-},${TG:-},$VERDICT" >> "$OUT/results.csv"
  printf "    cap %s: peak %sC  t85=%ss t88=%ss  pp=%s tg=%s  -> %s\n" \
         "$CAP" "$PEAK" "${T85:-none}" "${T88:-none}" "${PP:-?}" "${TG:-?}" "$VERDICT"
done

kill $PID $SPARK 2>/dev/null; while kill -0 $PID 2>/dev/null; do sleep 2; done
sudo -n nvidia-smi -lgc 300,2200 >/dev/null 2>&1
echo "════ sweep done ════"; column -t -s, "$OUT/results.csv"
