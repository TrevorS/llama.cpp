#!/usr/bin/env bash
# Robust refusal-ablation scale sweep. One server at a time, health-checked,
# port-verified between scales. Args: <direction.gguf> <at_ffn 0|1> <out_prefix> <scale...>
set -u
DIR="$1"; AT_FFN="$2"; PREFIX="$3"; shift 3; SCALES=("$@")
R=/home/trevor/Projects/llama.cpp/experiments/refusal
M=/home/trevor/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf
BIN=/home/trevor/Projects/llama.cpp/build/bin/llama-server
PORT=8080

kill_servers() {
  pkill -9 -f "bin/llama-server" 2>/dev/null
  for _ in $(seq 1 20); do
    ss -tln 2>/dev/null | grep -q ":$PORT " || return 0
    sleep 1
  done
}

for SC in "${SCALES[@]}"; do
  kill_servers
  LOG="$PREFIX.srv_$SC.log"
  ATFFN_ENV=""; [ "$AT_FFN" = "1" ] && ATFFN_ENV="LLAMA_CVEC_AT_FFN=1"
  env LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_HC_FUSED=1 LLAMA_CVEC_ABLATE=1 $ATFFN_ENV \
      LLAMA_CVEC_ABLATE_SCALE=$SC \
    "$BIN" -m "$M" --alias ds4-flash-iq3 --control-vector-scaled "$DIR:1.0" \
    --no-mmap -fa on -ngl 999 -c 8192 -ub 512 --temp 0 \
    --port $PORT --host 127.0.0.1 > "$LOG" 2>&1 &
  SRV=$!

  # wait for listening
  ok=0
  for _ in $(seq 1 50); do
    grep -aq "listening on" "$LOG" && { ok=1; break; }
    grep -aqE "couldn't bind|error while|failed to" "$LOG" && break
    sleep 8
  done
  if [ "$ok" != "1" ]; then
    echo "scale $SC: SERVER DID NOT START"; grep -aiE "bind|error|failed" "$LOG" | head -1
    kill -9 $SRV 2>/dev/null; continue
  fi

  # health check: a real request must return HTTP 200 with content before batch
  hc=0
  for _ in $(seq 1 10); do
    if curl -sf -m 60 http://127.0.0.1:$PORT/v1/chat/completions \
         -H 'Content-Type: application/json' \
         -d '{"model":"ds4-flash-iq3","messages":[{"role":"user","content":"Say OK"}],"max_tokens":5,"temperature":0}' \
         >/dev/null 2>&1; then hc=1; break; fi
    sleep 3
  done
  if [ "$hc" != "1" ]; then
    echo "scale $SC: HEALTH CHECK FAILED"; kill -9 $SRV 2>/dev/null; continue
  fi

  PROMPTS_FILE="${PROMPTS:-$R/harmful_hard.txt}"
  python3 "$R/eval_refusal.py" --prompts "$PROMPTS_FILE" \
    --out "$PREFIX.c_$SC.jsonl" --max-tokens 768 2>/dev/null
  echo "scale $SC done ($(grep -ac . "$PREFIX.c_$SC.jsonl") prompts)"
  kill -9 $SRV 2>/dev/null; wait $SRV 2>/dev/null
done
kill_servers
echo "SWEEP_DONE"
