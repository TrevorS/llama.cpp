#!/usr/bin/env bash
# DS4-Flash-0731 + DSpark, 256k context, served on the LAN for a coding agent.
#
# Settings are the measured optima from the 2026-07/08 sweeps, not guesses:
#   POWER=85         agentic traffic is prefill-heavy when cache reuse misses; P90 wedged
#                    the box on exactly that pattern (10th wedge, 2026-07-28). P85 costs
#                    ~5.8% tg vs P90 and is the defensible default for an unknown client.
#   GRANULARITY=layer  chunk-paces the DIRECT-EXEC prefill path (E2). Structurally inert
#                    for decode (replayed graphs can't host-sleep), so it is purely a
#                    long-prompt-ingest burst lever -- which is what agents generate.
#   -c 262144        256k. Costs only ~550 MiB more than 128k: the per-token KV is tiny,
#                    the compute buffer dominates -- which is why UB, not CTX, is the
#                    lever below.
#   -ub 1024         MEASURED, not chosen for speed. At -ub 2048 the compute buffer is
#                    ~3.3 GiB and the box settles ~2.9 GiB free; a 33k-token agent prompt
#                    then drove MemAvailable to 2489 MiB and earlyoom began SIGTERMing
#                    processes (2026-08-01). The compute buffer scales with ubatch, so
#                    halving it returns ~1.4 GiB and keeps the full 256k context. Prefill
#                    cost is ~10-15% (was 357 t/s at 33k depth). On a UMA box a hard OOM
#                    takes the MACHINE, so headroom outranks prefill throughput here.
#   q8_0 K and V     DSV4 requires K and V to match (upstream 69e62fc77). Saves ~700 MiB.
#   n-max 3          best after the target quant was fixed (24.74 t/s vs 23.94 at n=2).
#   -ubd 256         draft-side ubatch; the draft does not need the target's 2048 and
#                    a narrow one frees ~0.5 GiB of draft compute buffer.
#   -lm none         NEVER mmap this model -- mmap OOMs the box (UMA, weights are resident).
#
# The prefill guard (default n=3, min 8192) is on and is what stands between a
# bulk re-prefill storm and a wedge. Do not disable it.
set -uo pipefail

PORT=${PORT:-8080}
CTX=${CTX:-262144}
UB=${UB:-1024}
# -c is TOTAL context in llama.cpp; each slot gets CTX/NP. NP>1 stops a long
# turn head-of-line blocking every request behind it (-np 1 queues them, and a
# client timeout then fires without the request ever reaching a slot).
NP=${NP:-1}
# RAM prompt-cache tier. Upstream defaults --cache-ram to 8192 MiB, which on this
# box is larger than the entire post-load headroom (~3.8 GiB avail at 384k/np2).
# earlyoom SIGTERMs at 2% avail (2492 MiB), so the default allowance alone can walk
# the server into a kill - it did, five times between 08-02 21:13 and 08-03 04:59.
# Our eager L2 store writes the live slot prefix straight to disk and bypasses the
# RAM tier entirely, and over-limit states divert to disk rather than being dropped,
# so a small RAM tier costs no reuse. Must stay > 0: --cache-disk requires it.
CACHE_RAM=${CACHE_RAM:-1024}
# Eager-store headroom gate, MiB. The eager L2 store allocates a full state blob on the
# HOST (~350 MiB at 36k tokens, ~2.3 GiB at 384k) and it grows with the conversation, so a
# deep session is the largest allocation this server makes. earlyoom here runs -m 2,1 =
# SIGTERM at 2% of 124610 MiB = 2492 MiB, so the gate must leave MORE than that free or the
# store it just permitted gets the server killed. 3584 clears the trigger by ~1.1 GiB.
# Cost of a skipped store is a re-prefill; cost of a kill is the whole session.
#
# MEASURED 08-03 on a live 81k-token agent session, which corrected two guesses:
#   * state size is ~0.00211 MiB/token + ~97 MiB base -- 260 MiB at 77k tokens, so a FULL
#     262144-token state is only ~650 MiB, not the ~2.3 GiB the older scaling law implied
#   * avail settles at ~3.2-3.4 GiB during a deep session (it recovers between turns; it
#     does NOT drift down monotonically the way the pre-fix config did)
# 3072 was too tight against that: it skipped a 260 MiB store that would have left 2942 MiB,
# i.e. still 450 MiB clear of the trigger. 2700 permits a worst-case ~650 MiB store at the
# observed floor while keeping ~200 MiB above earlyoom. Do not go below ~2600 (the trigger
# is 2492) and do not go much above ~2800 (that blocks every deep store, silently disabling
# the disk cache and putting the 244 s cold prefill back on the table after any restart).
STORE_MIN_FREE=${STORE_MIN_FREE:-2700}
POWER=${POWER:-85}
NMAX=${NMAX:-3}
SESSION=${SESSION:-ds4-serve}
# refusal-direction ablation (Arditi projection). CVEC=0 serves the stock model.
CVEC=${CVEC:-0}
# Prefill guard: strike out a repeated cold bulk re-prefill with 503 rather than
# start work that has hard-locked this box. Cache reuse is unaffected.
GUARD=${GUARD:-2}
GUARD_MIN=${GUARD_MIN:-8192}
# DeepSeek's published sampling for V4-Flash: temperature 1.0, top_p 1.0.
# The 0731 quant repo publishes top_p 0.95 specifically for Code Agent tasks,
# which is what this box serves, so that is the default here.
# min_p and top_k are set explicitly because llama.cpp's defaults (0.05 / 40)
# are NOT DeepSeek's and silently truncate the distribution that temp 1.0 is
# meant to sample from. Clients sending their own sampling override all of this.
TEMP=${TEMP:-1.0}
TOP_P=${TOP_P:-0.95}
MIN_P=${MIN_P:-0.0}
TOP_K=${TOP_K:-0}
CVEC_DIR=~/Projects/ds4-refusal/llamacpp-iq3
CVEC_GGUF=$CVEC_DIR/weights/native_nothink.gguf
CVEC_SCALE_FILE=$CVEC_DIR/cvec_scale.txt   # live-read each graph build; echo 2.0 > it to retune

M=~/models/ds4/0731/unsloth-iq3xxs-v2/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf
D=~/models/ds4/0731/DSpark-dflash-bf16.gguf   # arch dflash; the old arch-dspark file no longer loads
CACHE_DIR=~/.cache/llama.cpp/ds4-prompt-cache
SLOT_DIR=~/.cache/llama.cpp/ds4-slots        # enables POST /slots/{id}?action=save|restore
LOG=~/.cache/llama.cpp/ds4-serve.log

cd /home/trevor/Projects/llama.cpp
mkdir -p "$CACHE_DIR" "$SLOT_DIR" "$(dirname "$LOG")"

soc(){ for z in /sys/class/thermal/thermal_zone*/temp; do cat "$z" 2>/dev/null; done|sort -rn|head -1|awk '{printf "%.1f",$1/1000}'; }

# Pre-flight the budget. On UMA a hard OOM takes the MACHINE, not a process, and
# earlyoom is poll-based so it cannot stop a single large CUDA allocation.
python3 - "$CTX" "$UB" <<'PY' || { echo "PREFLIGHT REFUSED - not enough headroom"; exit 1; }
import sys
ctx, ub = int(sys.argv[1]), int(sys.argv[2])
TOTAL=124610; SYS=6300; tgt=98961; dft_w=11295; ubd=256
context=(96.75+ctx*0.00562)*0.56
compute=(3017+ctx*0.001007)*(ub/2048.0)*(1.15 if ub<2048 else 1.0)
dft=dft_w+13+1074*(ubd/2048.0)
tot=tgt+context+compute+dft+SYS; free=TOTAL-tot
print(f"  preflight: {tot:.0f} MiB used, {free:.0f} MiB free")
sys.exit(0 if free >= 3400 else 1)
PY

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "  tmux session '$SESSION' already exists -- kill it first: tmux kill-session -t $SESSION"
    exit 1
fi
pgrep -x llama-server >/dev/null && { echo "  a llama-server is already running"; exit 1; }

echo "  SoC $(soc)C  avail $(( $(grep MemAvailable /proc/meminfo|tr -dc '0-9')/1024 )) MiB"
echo "  starting in tmux session '$SESSION' (log: $LOG)"

if [ "$CVEC" = "1" ]; then
    [ -f "$CVEC_GGUF" ] || { echo "  MISSING control vector: $CVEC_GGUF"; exit 1; }
    [ -f "$CVEC_SCALE_FILE" ] || echo 2.0 > "$CVEC_SCALE_FILE"
    CVEC_ENV="LLAMA_CVEC_ABLATE=1 LLAMA_CVEC_AT_FFN=1 LLAMA_CVEC_FFN_ONLY=1 LLAMA_CVEC_SCALE_FILE=$CVEC_SCALE_FILE"
    CVEC_ARG="--control-vector-scaled $CVEC_GGUF:1.0"
    echo "  refusal ablation ON  (ffn-only, scale $(cat "$CVEC_SCALE_FILE"))"
else
    CVEC_ENV=""; CVEC_ARG=""
    echo "  refusal ablation OFF (stock model)"
fi

# setsid: detach from the tmux pane's controlling terminal and process group, so a
# stray Ctrl-C in an attached pane cannot take the server down with it. Kept for
# that reason only - it did NOT stop the mystery shutdowns, which turned out to be
# earlyoom SIGTERM (see CACHE_RAM above). Note earlyoom logs those as
# "kill failed: Timer expired", because llama-server's graceful exit exceeds its
# 10 s wait - so a successful kill looks like a failed one in the journal.
tmux new-session -d -s "$SESSION" "
setsid env GGML_CUDA_POWER=$POWER GGML_CUDA_GRANULARITY=layer \
    LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_HC_FUSED=1 \
    LLAMA_SERVER_PREFILL_GUARD=$GUARD LLAMA_SERVER_PREFILL_GUARD_MIN=$GUARD_MIN \
    LLAMA_SERVER_STORE_MIN_FREE_MB=$STORE_MIN_FREE $CVEC_ENV \
build/bin/llama-server \
  -m '$M' --alias ds4-flash-0731 -lm none \
  -fa on -ngl 999 -c $CTX -ub $UB -np $NP --kv-unified \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-type draft-dspark -md '$D' -ngld 999 --spec-draft-n-max $NMAX --spec-draft-ubatch 256 \
  --cache-ram $CACHE_RAM --cache-disk '$CACHE_DIR' --cache-disk-mb 65536 \
  --slot-save-path '$SLOT_DIR' \
  --jinja --temp $TEMP --top-p $TOP_P --min-p $MIN_P --top-k $TOP_K \
  --reasoning-format deepseek --cache-reuse 256 $CVEC_ARG \
  --host 0.0.0.0 --port $PORT --metrics --slots \
  2>&1 | tee '$LOG'
"

echo -n "  loading (98 GiB, expect a few minutes)"
for i in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    tmux has-session -t "$SESSION" 2>/dev/null || { echo; echo "  SESSION DIED"; tail -30 "$LOG"; exit 1; }
    echo -n "."; sleep 5
done
echo
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "  FAILED TO BECOME HEALTHY"; tail -30 "$LOG"; exit 1; }

LANIP=$(ip -4 -o addr show scope global dev enP7s7 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
TSIP=$(ip -4 -o addr show scope global dev tailscale0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
echo
echo "  READY.  SoC $(soc)C  avail $(( $(grep MemAvailable /proc/meminfo|tr -dc '0-9')/1024 )) MiB"
echo "    LAN        http://${LANIP:-?}:$PORT/v1"
echo "    Tailscale  http://${TSIP:-?}:$PORT/v1"
echo "    model id   ds4-flash-0731"
echo "    attach     tmux attach -t $SESSION   (Ctrl-B then d detaches; Ctrl-C KILLS the server)"
echo "    stop       pkill -x llama-server   (setsid detaches it from tmux, so kill-session alone will NOT stop it)"
