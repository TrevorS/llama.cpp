#!/usr/bin/env zsh
# gates.sh — one-command validation battery for the DSV4 branch (P4).
#
# Usage:
#   gates.sh quick    # ops + coherence + determinism        (~minutes)
#   gates.sh std      # quick + PPL trio + passkey battery   (default-flip gate)
#   gates.sh full     # std + depth perf legs                (milestone)
#   gates.sh ops|coherence|determinism|ppl|passkey|depth     (single stage)
#
# Hard gates exit non-zero; perf legs are soft gates (WARN) because
# cross-process machine-state variance is a known confound on GB10.
# NEVER build while this is running (model residency rule).
set -u
cd "$(dirname "$0")/../.."

MODEL=${GATES_MODEL:-~/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf}
BIN=./build/bin
RUNDIR=experiments/ds4-tile/gates-runs/$(date +%Y%m%d-%H%M%S)
mkdir -p $RUNDIR
typeset -a FAILS WARNS
FAILS=(); WARNS=()

# Common serving-shape args. Defaults are the fast profile as of P4a —
# no env needed. EXACT_ENV is the official-exact reference profile.
COMMON=(-c 16384 -ub 2048 -b 2048 -ngl 999 -fa on --no-mmap)
EXACT_ENV=(LLAMA_DSV4_LID_EXACT=1 LLAMA_DSV4_LID_QAT_WRITE=1)
# Conservative numerics reference: every selection/attention approximation off.
CONSERVATIVE_ENV=(LLAMA_DSV4_CSA_GATHER=0 LLAMA_DSV4_LID_INT8=0 LLAMA_DSV4_LID_DEC=0 LLAMA_DSV4_CSA_TILE=0)

pass() { print -r -- "  [PASS] $1"; }
fail() { print -r -- "  [FAIL] $1"; FAILS+=("$1"); }
warn() { print -r -- "  [WARN] $1"; WARNS+=("$1"); }

# 12k-token deterministic prompt from the immutable upstream-base blob so the
# battery does not drift as the working tree changes.
make_prompt() {
    local out=$1 bytes=$2
    git show e3546c794:ggml/src/ggml.c | head -c $bytes > $out
}

# ---------------------------------------------------------------- stage: ops
stage_ops() {
    print "== ops: test-backend-ops, default env then EXACT profile =="
    local ops=(DSV4_LID_TOPK DSV4_LID_UNION DSV4_LID_MEMB DSV4_HC_FUSED DSV4_FP4_RT DSV4_MOE_GATE_UP)
    local op rc
    for op in $ops; do
        $BIN/test-backend-ops test -o $op > $RUNDIR/ops-$op.log 2>&1
        rc=$?
        if (( rc == 0 )) && grep -q "OK" $RUNDIR/ops-$op.log; then
            pass "ops/$op"
        else
            fail "ops/$op (see $RUNDIR/ops-$op.log)"
        fi
    done
    # EXACT profile: lid_topk must match CPU reference with ZERO tolerance.
    env $EXACT_ENV $BIN/test-backend-ops test -o DSV4_LID_TOPK > $RUNDIR/ops-exact.log 2>&1
    if (( $? == 0 )) && grep -q "OK" $RUNDIR/ops-exact.log; then
        pass "ops/DSV4_LID_TOPK exact-profile (zero tolerance)"
    else
        fail "ops/DSV4_LID_TOPK exact-profile (see $RUNDIR/ops-exact.log)"
    fi
}

# ---------------------------------------------- stage: coherence (defaults)
run_completion() {
    # run_completion <outfile> <ngen> [ENV=VAL ...]
    local out=$1 ngen=$2; shift 2
    env "$@" $BIN/llama-completion -m $MODEL -f $RUNDIR/prompt.txt -n $ngen \
        --temp 0 -s 42 $COMMON -no-cnv --simple-io --no-display-prompt \
        > $out 2> $out.log
}

stage_coherence() {
    print "== coherence: 64 greedy tokens at defaults on a 12k prompt =="
    make_prompt $RUNDIR/prompt.txt 48000
    run_completion $RUNDIR/coherence.out 64
    local n_bytes=$(wc -c < $RUNDIR/coherence.out)
    if (( n_bytes < 40 )); then
        fail "coherence: output only $n_bytes bytes (instant EOS or crash; $RUNDIR/coherence.out.log)"
        return
    fi
    # degenerate-output tripwire: any 12+ repeat of one character
    if grep -qE '(.)\1{11,}' $RUNDIR/coherence.out; then
        fail "coherence: degenerate repetition in output"
    else
        pass "coherence: $n_bytes bytes, no degenerate repetition (eyeball: $RUNDIR/coherence.out)"
    fi
}

# ------------------------------------------- stage: determinism (defaults)
stage_determinism() {
    print "== determinism: identical back-to-back greedy runs, default env =="
    [[ -f $RUNDIR/prompt.txt ]] || make_prompt $RUNDIR/prompt.txt 48000
    run_completion $RUNDIR/det1.out 64
    run_completion $RUNDIR/det2.out 64
    if cmp -s $RUNDIR/det1.out $RUNDIR/det2.out; then
        pass "determinism: run1 == run2"
    else
        # one retry — settled-state pairs are the valid gate (see PROGRESS
        # machine-state variance findings); a fresh pair right after other
        # GPU work can split spuriously.
        run_completion $RUNDIR/det3.out 64
        if cmp -s $RUNDIR/det2.out $RUNDIR/det3.out; then
            warn "determinism: run1 differed but run2 == run3 (settled) — machine-state variance"
        else
            fail "determinism: three runs, no settled pair"
        fi
    fi
}

# -------------------------------------------------------- stage: PPL trio
ppl_of() { grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $1 | grep -oE '[0-9.]+' ; }

stage_ppl() {
    print "== PPL c32768 trio: defaults vs exact vs conservative =="
    # ~650KB from immutable blobs -> ~10 chunks at c32768 (~2 chars/tok code)
    git show e3546c794:ggml/src/ggml.c > $RUNDIR/corpus.txt
    local -a cfgs; cfgs=(defaults exact conservative)
    local cfg
    for cfg in $cfgs; do
        local -a envv; envv=()
        [[ $cfg == exact ]] && envv=($EXACT_ENV)
        [[ $cfg == conservative ]] && envv=($CONSERVATIVE_ENV)
        env $envv $BIN/llama-perplexity -m $MODEL -f $RUNDIR/corpus.txt \
            -c 32768 -ub 2048 -b 2048 -ngl 999 -fa on --no-mmap \
            > $RUNDIR/ppl-$cfg.log 2>&1
        local vals=($(ppl_of $RUNDIR/ppl-$cfg.log))
        if [[ -z ${vals[1]:-} ]]; then
            fail "ppl/$cfg: no final estimate (see $RUNDIR/ppl-$cfg.log)"
            return
        fi
        print "  ppl/$cfg = ${vals[1]} +/- ${vals[2]}"
        eval "ppl_$cfg=${vals[1]}; err_$cfg=${vals[2]}"
    done
    # gate: defaults and exact each within 2 sigma of conservative
    local gate
    gate=$(python3 -c "
import sys
d,e,c,ec = $ppl_defaults,$ppl_exact,$ppl_conservative,$err_conservative
bad = [n for n,v in (('defaults',d),('exact',e)) if abs(v-c) > 2*ec]
print(','.join(bad) if bad else 'ok')")
    if [[ $gate == ok ]]; then
        pass "ppl trio within 2 sigma of conservative reference"
    else
        fail "ppl trio: $gate outside 2 sigma of conservative"
    fi
}

# ---------------------------------------------------- stage: passkey battery
stage_passkey() {
    print "== passkey: 5 depths on a ~42k-token prompt, defaults + tile forced =="
    local key=71432
    local junk="The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    local pos rc=0
    for pos in 10 40 70 90 97; do
        python3 - "$RUNDIR/pk-$pos.txt" $pos <<'PYEOF'
import sys
out, pos = sys.argv[1], int(sys.argv[2])
junk = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
needle = " The pass key is 71432. Remember it. 71432 is the pass key. "
n = 1900  # ~42k tokens of junk at ~22 tok/repeat
k = max(1, n*pos//100)
with open(out, "w") as f:
    f.write("There is a pass key hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about it.\n")
    f.write(junk*k + needle + junk*(n-k))
    f.write("\nWhat is the pass key? The pass key is")
PYEOF
        env LLAMA_DSV4_CSA_TILE_MIN=2048 $BIN/llama-completion -m $MODEL \
            -f $RUNDIR/pk-$pos.txt -n 12 --temp 0 -s 42 \
            -c 49152 -ub 2048 -b 2048 -ngl 999 -fa on --no-mmap \
            -no-cnv --simple-io --no-display-prompt \
            > $RUNDIR/pk-$pos.out 2> $RUNDIR/pk-$pos.log
        if grep -q "$key" $RUNDIR/pk-$pos.out; then
            pass "passkey@$pos%"
        else
            fail "passkey@$pos% (out: $(head -c 80 $RUNDIR/pk-$pos.out | tr '\n' ' '))"
        fi
    done
}

# ------------------------------------------------- stage: depth perf legs
bench_metric() { # bench_metric <log> <pp|tg> -> t/s
    grep -E "$2[0-9]+" $1 | grep -oE '[0-9]+\.[0-9]+ ±' | head -1 | grep -oE '[0-9]+\.[0-9]+'
}

stage_depth() {
    print "== depth legs (SOFT gates; refs: pp2048@d65536 >= 285, tg64@d131072 >= 13.5) =="
    # GB10 power/thermal caps move same-config numbers ~4% — record clock state
    # around every leg so outliers are attributable (experiments/profiles/clocksnap.sh).
    local snap=experiments/profiles/clocksnap.sh
    $snap pre-pp  >> $RUNDIR/clocks.log
    $BIN/llama-bench -m $MODEL -fa on -ub 2048 -b 2048 -mmp 0 -r 1 \
        -p 2048 -n 0 -d 65536 > $RUNDIR/depth-pp.log 2>&1
    $snap post-pp >> $RUNDIR/clocks.log
    local pp=$(bench_metric $RUNDIR/depth-pp.log pp)
    $snap pre-tg  >> $RUNDIR/clocks.log
    $BIN/llama-bench -m $MODEL -fa on -ub 2048 -b 2048 -mmp 0 -r 1 \
        -p 0 -n 64 -d 131072 > $RUNDIR/depth-tg.log 2>&1
    $snap post-tg >> $RUNDIR/clocks.log
    local tg=$(bench_metric $RUNDIR/depth-tg.log tg)
    print "  pp2048@d65536 = ${pp:-?} t/s   tg64@d131072 = ${tg:-?} t/s"
    print "  clock state: $RUNDIR/clocks.log"
    [[ -n $pp ]] && python3 -c "exit(0 if $pp >= 285 else 1)" && pass "pp@d65536 $pp" || warn "pp@d65536 ${pp:-missing} < 285 ref"
    [[ -n $tg ]] && python3 -c "exit(0 if $tg >= 13.5 else 1)" && pass "tg@d131072 $tg" || warn "tg@d131072 ${tg:-missing} < 13.5 ref"
}

# ------------------------------------------------------------------ driver
mode=${1:-std}
case $mode in
    quick) stages=(ops coherence determinism) ;;
    std)   stages=(ops coherence determinism ppl passkey) ;;
    full)  stages=(ops coherence determinism ppl passkey depth) ;;
    ops|coherence|determinism|ppl|passkey|depth) stages=($mode) ;;
    *) print "usage: gates.sh [quick|std|full|<stage>]"; exit 2 ;;
esac

print "gates.sh $mode -> $RUNDIR"
for s in $stages; do stage_$s; done

print ""
print "== summary =="
(( ${#WARNS} )) && { print "WARNS:"; printf '  %s\n' $WARNS }
if (( ${#FAILS} )); then
    print "FAILS:"; printf '  %s\n' $FAILS
    exit 1
fi
print "ALL HARD GATES PASS ($mode)"
