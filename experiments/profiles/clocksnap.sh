#!/usr/bin/env zsh
# clocksnap.sh [label] — one-line GPU clock/throttle/temperature record for
# bench logs. GB10 power/thermal capping moves same-config numbers by ~4%
# (SW Power Capping counter reached 77 min in one boot, 2026-07-14), so every
# perf leg should carry a snapshot before and after.
q() { nvidia-smi --query-gpu=$1 --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' }
active=$(nvidia-smi -q -d PERFORMANCE 2>/dev/null | awk -F': ' '
    /Clocks Event Reasons Counters/{exit}
    /: Active/{gsub(/^ +| +$/,"",$1); s=s (s?",":"") $1}
    END{print s?s:"none"}')
capus=$(nvidia-smi -q -d PERFORMANCE 2>/dev/null | awk -F': ' '/SW Power Capping/{gsub(/ us/,"",$2); print $2; exit}')
print -r -- "$(date +%H:%M:%S) ${1:-snap} sm=$(q clocks.sm)MHz temp=$(q temperature.gpu)C draw=$(q power.draw)W throttle=[$active] powercap_us=$capus"
