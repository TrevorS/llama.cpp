#!/usr/bin/env bash
# Run a list of hashification legs back to back with a cooldown between each.
# Each leg is short (100 chunks of 512 at ub 2048, ~2 min of GPU) and the model reload
# between legs is disk-bound, so the duty cycle stays low without extra pacing - but the
# explicit cooldown is what keeps a chain from becoming one long accumulation.
#
# Usage: kld_curve.sh <cooldown_s> <name:hsfy:layers> [<name:hsfy:layers> ...]
#   hsfy may be "none"; layers may be "-" for the whole table.
#   corpus/base default to wiki inside kld_leg.sh; override with CORPUS= and BASE= env.
set -u
COOL=${1:?usage: kld_curve.sh <cooldown_s> <name:hsfy:layers> ...}
shift
DIR=$(dirname "$0")
CORPUS=${CORPUS:-/home/trevor/models/datasets/wiki.test.raw}
BASE=${BASE:-/home/trevor/models/ds4/kld/base-udiq3-c512-n100.kld}
CHUNKS=${CHUNKS:-100}
DUTY=${DUTY:-85}

for spec in "$@"; do
    IFS=: read -r name hsfy layers <<< "$spec"
    echo "### leg $name  ($(date +%H:%M:%S))"
    "$DIR/kld_leg.sh" "$name" "$hsfy" "$layers" "$CORPUS" "$BASE" "$CHUNKS" "$DUTY"
    echo "### cooldown ${COOL}s"
    sleep "$COOL"
done
echo "### chain done ($(date +%H:%M:%S))"
