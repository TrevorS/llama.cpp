#!/usr/bin/env python3
"""Collect finished kld_leg.sh runs into one comparison table.

Reports each leg against the null (the unmodified model re-run against its own stored
logits), because that null is not zero: batch splitting and reduction order alone move
~0.04% of top-1 predictions and put a small positive bias on the paired PPL ratio. Any
effect at or below the null row is not measurable with this instrument.
"""
import glob
import os
import re
import sys

DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/trevor/.cache/hashify-kld"

PATS = {
    "ppl":   re.compile(r"^Mean PPL\(Q\)\s+:\s+([\d.]+)"),
    "lnr":   re.compile(r"^Mean ln\(PPL\(Q\)/PPL\(base\)\)\s+:\s+([-\d.]+)\s+±\s+([\d.]+)"),
    "kld":   re.compile(r"^Mean\s+KLD:\s+([-\d.]+)\s+±\s+([\d.]+)"),
    "rms":   re.compile(r"^RMS Δp\s+:\s+([\d.]+)"),
    "top":   re.compile(r"^Same top p:\s+([\d.]+)\s+±\s+([\d.]+)"),
}


def parse(path):
    out = {}
    with open(path, errors="replace") as f:
        for line in f:
            for k, p in PATS.items():
                m = p.match(line)
                if m:
                    out[k] = tuple(float(x) for x in m.groups())
    return out


rows = []
for path in sorted(glob.glob(os.path.join(DIR, "*.log"))):
    r = parse(path)
    if "top" not in r:
        continue  # still running or aborted
    rows.append((os.path.basename(path)[:-4], r))

null = dict(rows).get("null")
print(f"{'leg':<16} {'PPL':>8} {'ln ratio':>18} {'KLD':>20} {'RMS dp':>8} {'same top-1':>16}")
for name, r in rows:
    ln = f"{r['lnr'][0]:.5f} ±{r['lnr'][1]:.5f}"
    kld = f"{r['kld'][0]:.6f} ±{r['kld'][1]:.6f}"
    top = f"{r['top'][0]:.3f} ±{r['top'][1]:.3f}"
    print(f"{name:<16} {r['ppl'][0]:>8.4f} {ln:>18} {kld:>20} {r['rms'][0]:>7.3f}% {top:>16}")
if null:
    print(f"\nnull floor: KLD {null['kld'][0]:.6f}, same top-1 {null['top'][0]:.3f}%, "
          f"ln ratio bias {null['lnr'][0]:+.5f} - treat anything inside these as unmeasured")
