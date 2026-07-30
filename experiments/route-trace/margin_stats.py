#!/usr/bin/env python3
"""Top-1 margin distribution from a llama-perplexity `--kl-divergence-base` file.

Why: the per-layer routing result held on wikitext and did not replicate on code, and the
proposed reason is that a routing perturbation can only flip a prediction where the top-2 are
close. That is a claim about the BASE distribution, so it can be tested without touching the
GPU: if it is right, the fraction of tokens whose top-1 margin is below some threshold should
differ between the two corpora by about the ratio of their observed flip rates.

File layout (from tools/perplexity/perplexity.cpp):
  "_logits_" | i32 n_ctx | i32 n_vocab | i32 n_chunk | i32 tokens[n_chunk*n_ctx]
  then per chunk: n_tok = n_ctx - 1 - n_ctx/2 records of nv = 2*((n_vocab+1)/2)+4 u16,
  each record = f32 scale, f32 min_log_prob, then n_vocab u16 q[i]
  with logit[i] - min_logit = q[i]*scale (clipped at max_logit - 16).
"""
import argparse
import struct

import numpy as np


def open_kld(path):
    f = open(path, "rb")
    magic = f.read(8)
    assert magic == b"_logits_", magic
    n_ctx, n_vocab, n_chunk = struct.unpack("<iii", f.read(12))
    tok_bytes = n_chunk * n_ctx * 4
    nv = 2 * ((n_vocab + 1) // 2) + 4
    n_tok = n_ctx - 1 - n_ctx // 2
    base = 20 + tok_bytes
    return f, dict(n_ctx=n_ctx, n_vocab=n_vocab, n_chunk=n_chunk, nv=nv,
                   n_tok=n_tok, base=base, rec=nv * 2)


def margins(path, n_sample, seed=0):
    f, m = open_kld(path)
    total = m["n_chunk"] * m["n_tok"]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(total, size=min(n_sample, total), replace=False))
    out = np.empty(len(idx), dtype=np.float64)
    for j, i in enumerate(idx):
        f.seek(m["base"] + int(i) * m["rec"])
        buf = f.read(m["rec"])
        scale, _minlp = struct.unpack("<ff", buf[:8])
        q = np.frombuffer(buf, dtype=np.uint16, offset=8, count=m["n_vocab"])
        # only the top two matter, and they sit at the top of the quantised range
        top2 = np.partition(q, -2)[-2:]
        out[j] = float(top2[1] - top2[0]) * scale
    f.close()
    return out, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kld", nargs="+")
    ap.add_argument("-n", "--n-sample", type=int, default=4000)
    ap.add_argument("--flip-rate", type=float, nargs="*", default=[],
                    help="observed top-1 flip rate per file, as a fraction; the first file's "
                         "rate calibrates a margin threshold and the rest are predicted")
    args = ap.parse_args()

    data = {}
    for p in args.kld:
        mg, meta = margins(p, args.n_sample)
        data[p] = mg
        qs = np.percentile(mg, [1, 5, 10, 25, 50, 75])
        print(f"{p}\n  n={len(mg)} of {meta['n_chunk']*meta['n_tok']} tokens, "
              f"n_vocab={meta['n_vocab']}")
        print("  margin (logit units) pct  1%%:%.4f  5%%:%.4f  10%%:%.4f  25%%:%.4f  "
              "50%%:%.4f  75%%:%.4f" % tuple(qs))
        for t in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0):
            print(f"    P(margin < {t:>4}) = {100.0*np.mean(mg < t):6.2f}%")

    if len(args.flip_rate) >= 2 and len(args.kld) >= 2:
        ref, rest = args.kld[0], args.kld[1:]
        thr = float(np.quantile(data[ref], args.flip_rate[0]))
        print(f"\ncalibrating on {ref}: flip rate {100*args.flip_rate[0]:.3f}% "
              f"=> margin threshold {thr:.5f}")
        for p, obs in zip(rest, args.flip_rate[1:]):
            pred = float(np.mean(data[p] < thr))
            print(f"  {p}\n    predicted flip rate {100*pred:.3f}%   "
                  f"observed {100*obs:.3f}%   ratio pred/obs {pred/obs if obs else float('nan'):.2f}")


if __name__ == "__main__":
    main()
