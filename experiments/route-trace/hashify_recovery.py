#!/usr/bin/env python3
"""Compare a distilled .hsfy table against ground truth for layers that really are hashed.

DS4-Flash's first `hash_layer_count` layers route by a trained token->expert table, so for
those layers the trace-distilled table has a known correct answer. Any disagreement on a
token the corpus actually covered is a bug in the trace/distil pipeline, not a modelling
error - which makes this the self-test for the whole distillation path. Tokens the corpus
never showed fall back to a generic set, and that miss rate is the confound to subtract
from the deeper-layer interventions.

Usage: hashify_recovery.py DISTILLED.hsfy TRUTH.hsfy
"""
import struct
import sys
from collections import Counter

MAGIC = 0x59465348


def read(path):
    with open(path, "rb") as f:
        magic, ver, n_vocab, n_used, n_layers = struct.unpack("<IIIII", f.read(20))
        assert magic == MAGIC and ver == 1, (hex(magic), ver)
        out = {}
        for _ in range(n_layers):
            il, = struct.unpack("<i", f.read(4))
            out[il] = struct.unpack(f"<{n_used * n_vocab}i", f.read(4 * n_used * n_vocab))
        return n_vocab, n_used, out


def main():
    dpath, tpath = sys.argv[1], sys.argv[2]
    nv_d, nu_d, dist = read(dpath)
    nv_t, nu_t, truth = read(tpath)
    assert (nv_d, nu_d) == (nv_t, nu_t), ((nv_d, nu_d), (nv_t, nu_t))

    for il in sorted(set(dist) & set(truth)):
        d, t = dist[il], truth[il]
        # the distilled table pre-fills every uncovered token with one global fallback set,
        # so the modal row identifies the fallback and everything else is covered
        rows = [tuple(d[i * nu_d:(i + 1) * nu_d]) for i in range(nv_d)]
        fallback = Counter(rows).most_common(1)[0][0]
        exact = same_set = covered = 0
        for i, row in enumerate(rows):
            if row == fallback:
                continue
            covered += 1
            tr = tuple(t[i * nu_d:(i + 1) * nu_d])
            if row == tr:
                exact += 1
            if set(row) == set(tr):
                same_set += 1
        # how many of the fallback rows would have been right anyway
        fb_hit = sum(1 for i in range(nv_d)
                     if rows[i] == fallback
                     and set(fallback) == set(t[i * nu_d:(i + 1) * nu_d]))
        print(f"layer {il}: covered {covered} ({100.0*covered/nv_d:.1f}% of vocab)  "
              f"exact-order {100.0*exact/max(covered,1):.2f}%  "
              f"same-set {100.0*same_set/max(covered,1):.2f}%  "
              f"fallback-rows-correct {fb_hit}/{nv_d-covered}")


if __name__ == "__main__":
    main()
