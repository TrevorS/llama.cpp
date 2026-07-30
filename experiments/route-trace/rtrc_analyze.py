#!/usr/bin/env python3
"""Analyze LLAMA_ROUTE_TRACE spools (.rtrc).

File format (little-endian), written by llama-context.cpp:
  header: u32 magic "RTRC" (0x43525452), u32 version (1)
  record type 0 (tokens): u8 0, u32 seq, u32 n_tokens, i32 tok[n], i32 pos[n]
  record type 1 (layer):  u8 1, u32 seq, i32 il, u32 k, u32 n_tokens, i32 ids[k*n]

Analyses:
  --summary                 record counts, layers, token volume
  --hash-check N            verify layers < N are exactly token-determined (DS4: N=3)
  --marginal                per-layer token-marginal top-k coverage, 25/75 split
                            (Sem-MoE App A.1 protocol, lifted to per-layer)
  --adjacent                adjacent-position expert-set overlap per layer,
                            within-eval (chain/verify rows) vs cross-eval
  --correspond DFT.rtrc     MTP-draft router vs target per-layer routing:
                            confusion-mapped overlap + raw agreement
"""
import argparse
import random
import struct
import sys
from collections import Counter, defaultdict

MAGIC = 0x43525452


def read_records(path, n_expert=256):
    """Yield (seq, toks, poss, {il: [set(expert_ids) per row]}) grouped per eval."""
    with open(path, "rb") as f:
        hdr = f.read(8)
        magic, ver = struct.unpack("<II", hdr)
        assert magic == MAGIC, f"{path}: bad magic {magic:#x}"
        assert ver == 1, f"{path}: unsupported version {ver}"

        # a spool from a crashed run ends mid-record; treat a short read as EOF
        def rd(n):
            b = f.read(n)
            if len(b) < n:
                raise EOFError
            return b

        cur = None  # (seq, toks, poss, layers)
        dropped = {}
        try:
            while True:
                tb = f.read(1)
                if not tb:
                    break
                (rtype,) = struct.unpack("<B", tb)
                if rtype == 0:
                    (seq, n) = struct.unpack("<II", rd(8))
                    if n == 0:
                        # a hard crash leaves the file zero-padded to a block boundary;
                        # NUL bytes parse as empty token records - stop here
                        print(f"note: {path} has a NUL-padded tail (power-loss write) - "
                              f"stopping at the last complete eval", file=sys.stderr)
                        break
                    toks = struct.unpack(f"<{n}i", rd(4 * n))
                    poss = struct.unpack(f"<{n}i", rd(4 * n))
                    if cur is not None:
                        yield cur
                    cur = (seq, toks, poss, {})
                elif rtype == 1:
                    (seq, il, k, n) = struct.unpack("<IiII", rd(16))
                    ids = struct.unpack(f"<{k * n}i", rd(4 * k * n))
                    if cur is None or cur[0] != seq:
                        print(f"warn: layer record seq {seq} without token block", file=sys.stderr)
                        continue
                    # top-k selection always yields k DISTINCT in-range expert ids, so a
                    # row failing either test is a readback that missed the real write (this happened under
                    # CUDA graph replay, before the hook forced graphs off). Keep positional
                    # alignment with the token list and mark such rows None so the analyses
                    # skip them instead of letting expert 0 dominate the statistics.
                    rows = []
                    n_bad = 0
                    for i in range(n):
                        chunk = ids[i * k:(i + 1) * k]
                        if len(set(chunk)) < k or min(chunk) < 0 or max(chunk) >= n_expert:
                            rows.append(None)
                            n_bad += 1
                        else:
                            rows.append(frozenset(chunk))
                    if n_bad:
                        dropped[il] = dropped.get(il, 0) + n_bad
                    if n_bad == n:
                        continue
                    cur[3][il] = rows
                else:
                    raise ValueError(f"{path}: unknown record type {rtype}")
        except EOFError:
            print(f"note: {path} ends in a truncated record (crashed run) - "
                  f"dropping the final partial eval", file=sys.stderr)
            cur = None
        if cur is not None:
            yield cur
        if dropped:
            worst = sorted(dropped.items(), key=lambda kv: -kv[1])[:8]
            print(f"warn: dropped degenerate rows (layer: n_rows): {worst}"
                  f"{' ...' if len(dropped) > 8 else ''} - those layers are under-sampled",
                  file=sys.stderr)


def load(path):
    evals = list(read_records(path))
    print(f"{path}: {len(evals)} evals, "
          f"{sum(len(e[1]) for e in evals)} token-rows, "
          f"layers: {sorted({il for e in evals for il in e[3]})[:5]}..."
          f"{sorted({il for e in evals for il in e[3]})[-3:]}")
    return evals


def a_summary(evals):
    n_rows = sum(len(e[1]) for e in evals)
    layers = sorted({il for e in evals for il in e[3]})
    ks = {len(next(iter(e[3][il]))) for e in evals for il in e[3] if e[3][il]}
    by_size = Counter(len(e[1]) for e in evals)
    print(f"evals={len(evals)} token-rows={n_rows} n_layers={len(layers)} k={sorted(ks)}")
    print(f"ubatch-size histogram (top 8): {by_size.most_common(8)}")


def a_hash_check(evals, n_hash):
    """Layers < n_hash must map token id -> one expert set, always."""
    for il in range(n_hash):
        tok2sets = defaultdict(set)
        for _, toks, _, layers in evals:
            rows = layers.get(il)
            if not rows:
                continue
            for t, s in zip(toks, rows):
                if s is not None:
                    tok2sets[t].add(s)
        bad = {t: len(ss) for t, ss in tok2sets.items() if len(ss) > 1}
        status = "EXACT" if not bad else f"VIOLATIONS: {len(bad)} tokens"
        print(f"layer {il}: {len(tok2sets)} distinct tokens -> {status}")


def a_marginal(evals, split=0.25):
    """Per-layer: build static token->top-k table on first `split` of evals,
    measure F1 of that table against the rest (Sem-MoE protocol, per-layer).
    kbar = mean size of the ACTUAL expert set (< k means duplicate ids in the
    row, which caps precision and must be read before trusting deep layers)."""
    n_train = max(1, int(len(evals) * split))
    train, test = evals[:n_train], evals[n_train:]
    print(f"train evals={len(train)} test evals={len(test)} split={split}")
    print(f"{'layer':>5} {'toks':>7} {'cov':>6} {'kbar':>5} {'P':>6} {'R':>6} {'F1':>6}")

    layers = sorted({il for e in evals for il in e[3]})
    for il in layers:
        expert_count = defaultdict(Counter)
        k_seen = 0
        for _, toks, _, lys in train:
            rows = lys.get(il)
            if not rows:
                continue
            for t, s in zip(toks, rows):
                if s is None:
                    continue
                expert_count[t].update(s)
                k_seen = max(k_seen, len(s))
        table = {t: frozenset(e for e, _ in c.most_common(k_seen))
                 for t, c in expert_count.items()}

        tp = fp = fn = n_cov = n_tot = k_sum = 0
        for _, toks, _, lys in test:
            rows = lys.get(il)
            if not rows:
                continue
            for t, s in zip(toks, rows):
                if s is None:
                    continue
                n_tot += 1
                k_sum += len(s)
                pred = table.get(t)
                if pred is None:
                    fn += len(s)
                    continue
                n_cov += 1
                tp += len(pred & s)
                fp += len(pred - s)
                fn += len(s - pred)
        if n_tot == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"{il:>5} {len(table):>7} {n_cov/n_tot:>6.3f} {k_sum/n_tot:>5.2f} "
              f"{prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")


def a_adjacent(evals):
    """Adjacent-position expert-set overlap per layer: within one eval
    (chain/verify rows, drafted tokens) vs across evals (committed sequence)."""
    print(f"{'layer':>5} {'within-eval':>12} {'n':>8} {'cross-eval':>11} {'n':>8}")
    layers = sorted({il for e in evals for il in e[3]})
    # committed adjacency: last-seen expert set per position
    for il in layers:
        w_num = w_den = 0
        last_by_pos = {}
        for _, _toks, poss, lys in evals:
            rows = lys.get(il)
            if not rows:
                continue
            for i in range(len(poss)):
                if rows[i] is not None:
                    last_by_pos[poss[i]] = rows[i]
            for i in range(1, len(poss)):
                if poss[i] == poss[i - 1] + 1 and rows[i] is not None and rows[i - 1] is not None:
                    a, b = rows[i - 1], rows[i]
                    w_num += len(a & b) / len(a | b)
                    w_den += 1
        c_num = c_den = 0
        pos_sorted = sorted(last_by_pos)
        for p0, p1 in zip(pos_sorted, pos_sorted[1:]):
            if p1 == p0 + 1:
                a, b = last_by_pos[p0], last_by_pos[p1]
                c_num += len(a & b) / len(a | b)
                c_den += 1
        w = w_num / w_den if w_den else float("nan")
        c = c_num / c_den if c_den else float("nan")
        print(f"{il:>5} {w:>12.4f} {w_den:>8} {c:>11.4f} {c_den:>8}")


def a_union(evals, n_expert=256):
    """Measured expert-union size across the rows of one multi-token eval (the
    speculative verify batch), against the independence prediction used by the
    published union laws: N(t) = E * (1 - ((E-K)/E)^t)."""
    by_t = defaultdict(lambda: defaultdict(list))  # t -> il -> [union sizes]
    k_obs = 6
    for _, toks, _, lys in evals:
        for il, rows in lys.items():
            good = [r for r in rows if r is not None]
            t = len(good)
            if t < 2:
                continue
            k_obs = len(good[0])
            by_t[t][il].append(len(frozenset().union(*good)))
    print(f"{'t':>3} {'layers':>7} {'measured':>9} {'independ.':>10} {'ratio':>6} {'perfect':>8} {'n':>7}")
    for t in sorted(by_t):
        sizes = [v for il in by_t[t] for v in by_t[t][il]]
        if not sizes:
            continue
        meas = sum(sizes) / len(sizes)
        indep = n_expert * (1.0 - ((n_expert - k_obs) / n_expert) ** t)
        print(f"{t:>3} {len(by_t[t]):>7} {meas:>9.2f} {indep:>10.2f} "
              f"{meas/indep:>6.3f} {k_obs*t:>8} {len(sizes):>7}")


def a_correspond(evals_tgt, evals_dft):
    """Join target and draft traces on (pos, token); per target layer, compute
    (a) raw expert-id agreement, (b) confusion-argmax-mapped overlap between the
    draft router's set and the target layer's set."""
    dft_by_key = {}
    for _, toks, poss, lys in evals_dft:
        rows = lys.get(next(iter(lys))) if lys else None
        if rows is None:
            continue
        il0 = sorted(lys)[0]
        for t, p, s in zip(toks, poss, lys[il0]):
            if s is not None:
                dft_by_key[(p, t)] = s

    layers = sorted({il for e in evals_tgt for il in e[3]})
    print(f"draft rows: {len(dft_by_key)}")
    print(f"{'layer':>5} {'joined':>8} {'raw':>8} {'chance':>8} {'mapped':>10} {'shuffled':>10}")
    for il in layers:
        pairs = []
        for _, toks, poss, lys in evals_tgt:
            rows = lys.get(il)
            if not rows:
                continue
            for t, p, s in zip(toks, poss, rows):
                if s is None:
                    continue
                d = dft_by_key.get((p, t))
                if d is not None:
                    pairs.append((d, s))
        if not pairs:
            continue
        raw = sum(len(d & s) / len(s) for d, s in pairs) / len(pairs)

        def fit_and_score(ps):
            conf = defaultdict(Counter)
            for d, s in ps:
                for de in d:
                    conf[de].update(s)
            m = {de: c.most_common(1)[0][0] for de, c in conf.items()}
            return sum(len({m[de] for de in d} & s) / len(s) for d, s in ps) / len(ps)

        mapped = fit_and_score(pairs)
        # null: re-pair each draft set with a different target set and refit the mapping
        # the same way. The confusion mapping is fitted on the data it scores, so it
        # inflates by construction - only mapped >> shuffled means real correspondence.
        rng = random.Random(42)
        shuf_scores = []
        for _ in range(3):
            tgt = [s for _, s in pairs]
            rng.shuffle(tgt)
            shuf_scores.append(fit_and_score([(d, t) for (d, _), t in zip(pairs, tgt)]))
        shuffled = sum(shuf_scores) / len(shuf_scores)
        chance = sum(len(d) * len(s) / 256.0 / len(s) for d, s in pairs) / len(pairs)
        print(f"{il:>5} {len(pairs):>8} {raw:>8.4f} {chance:>8.4f} {mapped:>10.4f} {shuffled:>10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="target .rtrc file")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--hash-check", type=int, metavar="N")
    ap.add_argument("--marginal", action="store_true")
    ap.add_argument("--split", type=float, default=0.25,
                    help="fraction of evals used to build the token table (default 0.25)")
    ap.add_argument("--adjacent", action="store_true")
    ap.add_argument("--union", action="store_true")
    ap.add_argument("--correspond", metavar="DFT_RTRC")
    args = ap.parse_args()

    evals = load(args.trace)
    if args.summary:
        a_summary(evals)
    if args.hash_check is not None:
        a_hash_check(evals, args.hash_check)
    if args.marginal:
        a_marginal(evals, args.split)
    if args.adjacent:
        a_adjacent(evals)
    if args.union:
        a_union(evals)
    if args.correspond:
        a_correspond(evals, load(args.correspond))


if __name__ == "__main__":
    main()
