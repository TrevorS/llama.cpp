#!/usr/bin/env python3
"""Compare a llama-perplexity --save-all-logits dump against the API reference.

Reads <arm>.kld (format: "_logits_" magic, u32 n_ctx, i32 n_vocab, i32
n_chunk, i32 tokens[n_chunk*n_ctx], then per chunk (n_ctx-1-first) rows of
uint16[nv], nv = 2*((n_vocab+1)/2)+4; row header = 2 floats (scale,
min_log_prob), logprob_i = min_log_prob + q_i*scale, q==0 means <= floor)
plus meta.json from build_corpus.py.

Dump row r of chunk c holds the distribution AFTER position c*n_ctx+first+r,
i.e. it predicts the token at dump position c*n_ctx+first+r+1. A unit's
file-token index t sits at dump position t+1 (global BOS shift), so the API
token at unit_tok_idx u (file tok unit_base+u) is predicted by row
(unit_base + u + 1) - (c*n_ctx + first) - 1 of its chunk.

Metrics per comparable position: our logprob of the API token, top-1
agreement, API-token rank in our distribution, truncated-KL and top-20
overlap vs the API's top-20.

Usage: analyze.py <arm>.kld [--meta meta.json] [--per-unit]
"""
import argparse
import json
import os
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def read_dump(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == b"_logits_", f"bad magic {magic!r}"
        (n_ctx,) = struct.unpack("<I", f.read(4))
        n_vocab, n_chunk = struct.unpack("<ii", f.read(8))
        tokens = np.fromfile(f, dtype=np.int32, count=n_chunk * n_ctx)
        nv = 2 * ((n_vocab + 1) // 2) + 4
        rows_per_chunk = n_ctx - 1 - n_ctx // 2
        raw = np.fromfile(f, dtype=np.uint16)
    expect = n_chunk * rows_per_chunk * nv
    assert raw.size == expect, f"row payload {raw.size} != {expect}"
    raw = raw.reshape(n_chunk * rows_per_chunk, nv)
    return n_ctx, n_vocab, n_chunk, tokens, raw, nv, rows_per_chunk


def row_logprobs(raw_row, n_vocab):
    scale, min_lp = struct.unpack("<ff", raw_row[:4].tobytes())
    q = raw_row[4:4 + n_vocab].astype(np.float32)
    lp = min_lp + q * scale
    return lp, min_lp  # q==0 entries sit at the floor min_lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--meta", default=os.path.join(HERE, "meta.json"))
    ap.add_argument("--per-unit", action="store_true")
    args = ap.parse_args()

    with open(args.meta) as f:
        meta = json.load(f)
    n_ctx_m, first = meta["n_ctx"], meta["first"]
    n_ctx, n_vocab, n_chunk, tokens, raw, _nv, rpc = read_dump(args.dump)
    assert n_ctx == n_ctx_m, f"n_ctx mismatch dump {n_ctx} vs meta {n_ctx_m}"

    agg = {"n": 0, "top1": 0, "lp_ours": 0.0, "lp_api": 0.0,
           "rank_sum": 0, "rank_le5": 0, "kl20": 0.0, "kl20_n": 0,
           "ov20": 0.0, "ov20_n": 0, "floor_hits": 0}
    unit_rows = []

    for unit in meta["units"]:
        c = unit["chunk"]
        if c >= n_chunk:
            print(f"[{unit['file']}] chunk {c} beyond dump ({n_chunk}), skipping")
            continue
        base = unit["unit_base_file_tok"]
        u = {"file": unit["file"], "n": 0, "top1": 0, "lp_ours": 0.0, "lp_api": 0.0}
        for cp in unit["comparable"]:
            dump_pos = base + cp["unit_tok_idx"] + 1     # +1: global BOS
            r = dump_pos - (c * n_ctx + first) - 1       # row predicting dump_pos
            if r < 0 or r >= rpc:
                continue
            # token-identity check: the dump's token stream must match ours
            if tokens[dump_pos] != cp["id"]:
                print(f"[{unit['file']}] token mismatch at dump {dump_pos}: "
                      f"{tokens[dump_pos]} vs {cp['id']} — corpus drift, skipping unit")
                break
            lp, floor = row_logprobs(raw[c * rpc + r], n_vocab)
            ours = float(lp[cp["id"]])
            if ours <= floor + 1e-9:
                agg["floor_hits"] += 1
            argmax = int(np.argmax(lp))
            rank = int((lp > ours).sum()) + 1
            agg["n"] += 1
            u["n"] += 1
            agg["top1"] += argmax == cp["id"]
            u["top1"] += argmax == cp["id"]
            agg["lp_ours"] += ours
            u["lp_ours"] += ours
            agg["lp_api"] += cp["api_logprob"]
            u["lp_api"] += cp["api_logprob"]
            agg["rank_sum"] += rank
            agg["rank_le5"] += rank <= 5
            top = [t for t in cp["top"] if t["id"] >= 0 and t["logprob"] > -9998]
            if len(top) >= 5:
                api_lp = np.array([t["logprob"] for t in top])
                api_p = np.exp(api_lp)
                our_lp_t = lp[[t["id"] for t in top]]
                agg["kl20"] += float((api_p * (api_lp - our_lp_t)).sum())
                agg["kl20_n"] += 1
                our_top = set(np.argpartition(lp, -len(top))[-len(top):].tolist())
                agg["ov20"] += len(our_top & {t["id"] for t in top}) / len(top)
                agg["ov20_n"] += 1
        unit_rows.append(u)

    n = max(1, agg["n"])
    print(f"\n== {os.path.basename(args.dump)} vs API ({agg['n']} positions) ==")
    print(f"top-1 agreement      : {100.0*agg['top1']/n:.2f}%")
    print(f"mean logprob (ours)  : {agg['lp_ours']/n:.4f}")
    print(f"mean logprob (API)   : {agg['lp_api']/n:.4f}")
    print(f"delta xent (ours-API): {(agg['lp_api']-agg['lp_ours'])/n:.4f} nats/tok")
    print(f"API-token rank       : mean {agg['rank_sum']/n:.2f}, <=5: {100.0*agg['rank_le5']/n:.1f}%")
    if agg["kl20_n"]:
        print(f"truncated KL(api||us): {agg['kl20']/agg['kl20_n']:.4f} nats (top-20, {agg['kl20_n']} pos)")
        print(f"top-20 overlap       : {100.0*agg['ov20']/agg['ov20_n']:.1f}%")
    print(f"floor-clamped reads  : {agg['floor_hits']} (our logprob below 16-nat window)")
    if args.per_unit:
        for u in unit_rows:
            if u["n"]:
                print(f"  {u['file']}: n={u['n']} top1={100.0*u['top1']/u['n']:.1f}% "
                      f"lp_ours={u['lp_ours']/u['n']:.3f} lp_api={u['lp_api']/u['n']:.3f}")


if __name__ == "__main__":
    main()
