#!/usr/bin/env python3
"""Build a .hsfy token->expert table for LLAMA_DSV4_HASHIFY.

Two sources:
  --from-gguf MODEL --layers 0,1,2
      read this model's own ffn_gate_tid2eid tensors. Full vocab coverage, so
      hashifying those layers must reproduce the unmodified model exactly - that is
      the self-test for the runtime override.

  --from-traces T.rtrc [T2.rtrc ...] --layers 3-42
      distill a static table from route traces: per (layer, token) take the
      most-frequently-selected n_expert_used experts. Tokens absent from the corpus
      get the layer's globally most-frequent expert set, so the runtime never has to
      handle a miss. Coverage is reported per layer - it is a headline number, since
      an uncovered token is routed by a generic set rather than its own.

Layout (little-endian): u32 magic "HSFY", u32 version=1, u32 n_vocab,
u32 n_expert_used, u32 n_layers, then per layer { i32 il, i32 ids[n_expert_used*n_vocab] }
"""
import argparse
import struct
import sys
from collections import Counter, defaultdict

MAGIC = 0x59465348


def parse_layers(spec):
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


def from_gguf(model_path, layers):
    sys.path.insert(0, "/home/trevor/Projects/llama.cpp/gguf-py")
    from gguf import GGUFReader

    # a split model keeps blk tensors in later shards, so scan the whole set
    import glob
    import re
    paths = [model_path]
    if re.search(r"-\d{5}-of-\d{5}\.gguf$", model_path):
        paths = sorted(glob.glob(re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", model_path)))
    tensors = {}
    for p in paths:
        for t in GGUFReader(p).tensors:
            tensors.setdefault(t.name, t)
    tables = {}
    n_vocab = n_used = None
    for il in layers:
        name = f"blk.{il}.ffn_gate_tid2eid.weight"
        t = tensors.get(name)
        if t is None:
            print(f"error: {name} not in {model_path} "
                  f"(only the first hash_layer_count layers have one)", file=sys.stderr)
            sys.exit(1)
        # gguf reports shape reversed relative to ggml ne[]
        dims = [int(d) for d in t.shape]
        used, vocab = min(dims), max(dims)
        data = t.data.reshape(-1).astype("int32")
        assert data.size == used * vocab, (data.size, used, vocab)
        n_vocab, n_used = vocab, used
        tables[il] = data.tolist()
        print(f"layer {il}: tid2eid {used} x {vocab} read from gguf")
    return n_vocab, n_used, tables


def from_traces(paths, layers, n_vocab, n_expert=256):
    sys.path.insert(0, "/home/trevor/Projects/llama.cpp/experiments/route-trace")
    from rtrc_analyze import read_records

    counts = defaultdict(lambda: defaultdict(Counter))  # il -> token -> Counter
    globals_ = defaultdict(Counter)                     # il -> Counter
    n_bad_range = {}
    n_used = 0
    for p in paths:
        for _, toks, _, lys in read_records(p):
            for il, rows in lys.items():
                if il not in layers:
                    continue
                for t, s in zip(toks, rows):
                    if s is None:
                        continue
                    # a corrupted readback can yield distinct-but-invalid ids, which the
                    # trace-level "k distinct" rule cannot catch; range-check here so
                    # garbage never reaches a table the GPU will index with
                    if min(s) < 0 or max(s) >= n_expert:
                        n_bad_range[il] = n_bad_range.get(il, 0) + 1
                        continue
                    n_used = max(n_used, len(s))
                    counts[il][t].update(s)
                    globals_[il].update(s)

    tables = {}
    for il in layers:
        if il not in counts:
            print(f"warn: layer {il} has no trace data - skipped", file=sys.stderr)
            continue
        fallback = [e for e, _ in globals_[il].most_common(n_used)]
        while len(fallback) < n_used:
            fallback.append(fallback[-1] if fallback else 0)

        flat = fallback * n_vocab
        n_cov = 0
        for tok, c in counts[il].items():
            if tok < 0 or tok >= n_vocab:
                continue
            top = [e for e, _ in c.most_common(n_used)]
            for e in fallback:                     # pad from the fallback set...
                if len(top) >= n_used:
                    break
                if e not in top:
                    top.append(e)
            while len(top) < n_used:               # ...and never leave a short row,
                top.append(top[-1])                # which would shrink the flat list
            assert len(top) == n_used
            flat[tok * n_used:(tok + 1) * n_used] = top
            n_cov += 1
        assert len(flat) == n_vocab * n_used, (il, len(flat))
        assert min(flat) >= 0 and max(flat) < n_expert, (il, min(flat), max(flat))
        tables[il] = flat
        print(f"layer {il}: {n_cov} tokens covered ({100.0*n_cov/n_vocab:.1f}% of vocab), "
              f"fallback set {fallback}")
    if n_bad_range:
        print(f"warn: skipped rows with out-of-range ids (layer: n): "
              f"{sorted(n_bad_range.items(), key=lambda kv: -kv[1])[:6]}", file=sys.stderr)
    return n_used, tables


def write(path, n_vocab, n_used, tables):
    with open(path, "wb") as f:
        f.write(struct.pack("<IIIII", MAGIC, 1, n_vocab, n_used, len(tables)))
        for il in sorted(tables):
            f.write(struct.pack("<i", il))
            f.write(struct.pack(f"<{len(tables[il])}i", *tables[il]))
    size = 20 + sum(4 + 4 * len(v) for v in tables.values())
    print(f"wrote {path}: {len(tables)} layers, {size/1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--layers", required=True, help="e.g. 0,1,2 or 3-42")
    ap.add_argument("--from-gguf", metavar="MODEL")
    ap.add_argument("--from-traces", nargs="+", metavar="RTRC")
    ap.add_argument("--n-expert", type=int, default=256, help="routed expert count")
    ap.add_argument("--n-vocab", type=int, default=129280,
                    help="vocab size for trace-built tables (default DS4-Flash 129280)")
    args = ap.parse_args()

    layers = parse_layers(args.layers)
    if args.from_gguf:
        n_vocab, n_used, tables = from_gguf(args.from_gguf, layers)
    elif args.from_traces:
        n_used, tables = from_traces(args.from_traces, set(layers), args.n_vocab, args.n_expert)
        n_vocab = args.n_vocab
    else:
        ap.error("need --from-gguf or --from-traces")

    if not tables:
        print("error: no layers built", file=sys.stderr)
        sys.exit(1)
    write(args.out, n_vocab, n_used, tables)


if __name__ == "__main__":
    main()
