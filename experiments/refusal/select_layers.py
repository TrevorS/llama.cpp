#!/usr/bin/env python3
"""Transform a refusal direction with gemma/ds4 techniques, offline (no re-capture).

Reads an existing unit-normalized direction .f32 + its .json (sep_per_layer) and:
  * LAYER SELECTION (gemma --top-pcts): keep only the top-K layers by separation,
    zero the rest (a zeroed row = no ablation at that layer).
  * SEPARATION WEIGHTING (ds4 raw-norm): scale each kept layer by sep/median_sep,
    so high-signal decision layers get more ablation, low-signal layers less.
Emits a new control-vector .gguf ready to load.
"""
import argparse, json
import numpy as np

N_LAYER, N_EMBD = 43, 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-f32", required=True)
    ap.add_argument("--in-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-pct", type=float, default=50.0, help="keep this %% of layers by separation")
    ap.add_argument("--weight", choices=["unit", "sep"], default="sep",
                    help="unit = equal per layer; sep = scale by sep/median (ds4-style)")
    args = ap.parse_args()

    dirs = np.fromfile(args.in_f32, dtype=np.float32).reshape(N_LAYER, N_EMBD)
    meta = json.load(open(args.in_json))
    sep = np.array(meta["sep_per_layer"], dtype=np.float32)

    # rank layers by separation, keep top-K
    k = max(1, int(round(N_LAYER * args.top_pct / 100.0)))
    keep = set(np.argsort(-sep)[:k].tolist())

    out = np.zeros_like(dirs)
    med = float(np.median(sep[list(keep)]))
    for l in range(N_LAYER):
        if l not in keep:
            continue
        if args.weight == "sep":
            out[l] = dirs[l] * (sep[l] / med)   # concentrate on high-signal layers
        else:
            out[l] = dirs[l]

    kept_sorted = sorted(keep)
    print(f"kept {k}/{N_LAYER} layers (top {args.top_pct}% by sep): {kept_sorted}")
    print(f"weight={args.weight}, median sep among kept={med:.2f}")

    import gguf
    w = gguf.GGUFWriter(args.out + ".gguf", "controlvector")
    w.add_string("controlvector.model_hint", "deepseek4")
    w.add_uint32("controlvector.layer_count", N_LAYER)
    for l in range(N_LAYER):
        w.add_tensor(f"direction.{l + 1}", out[l])
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {args.out}.gguf")


if __name__ == "__main__":
    main()
