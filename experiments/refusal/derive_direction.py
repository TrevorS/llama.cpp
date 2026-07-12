#!/usr/bin/env python3
"""Derive a DS4 refusal-ablation direction (43 x 4096 f32) from captured activations.

Stage 2 of the pipeline (capture -> DERIVE -> apply). Pure numpy, no GPU.

Input : an .npz with G (n, 43, 4096) harmful + B (n, 43, 4096) harmless final-token
        residual activations (from the llama.cpp cb_eval capture tool, or the legacy
        ds4 capture for math validation).
Output: <out>.f32  raw 43x4096 f32 row-major (inspection / ds4 compat)
        <out>.gguf  llama.cpp control-vector (tensors direction.<il>, unit-norm)
        <out>.json  metadata

Direction[l] = normalize( winsorized_mean(G[:,l]) - winsorized_mean(B[:,l]) ),
then Gram-Schmidt orthogonalized vs the harmless mean (remove refusal-specific
component only), unit-normalized per layer. Positive apply scale SUPPRESSES refusal.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "gguf-py"))

N_LAYER, N_EMBD = 43, 4096


def winsorized_mean(x, p=0.995):
    # x: (n, d) -> (d,) mean after clipping each dim to its [1-p, p] quantiles
    lo = np.quantile(x, 1 - p, axis=0)
    hi = np.quantile(x, p, axis=0)
    return np.clip(x, lo, hi).mean(axis=0)


def gram_schmidt(d, ref):
    # remove the component of d along ref (double pass for numerical stability)
    for _ in range(2):
        r = ref / (np.linalg.norm(ref) + 1e-8)
        d = d - np.dot(d, r) * r
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True, help=".npz with G and B arrays")
    ap.add_argument("--out", required=True, help="output basename")
    ap.add_argument("--winsorize", type=float, default=0.995)
    ap.add_argument("--no-orthogonalize", action="store_true")
    args = ap.parse_args()

    z = np.load(args.acts)
    G, B = z["G"].astype(np.float32), z["B"].astype(np.float32)
    assert G.shape[1:] == (N_LAYER, N_EMBD), f"bad G shape {G.shape}"
    assert B.shape[1:] == (N_LAYER, N_EMBD), f"bad B shape {B.shape}"
    print(f"loaded G={G.shape} B={B.shape}")

    dirs = np.zeros((N_LAYER, N_EMBD), np.float32)
    seps = []
    for l in range(N_LAYER):
        mg = winsorized_mean(G[:, l], args.winsorize)
        mb = winsorized_mean(B[:, l], args.winsorize)
        d = mg - mb
        seps.append(float(np.linalg.norm(d)))
        if not args.no_orthogonalize:
            d = gram_schmidt(d, mb)
        n = np.linalg.norm(d)
        dirs[l] = d / n if n > 1e-8 else d

    # cross-layer alignment (low = layers carry distinct directions, good)
    cos = []
    for l in range(N_LAYER - 1):
        cos.append(float(np.dot(dirs[l], dirs[l + 1])))
    adj_cos = float(np.median(np.abs(cos)))

    dirs.tofile(args.out + ".f32")

    meta = dict(format="llamacpp-refusal-direction-v1", shape=[N_LAYER, N_EMBD],
                pairs=int(min(G.shape[0], B.shape[0])), winsorize=args.winsorize,
                orthogonalize=not args.no_orthogonalize, adj_cos_median=adj_cos,
                sep_per_layer=[round(s, 4) for s in seps])
    json.dump(meta, open(args.out + ".json", "w"), indent=2)

    # write llama.cpp control-vector gguf (unit dirs; scale applied at inference)
    try:
        import gguf
        w = gguf.GGUFWriter(args.out + ".gguf", "controlvector")
        w.add_string("controlvector.model_hint", "deepseek4")
        w.add_uint32("controlvector.layer_count", N_LAYER)
        # llama.cpp applies direction.<il> AT layer il, for il in [1, n_layer-1]
        # (layer 0 is never steered; index 0 is rejected). So direction.il must be
        # the vector DERIVED at layer il — a straight il->il map, NOT il+1. The old
        # +1 shift put every layer's vector one layer late and dropped layer 42's
        # (the </think> decision layer, largest separation) onto direction.43 where
        # nothing applies it. That off-by-one alone collapsed the flip rate to ~5%.
        for l in range(1, N_LAYER):
            w.add_tensor(f"direction.{l}", dirs[l])
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        gguf_ok = True
    except Exception as e:
        gguf_ok = False
        print(f"gguf write skipped: {e}")

    print(f"wrote {args.out}.f32 / .json" + (" / .gguf" if gguf_ok else ""))
    print(f"adj_cos_median={adj_cos:.4f}  sep range {min(seps):.2f}..{max(seps):.2f}")
    print("per-layer sep:", " ".join(f"{s:.1f}" for s in seps))


if __name__ == "__main__":
    main()
