#!/usr/bin/env python3
"""Surgically remove the "I"-subject-token entanglement from a refusal direction.

The scale-2 tic ("I is / I are" instead of "It is / You are") is the ablation
pushing the next-token logit for the literal token "I". That push is governed by
<dir_l, u_I> where u_I is the UNembedding row for "I", mapped back into residual
space through the final RMSNorm gain (diag(output_norm.weight)). Zero that inner
product and the ablation can no longer move the "I" logit, while the semantic
refusal component of dir_l (which lives off this axis) is preserved.

Edit (per layer l, then renormalize):
    dir_l' = dir_l - C Cᵀ dir_l
where C's columns are the orthonormalized confound axes:
  --mode contrastive : {u_I - u_It, u_I - u_You, u_I - u_We}   (most surgical)
  --mode subspace    : {u_I, u_It, u_You, u_We, u_They, u_He, u_She}  (robust)
Each u_t is diag(output_norm.weight) * dequant(output.weight)[t], unit-normalized.

Only layers >= --from-layer are edited (late layers align with unembedding space;
early layers barely touch output logits). Pure offline edit: rewrites .f32/.gguf.
"""
import argparse, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "gguf-py"))
from gguf import GGUFReader, GGUFWriter
from gguf.quants import dequantize

N_LAYER, N_EMBD = 43, 4096
# token ids from the DS4 tokenizer (/tokenize): no-space and leading-space variants
IDS = {
    "I":   [43, 342],  "It":  [2107, 983], "You": [3476, 2042],
    "We":  [2581, 1350], "They": [8474],   "He":  [3158], "She": [6737],
}


def load_unembed_rows(model_glob, norm_scale=True):
    """Return {word: unit vector in residual space} for the confound tokens."""
    want = {tid for ids in IDS.values() for tid in ids}
    rows, wnorm = {}, None
    for f in sorted(glob.glob(model_glob)):
        r = GGUFReader(f)
        for t in r.tensors:
            if t.name == "output_norm.weight":
                wnorm = np.array(t.data, dtype=np.float32)
            if t.name == "output.weight":
                full = dequantize(t.data, t.tensor_type).reshape(-1, N_EMBD)  # [vocab, n_embd]
                for tid in want:
                    rows[tid] = full[tid].astype(np.float32).copy()
    if not rows:
        raise SystemExit("output.weight not found in model shards")
    if norm_scale and wnorm is not None:
        rows = {k: v * wnorm for k, v in rows.items()}   # map into pre-norm residual space
    # per-word vector = mean of its token variants, unit-normalized
    def unit(v):
        n = np.linalg.norm(v); return v / n if n > 1e-8 else v
    return {w: unit(np.mean([rows[t] for t in ids], axis=0)) for w, ids in IDS.items()}


def build_confound(U, mode):
    if mode == "contrastive":
        cols = [U["I"] - U["It"], U["I"] - U["You"], U["I"] - U["We"]]
    else:  # subspace
        cols = [U[w] for w in ("I", "It", "You", "We", "They", "He", "She")]
    C = np.stack(cols, axis=1)            # [n_embd, k]
    Q, _ = np.linalg.qr(C)                # orthonormal basis of the confound span
    return Q                              # [n_embd, k]


def main():
    ap = argparse.ArgumentParser()
    # weights live in the steering project (moved out of llama.cpp); override as needed
    WEIGHTS = "/home/trevor/Projects/ds4-refusal/llamacpp-iq3/weights"
    ap.add_argument("--in-f32", default=f"{WEIGHTS}/native_nothink.f32")
    ap.add_argument("--out", default=f"{WEIGHTS}/native_notic")
    ap.add_argument("--model-glob",
                    default="/home/trevor/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/"
                            "DeepSeek-V4-Flash-UD-IQ3_XXS-*.gguf")
    ap.add_argument("--mode", choices=("contrastive", "subspace"), default="contrastive")
    ap.add_argument("--from-layer", type=int, default=30)
    ap.add_argument("--no-norm-scale", action="store_true")
    a = ap.parse_args()

    dirs = np.fromfile(a.in_f32, dtype=np.float32).reshape(N_LAYER, N_EMBD)
    U = load_unembed_rows(a.model_glob, norm_scale=not a.no_norm_scale)
    Q = build_confound(U, a.mode)         # [n_embd, k]

    removed = []
    out = dirs.copy()
    for l in range(a.from_layer, N_LAYER):
        d = dirs[l]
        comp = Q @ (Q.T @ d)              # projection onto confound span
        removed.append((l, float(np.linalg.norm(comp)), float(np.linalg.norm(comp)/ (np.linalg.norm(d)+1e-9))))
        d2 = d - comp
        n = np.linalg.norm(d2)
        out[l] = d2 / n if n > 1e-8 else d2

    out.tofile(a.out + ".f32")
    # gguf: direction.il = out[il], il in 1..42 (aligned convention)
    w = GGUFWriter(a.out + ".gguf", "controlvector")
    w.add_string("controlvector.model_hint", "deepseek4")
    w.add_uint32("controlvector.layer_count", N_LAYER)
    for l in range(1, N_LAYER):
        w.add_tensor(f"direction.{l}", out[l])
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

    print(f"mode={a.mode} from-layer={a.from_layer} confound-rank={Q.shape[1]}")
    print("layer  |proj|  frac-of-dir removed:")
    for l, mag, frac in removed:
        if l >= 38 or frac > 0.05:
            print(f"  {l:2d}   {mag:6.3f}   {100*frac:5.1f}%")
    print(f"wrote {a.out}.f32 / .gguf")

if __name__ == "__main__":
    main()
