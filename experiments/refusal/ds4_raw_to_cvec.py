#!/usr/bin/env python3
"""Convert a ds4.c raw steering direction (N_LAYER x N_EMBD f32) to a llama.cpp
control-vector gguf with the CORRECT layer alignment.

ds4.c applies row L at layer L (all 43 layers, 0..42). llama.cpp applies
`direction.<il>` at layer il, for il in [1, n_layer-1] (layer 0 is never
steered; index 0 is rejected by the loader). So the faithful mapping is:

    direction.il  :=  ds4_row[il]      for il in 1..n_layer-1

Row 0 is dropped (llama can't steer layer 0). The previous inline conversion
wrote direction.(L+1) = row L, shifting every layer by one and pushing the
critical last-layer direction (row 42, the </think> decision layer, largest
separation) onto direction.43 where NO layer applies it.
"""
import argparse, sys
import numpy as np
sys.path.insert(0, "gguf-py")
from gguf import GGUFWriter

N_LAYER, N_EMBD = 43, 4096

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/home/trevor/Projects/ds4-refusal/steering/profiles/refusal")
    ap.add_argument("--out", default="experiments/refusal/ds4_aligned.gguf")
    ap.add_argument("--renorm", action="store_true",
                    help="unit-normalize each row (ds4 rows are already unit-norm)")
    a = ap.parse_args()

    raw = np.fromfile(a.raw, dtype=np.float32)
    assert raw.size == N_LAYER * N_EMBD, f"expected {N_LAYER*N_EMBD}, got {raw.size}"
    rows = raw.reshape(N_LAYER, N_EMBD)

    w = GGUFWriter(a.out, "controlvector")
    w.add_string("controlvector.model_hint", "deepseek4")
    w.add_uint32("controlvector.layer_count", N_LAYER)

    for il in range(1, N_LAYER):          # il = 1..42, applied AT layer il
        v = rows[il].astype(np.float32)
        if a.renorm:
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
        w.add_tensor(f"direction.{il}", v)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    # report the two layers that matter for the off-by-one check
    print(f"wrote {a.out}: direction.1..direction.{N_LAYER-1} = ds4 rows 1..{N_LAYER-1}")
    print(f"  direction.42 norm={np.linalg.norm(rows[42]):.4f} (ds4 row 42, decision layer)")
    print(f"  direction.1  norm={np.linalg.norm(rows[1]):.4f} (ds4 row 1)")

if __name__ == "__main__":
    main()
