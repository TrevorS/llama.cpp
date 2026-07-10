import sys, numpy as np
harmful, harmless, out = sys.argv[1], sys.argv[2], sys.argv[3]
G = np.fromfile(harmful, dtype=np.float32).reshape(-1, 43, 4096)
B = np.fromfile(harmless, dtype=np.float32).reshape(-1, 43, 4096)
n = min(len(G), len(B))
np.savez(out, G=G[:n], B=B[:n])
print(f"packed G={G.shape} B={B.shape} -> {out} (n={n})")
