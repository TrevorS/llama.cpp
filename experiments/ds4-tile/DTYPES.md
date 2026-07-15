# DS4-Flash dtype map — official vs ds4.c vs this branch

Audited 2026-07-15 (3 parallel scouts: HF transformers deepseek_v4 +
paper arXiv:2606.19348; ds4.c/ds4_cuda.cu — full map in
DTYPES-ds4c.md; this branch @47e0599dc with all fast flags ON,
serving unsloth UD-IQ3_XXS). Summary + deviation ranking; per-claim
line refs live in the scout reports (PROGRESS iter 36).

## Three-way map (per role)

| Role | Official prod (paper) | ds4.c (on-box ref) | Ours (fast flags) |
| --- | --- | --- | --- |
| Experts gate/up | MXFP4 e2m1 4bpw -> FP8 mm, f32 acc | IQ2_XXS (~2.06bpw) + Q8_K acts | IQ2_S (~2.5bpw) + q8_1 acts, dp4a |
| Experts down | MXFP4 4bpw | Q2_K (~2.6bpw), mid re-quant | IQ3_XXS (~3.06bpw), mid re-quant |
| Dense attn proj | FP8 e4m3 128x128 | Q8_0 + q8_0 acts | Q8_0 + q8_1 acts |
| Activation stream | BF16 | f32 | f32 |
| Attn softmax | BF16 + max-sub, FP32 sinks | f32, sink-seeded | f32 FA accum (set_prec), sinks in LSE |
| Raw KV | BF16-rope + FP8-rest split | f32 (CPU f16-rt) | f16 unified |
| Compressed KV | (FP8-class) | E4M3 fake-quant blk-64 +-448 | f16, no fake-quant |
| Indexer q/k | FP4 cache + FP4 multiply | hadamard + e2m1 fake-quant, f16 wmma | hadamard + int8 grid (opt-in e2m1) |
| Index scores | BF16 for top-k ("99.7% recall") | f32 | f16 (radix on bits) |
| Router | sqrtsoftplus, bias FP32, x1.5 | same (f16 cuBLAS/f32) | same (BF16 wt @PREC_F32) |
| HC/Sinkhorn | FP32 strict, 20 iters | f32 | f32 fused |
| Embd / lm_head | BF16 / FP8 | F16 / Q8_0 | Q6_K / Q6_K |

Key facts settled by the audit:
- HADAMARD: absent from HF modeling code because a joint orthogonal
  rotation of q and k is invisible to exact dot products — it only
  matters THROUGH a quantizer (spreads outliers before block-32
  e2m1). The paper's FP4 path, ds4.c, and we all rotate; the BF16
  eager reference doesn't need to. All are canonical.
- The PAPER quantizes index scores to BF16 (7 mantissa bits) before
  top-k; our f16 store (10 bits) is FINER than official serving.
  Official attn softmax is BF16; ours is f32. Official KV is
  FP8-rest; ours f16. We sit above canonical precision in all of
  these.
- Even ds4.c deviates from the paper (f32 scores, f16-wmma multiply,
  E4M3-simulated cache). "Canonical" is a band, not a point.
- UD-IQ3_XXS reality check (from the GGUF, not the naming): experts
  are IQ2_S gate/up + IQ3_XXS down (layer 26 bumped IQ3_S/Q8_0);
  router BF16; attn linears Q8_0; embd/output Q6_K; indexer in 21/43
  layers. The custom dsv4_moe_gate_up tile kernel asserts IQ2_XXS
  and is NOT in play for this quant — experts run generic MMQ.

## Deviations, ranked by real risk

Lossier than official:
1. Experts ~2.5-3bpw vs official 4bpw e2m1 — the dominant quality
   gap, size-forced (official-grid routed experts ~137GB > 120GB
   box). ds4.c makes the same trade HARDER (IQ2_XXS/Q2_K): on
   experts we are strictly ABOVE the on-box ground truth.
2. Indexer default int8 grid (not e2m1): class B, 0.93 top-512
   overlap, PPL-neutral; canonical recoverable via
   LID_CACHE_MXFP4 + QAT_WRITE (+LID_EXACT for bit-exact selection)
   at pp -2..-3.6%.
3. Q6_K lm_head/embd vs FP8/BF16 — modest, unmeasured in isolation.

Benign-direction (more precise than canonical): f32 stream + f32
attn softmax (official BF16); f16 KV (official FP8-rest); f16
comp-KV (ds4.c E4M3 fake-quant — our values differ from ground truth
in the finer direction); f16 scores (paper BF16).

## Best-case verdict

With all fast flags on we run a higher-precision pipeline than
official production serving everywhere except expert weights (on a
box the official pipeline doesn't fit), with the indexer as the one
default trading canonical selection for speed under a measured
class-B bound. Against ds4.c — the only reference that runs here —
we are at-or-above fidelity in every role except the indexer
default, while holding the perf records (pp 518.7@d0 / 342.2@d65k /
302.7@d131k; mtp 24.3@1k / 27.6@8k t/s).

Step-3 corollary: the fp4-mma probe moves the indexer multiply ONTO
the official FP4 grid (container e2m1 x runtime e2m1-q) — closer to
canonical than the int8 default it replaces, and faster.
