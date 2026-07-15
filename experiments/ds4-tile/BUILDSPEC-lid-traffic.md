# Buildspec: lid traffic campaign (post-split #1 depth-scaling cost)

Scouted 2026-07-14 (iter 29 attribution + 4 parallel design scouts).
Detail pass 2026-07-15: see BUILDPLAN-lid-traffic.md for line-precise
hunks + corrections (pre-quant buffer 16.5 MiB/stream not 4.1MB; int8
K is 16.8 MB @131k, the 8.4MB figure was MXFP4-packed; merge re-reads
the score matrix per tree level — third traffic term).
All refs `ds4-flash-experiments`. Predecessor context: split-attention
(iter 27) made FA depth-flat; the lid chain is now the dominant
depth-scaling prefill term.

## Attribution (iter 29 + shallow legs, kernel shares of GPU-busy)

| group                     | d8192 | d32768 | d65536 | d131072 |
| ------------------------- | ----- | ------ | ------ | ------- |
| MoE mmq (+quantize)       | ~49%  | ~38%   | 35.7%  | 31.2%   |
| FA (dense-CSA / split)    | 20.5% | 29.4%  | 21.4%  | 18.3%   |
| lid: score_int8           | 2.8%  | 7.2%   | 12.7%  | 20.4%   |
| lid: topk chunk+merge     | <1%   | 2.8%   | 4.6%   | 9.7%    |

Two DIFFERENT bottlenecks inside lid:
- score kernel: SHARED-MEMORY-bandwidth-bound (K int8 16KB smem tile
  re-streamed once per head x64; 88% mem SoL; K itself is 8.4MB @d131k,
  L2-resident — DRAM is NOT the problem).
- topk chain: DRAM-bound on the [nt, n_lid] f32 score-matrix round-trip
  (268MB write + full re-read + merge re-reads ~ 0.77 GB/layer/ubatch
  @d131k; top_k=2048 vs SORT_N=4096 makes merge_group=2, a deep tree).

Also surfaced: FA share PEAKS at d32768 — the dense-CSA branch still
runs below TILE_MIN (n_csa>=12288 i.e. d>=49k). TILE_MIN was tuned
pre-split; probe whether split-tile wins in the d32k-49k band now
(zero-code env probe, results appended below when run).

## Levers, ranked (scout verdicts)

### 1. SORT_N 4096 -> 8192  [GO — build first]
Merge tree at top_k=2048 collapses 4 launches -> 2 (merge_group 2->4).
BLOCKER found: 64KB static smem won't compile (48KB cap) — convert the
3 topk kernels to dynamic `extern __shared__` + cudaFuncSetAttribute
(fits GB10's 99KB carveout; kernels are DRAM-bound, occupancy loss
acceptable). Host merge loop already fully SORT_N-parameterized; all
production n_lid values (16384..131072) divide 8192; partial-chunk
clamps verified present.
- Scope: dsv4_lid_topk.cuh:3 + .cu:819-897 + launch .cu:1006-1060,
  ~+22/-7, 2 files, zero new template instantiations.
- Gate gap: existing multi-chunk tests (n_lid 4097/5000/6000) fall
  under 8192 -> add one n_lid>8192 non-divisible case (~17000).
- Class: identical numerics (pure selection-width change).

### 2. f16 score store, d_idx==128 paths only  [GO — with #1]
Halves the 268MB round-trip. Contract verified CLEAN: scores buffer is
pool-only (.cu:1143-1145), read ONLY by topk kernels; EXACT rescore
recomputes from q/k directly (.cu:918-978) — no other consumer (full
grep). Keep smem sort values f32 (compare on dequantized f32 — sort
semantics unchanged). Restrict to d128 paths (int8/wmma/decode) whose
gates are already fp16-boundary class (3e-3/1.2e-2); scalar d64 path
keeps f32 + strict-0.0 gate.
- Scope: ~+16/-12 in dsv4_lid_topk.cu (4 score kernels' store, 3 topk
  reads, launch pointer types).
- Gate: existing DSV4_LID_TOPK suite + re-run EXACT oracle to confirm
  margin m=64 covers added pass-1 displacement.

### 3. Global int8-K pre-quant kernel  [GO — independent win]
From the fusion scout: pre-quantize lid-K [128, n_lid] -> global int8 +
per-comp scales ONCE per layer (4.1MB @d131k) instead of today's
per-block requant (every score block re-derives the same int8 tile).
Strictly less work; bit-identical if it reuses the lane-owns-4-dims +
shfl_xor amax layout (.cu:363-384). ~+35 LOC kernel + host wiring.

### 4. Fused score+chunk-topk  [PROTOTYPE — env-gated, ncu-gated]
Kills the score-matrix write + full chunk re-read (~0.5GB/layer DRAM
traffic; footprint is a WASH — candidate-value tree replaces scores).
A-safe (bit-identical) ONLY if bitonic + int8 quant + ascending-h accum
copied verbatim. Forces 1-token x 4096-comp blocks -> loses 16-token
smem-K amortization; K becomes a 512KB-per-chunk L2 stream x64 heads.
- RISK #1 (decides it): K L2-residency under chunk-major scheduling.
  ncu L2 hit-rate on the d131k leg is the go/no-go BEFORE any default.
- Scope: +195/-20 in dsv4_lid_topk.cu (2 new kernels: pre-quant from #3
  + fused score/chunk; merge reads stored cand-vals), env
  LLAMA_DSV4_LID_FUSED_CHUNK default OFF, decode path untouched.
- Gate: fused==nonfused bit-identity case + existing suite + EXACT 0.0.

### 5. fp4-mma score kernel, REGISTER-RESIDENT K only  [CONDITIONAL GO]
Prior negatives (iter 14c/15: tensor cores idle at 14% compute; int4
smem unpack tax 1.6x slower) killed every smem-resident-K variant —
but never tested K resident in REGISTERS with the head loop over A
fragments only. MLA (1 K per 64 heads) makes B-resident amortization
real: per-head smem traffic drops 16KB->~1KB (16x). fp4 (not int8-mma)
because 4-bit keeps a 2x wider comp tile resident per register.
- Schedule: B = tile<8,8> e2m1 (2 regs/64 dims) RESIDENT; loop h: load
  A = q_h e2m1, 2x mma m16n8k64 (mma.cuh:1126 wrapper, scale_vec::2X
  matches the packed cache's 4x e8m0-per-row exactly, byte-gather only);
  relu(C)*w_h epilogue. Comp tile Nc=16-32 (regs ~40); Nc=128 is
  register-dead (~140 regs).
- Numerics: = LLAMA_DSV4_LID_FP4's measured class (0.93 top-512
  overlap, PPL statistically identical) when K = packed cache; q
  runtime e2m1. EXACT backstop applies.
- PREREQ: LLAMA_DSV4_LID_CACHE_MXFP4 (packed K rows) — this kernel is
  the missing consumer that would justify flipping P3b default ON.
- KILL CRITERION: ncu — if Compute(SM) becomes the bound (per-head
  q-pack + scalar epilogue) before beating int8's 21.76ms reference,
  revert as documented negative (same disposition as int4).
- Scope: ~+180 LOC new kernel + ~50 repack helper (model on
  load_tiles_nvfp4 mmq.cuh:1068) + dispatch arm + FLAGS row, env
  LLAMA_DSV4_LID_FP4_MMA, BLACKWELL_MMA_AVAILABLE-gated.

### DEAD: 4-bit K in smem (dp4a)
Re-derivation of iter-15's measured negative (LLAMA_DSV4_LID_INT4,
1.6x SLOWER, reverted). dp4a has no int4 form; nibble->int8 unpack is
2-5x the dp4a op count on the same integer pipe. K:q smem traffic is
8:1 so the 44% traffic bound is real but unreachable. Do not rebuild.

## Build order + expected wall impact @d131k (lid = 30.1%)

1+2+3 first (one PR, ~+75/-20, mostly A-safe): topk 9.7% -> ~4-5%
expected; score kernel gets the pre-quant relief. Then 4 (prototype,
ncu-gated): most of the remaining round-trip. Then 5 (probe,
kill-criterion): score 20.4% x optimistic 1.3-1.6x -> ~5-7% wall.
Combined optimistic ceiling ~10-15% pp @d131k, more at 262k+.

## Gates (every step)

- test-backend-ops DSV4_LID_TOPK suite (incl EXACT zero-tolerance) +
  new cases noted per lever.
- gates.sh std at the end of the campaign; perf legs single-test with
  clocksnap (power/thermal caps move numbers ±4%).
- Any pass-1 numerics change: re-run the fp4_oracle displacement
  measurement for RESCORE_M.

## TILE_MIN probe result (2026-07-14, d32768 same-boot pair)

pp2048@d32768: TILE_MIN default (dense-CSA) 364.5 vs TILE_MIN=4096
(tile+split active) 370.1 = +1.5%. Weak positive — real but inside the
band where a default change needs more: a d40960 leg (mid-gap), and the
accuracy gates (u_cap=4096 against n_csa=8192 top-2048 unions truncates
more at shallow depth). Parked as a minor follow-up, not a lever.
