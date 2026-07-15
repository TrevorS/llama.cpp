# Buildplan: lid traffic campaign — line-precise implementation plan

Detail-scouted 2026-07-15 (5 parallel implementation scouts, code-only,
no GPU). Companion to BUILDSPEC-lid-traffic.md (the ranked lever
record); this doc is the execution reference. All line refs verified
against 534b2b320 (`dsv4_lid_topk.cu` = 1374 lines).

## Corrections to BUILDSPEC (measured against the tree)

- **RETRACTED 2026-07-14 (post-build): the n_lid correction below was
  WRONG.** n_lid = kq_mask->ne[0] is the COMPRESSED lid length
  (compressor ratio 4), not full depth: at d131072 n_lid ~ 32768+pad
  (the 33280 perf case IS the d131k serving shape). BUILDSPEC's
  original figures were right: score matrix 268MB, K f16 8.4MB,
  int8 K ~4.2MB, pre-quant buffer ~4.3MB/stream. The fused lever's
  L2 margin is comfortable after all; the ncu gate remains the
  formal check.
- ~~Pre-quant buffer is ~16.5 MiB/stream, not 4.1MB~~ (retracted, see
  above — holds only at ctx >= 512k where n_lid reaches 131072).
- ~~n_lid tracks full depth~~ (retracted, see above).
- **Merge re-reads the score matrix at EVERY tree level**
  (`row[idx]` gather, .cu:888) — a third traffic term the "~0.5GB"
  headline undercounted. Killing it requires the candidate-VALUE tree
  (fused plan §step-2 below).
- **"lane-owns-4-dims" is strided-by-32** (`d = lane + i*32`), not
  contiguous. Decode's mapping (`lane*4+i`) differs but amax is
  order-independent. Replication must use the strided form.
- **fp4-mma register budget**: ~50 regs @Nc=16, ~78 @Nc=32 (not ~40);
  Nc=128 is ~246 regs — dead as claimed, more so. Probe Nc=16 first.
- **supports_op needs no edit for SORT_N** — ggml-cuda.cu:4976 tests
  `op->ne[0] <= DSV4_TOPK_SORT_N`, auto-tracks the macro.

---

## STEP 1 — BUILT 2026-07-14. Outcome: f16+pre-quant land (~1% op win),
## SORT_N 8192 REVERTED (measured negative)

Measured (op-level A/B, test-backend-ops perf, same boot):
- SORT_N=8192: +13% op time @n_lid 8704, +7% @17000, flat @33280,
  +12% @33280-decode. ROOT CAUSE the launch-count argument missed:
  bitonic work grows N*log^2 N, so fewer-but-bigger chunks do ~30%
  MORE total sort work than the halved merge launches save. Reverted
  to 4096; the dynamic-smem + score_t template structure KEPT.
- f16 scores + int8-K pre-quant at SORT_N=4096: -0.6..-1.3% op time
  across all four deep shapes. Wall: flat within noise (lid op is
  score-dominated; topk's f16 win is small because the merge GATHER
  is random-access — 2B vs 4B lands in the same 32B sectors, so f16
  does NOT reduce gather sectors. The gather is killed by the
  candidate-VALUE tree, i.e. step 2's merge change, which also works
  for the UNFUSED path — promoted to the front of step 2).
- EXACT oracle with f16 pass-1: p100 51 vs 50 baseline @33k — m=64
  window holds (13 ranks headroom). Note baseline seed-2 was already
  50, tighter than the previously quoted 36.

Original plan below, kept for the record.

## STEP 1 (original plan) — one PR: SORT_N 8192 + f16 scores + int8-K pre-quant

Three levers, one PR, because they touch the same two kernels: the
3 topk kernels get dynamic smem AND a `score_t` template; the int8
score kernel gets a new K interface AND a half scores param. Composed
edit list below. Scope: `dsv4_lid_topk.cuh` (+1/-1), `dsv4_lid_topk.cu`
(~+95/-45), `tests/test-backend-ops.cpp` (+2). All numerics classes:
SORT_N identical, pre-quant bit-identical, f16 scores = existing d128
gate class; d64/scalar path untouched (strict-0.0 gate preserved).

### 1a. SORT_N 4096 -> 8192 (dynamic smem)

- `.cuh:3`: `#define DSV4_TOPK_SORT_N 8192u`.
- 3 topk kernels — replace static smem pairs at .cu:825-826 (single),
  850-851 (chunk), 879-880 (merge), identical 3-line block:
  ```cuda
  extern __shared__ float dsv4_topk_smem[];
  float    * vals = dsv4_topk_smem;
  uint32_t * idxs = reinterpret_cast<uint32_t *>(dsv4_topk_smem + SORT_N);
  ```
  4-byte alignment holds naturally; `dsv4_bitonic_sort` (.cu:796)
  already takes pointers — zero signature change. rescore_select<1024>
  (.cu:996) SORT_N-independent, untouched.
- Host `dsv4_topk_launch`: after `const int block = 1024;` (.cu:1011):
  `const size_t smem = (size_t) SORT_N * (sizeof(float) + sizeof(uint32_t));`
  (64KB). At each of the 4 launch sites (.cu:1014/1038/1047/1057):
  `CUDA_SET_SHARED_MEMORY_LIMIT((kernel<SORT_N, score_t>), smem);` then
  pass `smem` as 3rd `<<<>>>` arg. Macro at common.cuh:230-245 already
  encapsulates the run-once-per-device guard (fattn-mma-f16.cuh:1974
  precedent). Parens around the templated name are required.
- Occupancy: 1024 thr + 64KB -> 1 block/SM on GB10 (99KB carveout);
  fine — grids are n_tokens/(n_tokens x chunks)-wide, DRAM-bound.
- Invariants verified: partial-chunk clamp .cu:849 parametric;
  `n_lid<=SORT_N` single-kernel route .cu:1013 now covers <=8192;
  `merge_group = SORT_N/top_k` assert >=2 holds for top_k<=4096.
  Merge-launch count @n_lid=33280, top_k=2048: chunks 9->5, merges
  4->2 (tree 3->1, +final).
- Coverage shift: n_lid 4097/5000/6000 cases fall to the single-kernel
  path. Add TWO cases in test-backend-ops.cpp:
  - after :10480: `test_dsv4_lid_topk(GGML_TYPE_F16, 128, 64, 17000, 2048, 1, 512)`
    (multi-chunk, 17000 % 8192 = 616 partial clamp)
  - near :9670: `test_dsv4_lid_topk(GGML_TYPE_F32, 64, 4, 8193, 2, 1, 100)`
    (just over one 8192 boundary, scalar path)

### 1b. f16 score store (d128 paths only)

Kernel-path map (verified): scalar `dsv4_score_kernel` store :215 =
the d_idx!=128 path — KEEP f32. Half-store conversions: wmma :323,
int8 :439, decode :533 (all d128).

- Template the 3 topk kernels + `dsv4_topk_launch` on
  `<typename score_t>`; add load helper next to `dsv4_ldk` (.cu:32-33):
  ```cuda
  static __device__ __forceinline__ float dsv4_lds(const float * p) { return *p; }
  static __device__ __forceinline__ float dsv4_lds(const half  * p) { return __half2float(*p); }
  ```
  Load sites: single :829, chunk :854, merge :888 -> `dsv4_lds(row + ...)`.
  score_t=float compiles to a plain deref — d64 codegen unchanged.
  smem sort vals stay f32; `dsv4_better` semantics untouched. No
  vectorized stores/loads exist anywhere on this boundary (verified).
- 3 store sites wrap in `__float2half(...)`; params `float*` -> `half*`
  at kernel sigs :236/:344/:460. Masked cells: `__float2half(-inf)` =
  f16 -inf, sorts to bottom as today.
- Host alloc (replace .cu:1143-1145) — dual RAII allocs, the existing
  q_fp4_alloc idiom (:1186-1188):
  ```cuda
  const bool scores_half = (d_idx == 128);
  ggml_cuda_pool_alloc<float> scores_f32_alloc(pool);
  ggml_cuda_pool_alloc<half>  scores_h_alloc(pool);
  float * scores   = scores_half ? nullptr : scores_f32_alloc.alloc((size_t) nt * n_lid);
  half  * scores_h = scores_half ? scores_h_alloc.alloc((size_t) nt * n_lid) : nullptr;
  ```
  `sc_s` retyped half at :1261 (decode) and :1281 (wmma/int8); scalar
  branch :1310-1318 unchanged. Both `dsv4_topk_launch` call sites
  (:1337 EXACT pass-1, :1372 final) split into
  `scores_half ? <half>(... scores_h ...) : <float>(... scores ...)`.
- Consumer trace CLEAN: scores feeds ONLY the 4 score-kernel launches
  + the 2 topk_launch calls; EXACT rescore (:918-978) recomputes from
  q/k, LID_DUMP reads q/w/k. No other reader.
- Numerics: one rounding at the store only (compute stays f32);
  |score| O(1..1e2) vs f16 max 65504; strictly smaller than the
  int8/wmma quantization noise upstream. Existing d128 gates
  (1.2e-2 int8/dec, 3e-3 wmma; test:5842-5843) cover it.
- LOAD-BEARING GATE: EXACT-mode margin. Pass-1 over f16 scores adds
  displacement; rescore window n_cand = top_k + m (m=64) must still
  contain the true top-k (oracle p100 was 36). The EXACT oracle
  re-run is REQUIRED before landing, not optional. Bump
  LLAMA_DSV4_LID_RESCORE_M default if p100 approaches 64.

### 1c. Global int8-K pre-quant

BIT-IDENTITY VERDICT: SAFE. The in-kernel quant (.cu:363-384) reads
only the K-row's own 128 values — no token-window/q/mask coupling;
every blockIdx.y tile re-derives the identical tile today (128x
redundant at nt_s=2048). Per-row global pre-quant reproduces every
step: strided dims `d=lane+i*32`, fmaxf accum i=0..3, `__shfl_xor_sync`
o=16..1, `inv=127/amax`, `sk=amax/127`, `__float2int_rn`, dim-natural
`[comp*128+d]` layout.

- New `dsv4_prequant_k_int8_kernel<KT>` (~30 LOC) above the int8
  kernel (~:342): warp-per-comp, block 256 (8 warps), grid
  ceil(n_lid/8); outputs `int8 k_i8[n_lid][128]` + `float k_sc[n_lid]`.
  Padding comps skipped (score-kernel load zero-fills, matching old
  ks=0/sk=0 behavior).
- `dsv4_score_int8_kernel` interface: drop `template<KT>` + `k`/`nbk2`,
  add `const int8_t* k_i8, const float* k_sc`. Replace quant block
  :363-384 with a guarded smem copy loop (`ok ? k_i8[...] : 0`,
  `sk[c] = ok ? k_sc[comp] : 0`). Everything from :386 (q-quant, dp4a,
  epilogue) unchanged. smem footprint unchanged (16KB+2KB+0.5KB) —
  global read per block drops 32-64KB -> 16KB and per-block quant
  arithmetic disappears.
- Host: 2 pool allocs (16.5 MiB/stream int8 + scales) + per-stream
  pre-quant launch at the top of the int8 arm (d_idx==128 non-decode
  branch, ~:1276-1304); the f16/f32 int8 dispatch arms collapse to one
  (input already int8). Same stream — in-stream ordering suffices.
- Out of scope (documented): decode path DEFERRED (pre-quanting 16MB
  of K for 1 token is a traffic loss; its packed-direct arm reads raw
  MXFP4 anyway); wmma path NEVER (fp16 tensor-core, no int8 form).
- Gate: bit-identical -> existing suite + EXACT 0.0 unchanged is the
  proof. No new tests.

### Step-1 gates (in order)

1. `test-backend-ops test -o DSV4_LID_TOPK` default profile (incl the
   2 new cases).
2. Same with `LLAMA_DSV4_LID_EXACT=1 LLAMA_DSV4_LID_QAT_WRITE=1` —
   zero tolerance must hold.
3. EXACT oracle displacement re-run (RESCORE_M margin, per 1b).
4. Perf legs: single-test pp2048 @d65536 and @d131072, clocksnap
   pre/post, settle-gated, same-boot A/B vs 534b2b320 baseline.
   Expected: topk share 9.7% -> ~4-5% @d131k; score kernel gets the
   pre-quant relief on top.
5. gates.sh std battery at campaign end (not per-step).

---

## STEP 2c — RADIX-SELECT topk (scouted 2026-07-14, WINNER of the
## K=N/2 design round; supersedes 2a(B) value tree and 2b fusion)

Measured basis: at production shape (n_lid 33280, nt 2048, top_k
2048) topk = 50.4ms of the 121.7ms op (chunk 24.0 + merge 26.4, 1+4
launches, merge_group=2); chunk kernel is smem-COMPUTE bound (SM
77.8% / L1 78.0% / L2 0.5%). 2a(A) partial bitonic (landed
fe6043410) only gets 0.93 at K=N/2.

Design (one kernel/token replaces chunk+merge on the d128/half path):
- key(h): canonicalize -0 first (`hb==0x8000 -> 0` — REQUIRED: +0/-0
  compare equal in f32 with index tie-break, but have distinct f16
  bits; -0 is reachable via __float2half(tiny negative); without the
  fold a top_k boundary slicing the ±0 group breaks set-equality),
  then `(hb&0x8000) ? ~hb : hb^0x8000` — key-eq == f32-value-eq, so
  the radix tie rule (all key>T, then lowest-index fill among ==T)
  IS dsv4_better exactly. No NaN possible (finite acc + {0,-INF};
  -INF -> key 0x03FF, smallest).
- grid=nt, block=1024, 2x 8-bit MSB-first smem-histogram passes ->
  threshold byte pair; pass 3 = deterministic ORDERED prefix-scan
  compaction (tiles ascending, in-tile scan preserves index order;
  atomic-append is WRONG — nondeterministic in which ==T wins);
  final dsv4_bitonic_sort<pow2(top_k)> over <=2048 selected (67.6k
  CE/tok = 3% of the old 2,076.7k).
- smem ~17KB @top_k 2048; NO pool scratch (candidate tree deleted on
  this path). First pass streams the 136MB matrix once (~0.5ms);
  passes 2-3 hit the L2-resident 66KB/token row.
- Fallbacks stay bitonic: d64/f32 scalar (strict-0.0 gate, untouched
  code path), decode/small-nt (nt < NT_RADIX_MIN ~16: one block per
  token starves the grid). Env LLAMA_DSV4_LID_RADIX default ON, =0
  reverts.
- Dispatch: both dsv4_topk_launch call sites (EXACT pass-1 ~:1407
  and final ~:1446) route the half arm to dsv4_topk_radix_launch;
  EXACT compat: radix yields the correct SET of n_cand candidates,
  rescore re-sorts. supports_op unchanged.
- Expected: topk 50.4 -> ~5ms, op 121.7 -> ~76 (-37.5%), ~9-11%
  serving wall @d131k. After landing, score kernel = 93% of the op.
- Scope: dsv4_lid_topk.cu +~180/-4 (1 kernel, 1 launcher, 1 key
  helper, 2 call-site edits), tests +1 boundary-tie eval case (many
  equal scores straddling top_k).
- Gates: all 5 op profiles at existing tolerances (0.0 gates live on
  untouched paths but EXACT pass-1 set-equality DOES exercise radix)
  + new tie case + op perf A/B both top_k shapes + serving legs.
- Ranked risks: (1) ±0 canonicalization omitted -> boundary set
  mismatch; (2) compaction must be ordered-scan, not atomic-append;
  (3) small-nt occupancy (gated); (4) none for d64 (untouched).

ALSO REJECTED this round: SORT_N=8192-with-partial-K — exact CE math
at production K=2048: chunk CE +7.9% (more tail padding), total
+4.6%, occupancy 2->1 blocks/SM; only pays at small K where radix
wins anyway. Dead in both full-sort and partial forms. SORT_N=16384
dead outright (128KB > GB10 99KB carveout).

## STEP 2 — fused score+chunk-topk (env prototype, ncu-gated)
## [KILLED 2026-07-14 — see PROGRESS iter 32: ceiling <1.7% vs 16x
## K-L2-traffic cost on the 58-70% score side; no 512k niche;
## running-topk-in-score-kernel geometrically blocked at 8:1
## selectivity. Kept below for the record.]

REVISED 2026-07-14 by the post-step-1 delta scout + the step-1 perf
findings. Changes vs the original design below:

- **Sub-step 2a, DO FIRST (may stand alone): candidate-VALUE tree for
  the UNFUSED path.** Step-1 measurement showed the merge's random
  `row[idx]` score-matrix gather is the topk term f16 can't touch
  (random 2B vs 4B = same 32B sectors). Storing (idx, half val) at
  chunk emit and merging over stored vals kills the per-level gather
  without any fusion. Small diff (merge kernel variant + chunk emits
  vals + launch threads a half val-scratch), A-safe: the stored half
  IS the same rounded score the gather would load via dsv4_lds.
  Measure op-level before deciding 2b.
- **Fused kernel (2b) bit-identity under f16 scores**: the fused
  kernel must round each score through __float2half BEFORE the sort
  (vals[i] = __half2float(__float2half(acc+mv)), matching the unfused
  store at :469 + load at :889) and the cand-val tree stores HALF —
  merge then reads dsv4_lds(cand_val_in + i), byte-identical to the
  unfused merge inputs.
- **Fused internal chunk = SORT_N (4096 after the step-1 revert)** —
  identical partition to the unfused chunk kernel, so intermediates
  match byte-for-byte; 40.5KB smem, 2 blocks/SM. (The delta scout's
  8192 recommendation predated the SORT_N revert; with the host back
  at 4096, 4096-internal is both the bit-identity choice AND the
  occupancy choice.)
- **Own launch wrapper** (dsv4_fused_topk_launch): level 0 = fused
  kernel (emits idx+val), merges read cand-vals; per-stream offsets
  like the int8 loop. Must be called INSIDE the d128 branch (k_i8
  pool alloc is branch-scoped) with an early return.
- **scores_h guard**: compute the fused predicate before the dual
  scores alloc and skip scores_h when fused — otherwise the fused
  path still reserves the matrix it eliminates.
- **top_k at the gated perf shapes is 512** (the 2048 in those cases
  is nt_s); serving uses top_k 2048. merge_group at 4096: 8 and 2.
- Predicate: fused && int8 && d128 && nt_s>1 && n_lid>SORT_N &&
  !exact && !k_packed_direct. Env LLAMA_DSV4_LID_FUSED_CHUNK, static
  lambda after the dec flag (~:1256).
- Scope (delta-scout): dsv4_lid_topk.cu +~155/-15 (fused kernel ~80,
  merge-val kernel ~45, launch wrapper ~50), tests +1-2, FLAGS +1.
- K L2-residency stays the ncu go/no-go for 2b, but with n_lid
  corrected to ~33k @d131k (int8 K ~4.2MB) the margin is comfortable;
  the risk is real only at 512k-class contexts (n_lid ~131k, 16.8MB).

Original design notes below, kept for the record.

### STEP 2 (original design)

Design settled by the scout; key decisions:

- **Stays SORT_N=4096 internally** — sort arrays + q staging = 40.5KB
  (fits static 48KB); at 8192 it's 72.5KB for no benefit. DECOUPLED
  from step-1a: merge compatibility keys on top_k/candidate_stride,
  not SORT_N; the two regimes never mix within one call. Do not
  entangle the PRs.
- **Depends on 1c**: fused blocks stream pre-quantized int8-K from
  global/L2 (512KB/chunk x 64 heads); staging K in smem is impossible
  (smem is full of sort arrays) and re-quantizing per block is the
  waste 1c removes.
- Grid `(nt, n_chunks)`, block 1024; per block: compute scores for
  one (token, 4096-comp chunk) straight into smem vals[], bitonic,
  emit top_k (idx, VAL) pairs. Score matrix never exists.
- **Candidate-VALUE tree**: scratch doubles to (u32 idx + f32 val)
  per candidate — tiny vs the matrix it kills. Merge kernel drops the
  `scores` param + `row[idx]` gather (:881/:888), reads
  `cand_val_in[set0*top_k + i]`, writes `cand_val_out` at non-final
  levels. Host merge loop (:1042-1059) threads the val pointers. This
  kills the per-level matrix gather — the third traffic term.
- Reuse verbatim (A-safety): `dsv4_bitonic_sort` + `dsv4_better`
  (:791-816), q-quant block (:397-417), dp4a inner + ascending-h
  relu-then-weight epilogue (:423-431), pre-quant kernel from 1c.
  New glue ~120 LOC.
- Env `LLAMA_DSV4_LID_FUSED_CHUNK` default OFF — static-lambda idiom
  (:1166-1169). Predicate:
  `fused && int8 && d128 && nt_s>1 && n_lid>SORT_N && !exact && !k_packed_direct`.
  Decode + shallow prefill (single-chunk) structurally excluded.
- Scope: .cu +~175/-15; tests +2; gates.sh +6 (fused re-run legs);
  FLAGS.md +1 row. Add prefill-shaped perf case
  `test_dsv4_lid_topk(F16,128,64,33280,2048,1,2048)` (existing :10480
  family is decode-shaped at :10481).
- Validation precedent (FA_SPLIT): (a) op-suite re-run under the flag
  incl EXACT combo; (b) greedy temp-0 A/B `cmp` at d>=65k, flag 0 vs 1
  — byte-identity is the claim, so test it as one.
- GO/NO-GO: ncu on the d131k-shaped perf case —
  `GGML_CUDA_DISABLE_GRAPHS=1 ncu --replay-mode application
  --launch-count 8 --metrics lts__t_sector_hit_rate.pct,
  lts__t_sectors.sum,dram__bytes.sum,gpu__time_duration.sum
  --kernel-name regex:'dsv4_fused_score_chunk' test-backend-ops perf
  -o DSV4_LID_TOPK -p <shape>` vs same-shape unfused. GO if DRAM bytes
  drop ~= eliminated matrix traffic AND K re-streams hit L2. NO-GO if
  DRAM flat (int8 K = 16.8MB @131k vs GB10 L2 — margin unknown, the
  measurement IS the resolution).

---

## STEP 3 — BUILD-READY 2026-07-14 (detail scout vs d89d2849b).
## Deltas vs the original notes below:

- **No nibble repack needed** (removes the hardest part): the
  container row IS an array of standard block_mxfp4 {e; qs[16]}
  (ggml-common.h:216-218; dsv4_qat_set_rows packs canonical order),
  and Blackwell mmq consumes qs by plain memcpy (mmq.cuh:934 — the
  feared quantize_mxfp4_mmq permutation does not exist in this
  tree). B(K) = raw 4B gathers from the 68B row; A(q) = pack with
  the dsv4_qat_set_rows routine into block_mxfp4 order in smem.
  Dot products are permutation-invariant when A and B agree on
  physical dim per k-slot — they do, verbatim.
- Geometry: reuse the int8 kernel's block=256 / grid
  ((n_lid+127)/128,(nt_s+15)/16); warp owns 16 comps (2 B-tiles
  resident, ~42-50 regs, ~5 blocks/SM, smem ~1.2KB vs int8's 18KB).
  A-smem is ~1.1KB (4-bit: 64B/token), not the 4.25KB prior note.
- Bypass: fp4_mma_active skips BOTH the f16 staging and the int8
  pre-quant; byte strides nbk2/nbk3 = k->nb[2]/nb[3]. Off-Blackwell
  one-shot warn -> int8 fallback.
- EXACT wiring: route pass-2 through the packed-direct inline-
  dequant rescore arm (predicate k_packed_direct || fp4_mma_active
  at ~:1413) — staged-f16-alive rejected (doubles K reads). q_d
  stays official f32 (fp4-mma skips the q fake-quant kernel).
- Test tolerances: fp4-mma is class B — the 1.2e-2 d128 gate WILL
  fail. Add an env-gated max_err branch (~8e-2 smoke, tightened to
  ~2x observed miss before commit); the REAL correctness proof is
  LID_FP4_MMA=1 + LID_EXACT=1 at 0.0 (load-bearing). Plus resident
  greedy A/B + PPL (the LID_FP4 acceptance protocol).
- Perf reference: int8 score = 71.1ms @33280x2048 (the old 21.76ms
  was n_lid=8704, not comparable). Add MXFP4 perf case
  (MXFP4,128,64,33280,2048,1,2048). KILL if sm>=l1tex AND duration
  >= 71.1ms, or regs spill >64, or wall >= 71.1ms.
- Marginal protocol: BOTH A/B arms run LID_CACHE_MXFP4=1 (container
  tax common-mode).
- Scope: dsv4_lid_topk.cu +~185/-4 (kernel ~120 + arm/gates), tests
  +~4, FLAGS +1.
- Post-radix context: after step 2c lands, score = 93% of the lid
  op — this probe is the remaining campaign.

## STEP 3 (original notes) — fp4-mma register-resident-K probe (kill-gated)

Feasibility CONFIRMED at the operand level:

- **Container feeds the mma transform-free** (verified bit-for-bit):
  P3b 17B block-32 layout (.cu:81-102) stores raw ue8m0 scale byte +
  nibbles whose bit pattern (`best|sgn`, sgn=8) IS the hardware e2m1
  encoding. `mma_block_scaled_fp4` (mma.cuh:1126-1154,
  `scale_vec::2X ... ue8m0`) takes one u32 = 2 e8m0 bytes per k64
  frag; a d=128 row = 2 frags = OR-pack the row's 4 scale bytes into
  2 u32. Own gather from the 68B row (mmq's `load_tiles_mxfp4_fp4`
  assumes split .e/.qs arrays — not reusable as-is; :936-941 is the
  pack model).
- Schedule: B = K resident. tile<8,8> B ne=2 -> 4 int32/thread per
  8 comps x 128 dims. Loop h ascending 0..63: pack q_h e2m1 (per-32
  e8m0, in-kernel prologue -> ~4.25KB smem A-layout, pack routine
  ~30-40 LOC modeled on dsv4_qat_set_rows_kernel :120-141 reusing
  `dsv4_e2m1_index`), ldmatrix A, 2x mma, drain C: `acc += relu(C)*w[h]`
  in f32 regs (mirrors int8 :430). Store scores via templated
  `ST*` (composes with 1b's half buffer).
- Register budget (CORRECTED): ~50 @Nc=16 — build Nc=16 first;
  Nc=32 (~78) only if occupancy shows headroom; Nc=128 dead (~246).
- Gating: env `LLAMA_DSV4_LID_FP4_MMA` default OFF; compile under
  `BLACKWELL_MMA_AVAILABLE` (common.cuh:286-288, __CUDA_ARCH__ 1200..
  1300 — this IS the Blackwell macro, nothing newer exists); host
  gate `k_is_mxfp4 && d_idx==128 && nt_s>1 && flag && blackwell`,
  fallback int8 with one-shot stderr warn (:1099-1104 idiom).
  Bypasses the prefill f16 staging (:1217-1226), reads 68B rows
  direct like the decode packed-direct arm (:1227-1230).
- WIRING BLOCKER (must fix in the PR): EXACT pass-2 rescore uses the
  staged `k_f16_d` (:1346-1352); if fp4-mma bypasses staging, keep
  staging alive when EXACT is set OR route pass-2 through the
  packed-direct inline-dequant arm (:1339-1345). Else EXACT breaks
  silently.
- Numerics: class B == LID_FP4's accepted class (0.93 top-512
  overlap, PPL statistically identical; PROGRESS:417/:491) — container
  holds QAT-exact values, mma consumes them exactly; only q-side
  per-32 e2m1 (already in LID_FP4's sim) + mma accumulation order
  (near-tie reshuffle) differ. EXACT backstop -> class A.
- Success framing: the probe pays the container tax (pp -2~3.6%)
  unless run inside the 512k packed profile where it's already paid —
  measure MARGINAL over int8 with LID_CACHE_MXFP4=1 in both arms.
- KILL: ncu `--set full --launch-count 20` (no --launch-skip,
  GGML_CUDA_DISABLE_GRAPHS=1) — kill if sm__throughput >=
  l1tex__throughput AND duration >= int8's 21.76ms @d131k reference
  (PROGRESS:568), or if launch__registers_per_thread shows spills.
  This adjudicates iter-15's "bound is L1, freeing compute won't
  help" prediction vs the register-resident rebuttal.
- Scope: .cu +~200 (kernel ~140 + q-pack ~40 + dispatch ~20), tests
  +~10 (extend the :5842 tolerance predicate to the new env), FLAGS
  +1 row, PROGRESS writeup.
