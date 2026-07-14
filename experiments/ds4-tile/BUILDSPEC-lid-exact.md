# Buildspec: official-exact LID selection (+ class-A promotions)

Target: `LLAMA_DSV4_LID_EXACT=1` — indexer selection bit-exact vs the official
DeepSeek QAT graph, at int8 first-pass speed, with mxfp4 cache storage.
Companion promotions: sinkhorn order-exactness, tile exact-profile cap.

Design decisions locked by the scouts (iter 21b/21c):
- ggml MXFP4 is a CONTAINER; rounding must be OURS (QAT: scale
  `exp2(ceil(log2(amax/6)))`, even-index tie-break — ggml's differs on both).
- The two-pass select lives INSIDE `ggml_cuda_op_dsv4_lid_topk` — no graph
  or op-signature changes; env-gated internals.
- CPU reference computes official semantics DIRECTLY (QAT fp32 score, exact
  top-512) → backend-ops compares GPU two-pass against ground truth, making
  the standard test suite the class-A verifier.

## Phase 0 — oracle: measure the displacement bound m  (no model changes)

- `experiments/ds4-tile/fp4_oracle.cpp`: add `--int8-displacement` mode:
  for each dumped token (LLAMA_DSV4_LID_DUMP data, plus a fresh deep dump at
  d65k+), compute rank of every official-top-512 member under the int8
  ranking of QAT scores; report p50/p99/p100 displacement.
- Ship `m = 2 * p100` rounded to 64. Expected m ≈ 32–64 (0.5% score error).
- Also dump at d512k depth once to confirm m does not grow with n_lid.
- Gate: displacement table in PROGRESS; m fixed as a constant with env
  override `LLAMA_DSV4_LID_RESCORE_M`.

## Phase 1 — sinkhorn C→A  (~20 LOC, independent)

- `dsv4_hc_fused.cu::dsv4_hc_sinkhorn_kernel`: replace the serial softmax
  sum with the exact shuffle-tree order ggml's `soft_max_f32` uses for
  ne0=8 rows (read that kernel first; replicate reduction order and eps/div
  sequence bit-for-bit). Max is order-free; only the exp-sum order matters.
- CPU ref: keep serial? NO — CPU ref must match the kernel; replicate the
  same tree order there (sum pairs in the same sequence).
- Gate: greedy A/B fused-vs-unfused (LLAMA_DSV4_HC_FUSED=0) TOKEN-IDENTICAL
  on a 4k prompt ×128 tokens (class-A gate now valid for it), same machine
  state (back-to-back runs).

## Phase 2 — two-pass rescore inside the fused op  (works on today's F16 cache)

All inside `ggml/src/ggml-cuda/dsv4_lid_topk.cu`:
- `dsv4_topk_launch(..., top_k)` called with `top_k + m` when exact mode on;
  candidate buffer widened (scratch sizing already parameterized).
- NEW `dsv4_lid_rescore_kernel` (one block per token):
    inputs:  cand[512+m] (pass-1 indices), q_qat f32 (A1 kernel output,
             already in pool when LID_FP4 path active), weights, mask,
             K rows (F16 today; Phase-3 packed later)
    compute: exact QAT fp32 score per candidate (warp per candidate:
             64 heads × 128 dims, serial-order head accumulation to match
             the CPU reference), then in-block bitonic over 1024 slots →
             emit exact top-512 (desc score, asc index tie-break).
- Env: `LLAMA_DSV4_LID_EXACT=1` ⇒ forces QAT q/k numerics (A1 path) +
  two-pass; supersedes LID_INT8/LID_DEC/LID_FP4 for selection semantics
  (they remain as pass-1 speed knobs).
- CPU ref (`ggml-cpu/ops.cpp` lid_topk): when exact env set, compute QAT
  fp32 scores + exact top-512 directly (no two-pass).
- Tests: existing DSV4_LID_TOPK cases run under `LLAMA_DSV4_LID_EXACT=1`
  with `max_err = 0` on the index set (exact-set compare); add an
  adversarial near-tie case (many equal scores) for tie-break parity.
- Gate: backend-ops exact-set 0-err; e2e greedy determinism + PPL spot.

## Phase 3 — mxfp4 storage

3a. QAT-at-write (numerics only, container stays F16):
- `deepseek4.cpp`: apply the A1 k-side fake-quant ONCE at cache-write time
  (graph: hadamard → fake-quant → cpy_k) instead of per-score-call; remove
  the k-side round-trip from score kernels when env set.
- Zero reader changes; q-side QAT stays in the score path.
- Gate: backend-ops + selection identical to Phase-2 output (same values).

3b. Packed container (storage + decode bandwidth).
SCOUTED 2026-07-14 (full blast-radius inventory; refs on this branch):

Two-step landing — correctness first, packed readers second:

3b-i. MXFP4 container + QAT write + staged-dequant reads (correctness):
- Decouple lid type: `llama-kv-cache-dsv4.cpp:1067` passes the SHARED
  `type_k` to all 4 sub-caches (raw/csa/hca/lid) — flip ONLY the lid ctor
  arg to GGML_TYPE_MXFP4 under `LLAMA_DSV4_LID_CACHE_MXFP4=1`; never touch
  the shared type_k. d=128 → 4 block-32 → 68 B/row. clear() memset-zero
  decodes to 0.0 (scale byte 0 + zero nibbles) — still deterministic.
- NEW op `GGML_OP_DSV4_QAT_SET_ROWS`: stock set_rows CANNOT be used —
  CUDA set-rows.cu:230-321 has no MXFP4 dst (GGML_ABORT) and all its
  quantize funcs are stock rounding; CPU ops.cpp:5061 would silently use
  ggml's from_float (wrong scale + tie-break). New op mirrors
  set_rows_cuda_quant with OUR quantizer (dsv4_fp4_quant_kernel math,
  scale exp2(ceil(log2(amax/6))), even-index tie-break). CPU ref mirrors
  via dsv4_fp4_quant_row_cpu. Write site: cpy_k ends in ggml_set_rows at
  llama-kv-cache.cpp:1327 and is SHARED by all caches — add a lid-only
  write path in deepseek4.cpp (~:1203) instead of editing cpy_k. The
  ggml_dsv4_fp4_rt insert (:1199) becomes redundant in packed mode
  (rounding folds into the scatter).
- Read side (ALL via existing staging, zero kernel changes): the
  LID_FP4 path already stages K into a dense f32 pool buffer
  (dsv4_lid_topk.cu:1007-1020, k_force_f32) — replace the fill with
  dequantize_row_mxfp4_cuda (convert.cu:751/809 already maps MXFP4) or a
  custom unpack; every kernel (wmma/int8/dec/rescore) takes its existing
  float arm. Do NOT lean on ggml_get_rows/ggml_cpy (no MXFP4 in either).
- Assert/dispatch fixes: dsv4_lid_topk.cu:892 + ops.cpp:8381 K-type
  asserts (add MXFP4); :895 nb[0]==type_size assert (block semantics);
  :960 nbk2/nbk3 element-stride division (packed = block strides — the
  landmine; staging sidesteps it, native readers must index by block);
  ggml-cuda.cu:4967 supports_op src[1] whitelist + new-op arm + CPU case.
  CPU lid_topk K branch (ops.cpp:8432-8459): add MXFP4 unpack arm
  (dequant-then-as-is; QAT already applied at write).
- Session save/restore: dsv4_state_write/read_k_cache
  (llama-kv-cache-dsv4.cpp:292/:319) is ggml_row_size byte-copy —
  type-agnostic, works unchanged; bump DSV4_K_CACHE_STATE_VER (:336) so
  F16 snapshots don't cross-load.
- Non-fused paths: ggml_lightning_indexer (:682, CPU-only) and manual
  mul_mat (:691) — mul_mat dequants MXFP4 natively; guard or document
  (not the resident GPU path).
- Gate: backend-ops packed cases (topk all variants + qat_set_rows,
  zero-tolerance under EXACT), selection set == P3a output exactly,
  --prompt-cache save/load round-trip, PPL spot.

3b-ii. Native packed readers (perf, after 3b-i gates):
- decode: dsv4_score_decode_kernel KT=packed — warp loads 68B/row
  (17 words lane-strided), LUT nibble→f32 × exp2(e−127), existing shuffle
  dot. Expect 2–3× on the latency-bound decode kernel; this is where the
  exact-mode +200µs/layer decode price gets clawed back.
- int8 prefill kernel packed stage; rescore packed-exact dequant
  (bit-identical to write rounding).
- Gate: decode kernel µs before/after; tg64@d131072/@d524288 re-legs;
  selection unchanged vs 3b-i.

## Phase 4 — profiles, defaults, docs

- Default flips (one commit, FLAGS.md same commit): FUSED_LID→ON,
  CSA_GATHER→ON, LID_DEC→ON, LID_INT8→ON, and once Phases 0–3 gate:
  LID_EXACT→ON (subsumes the int8 class-B caveat entirely).
- Exact-profile tile cap: measure union p100 at d512k (UNION_STATS leg),
  document `LLAMA_DSV4_CSA_TILE_UCAP` values for "exact" vs "fast" profiles
  (5376 covers d65k; expect ~7–9k at 512k; 256-multiples).
- Validation battery (one script, `experiments/ds4-tile/gates.sh` — build
  it as part of this phase so future flag flips are one command):
  backend-ops all DSV4 ops → shallow coherence → greedy determinism ×2 →
  PPL c32768 trio → passkey battery → pp/tg depth legs.

## Order & rationale

0 (measurement) → 1 (tiny, independent win) → 2 (class-A selection on
current storage — the semantic prize, no storage risk) → 3a (numerics
consolidation) → 3b (bandwidth/memory) → 4 (defaults + battery).
Each phase lands independently gated; any can stop the line without
stranding the previous ones.

## Risk register

- Phase 2 near-ties: two-pass must reproduce EXACT tie-break under equal
  fp32 scores — covered by the adversarial test case.
- Phase 3b touches kv-cache session save/restore (state files carry the lid
  cache): verify --prompt-cache save/load with packed cache; bump state
  version if layout changes.
- m violation at extreme depth: debug-mode runtime counter
  (LLAMA_DSV4_LID_EXACT_CHECK=1) rescoring pass-1 boundary; log-only.
- GB10 cross-process variance: all A/B gates run back-to-back same-state;
  transcript gates only for class-A phases (1, 2).
