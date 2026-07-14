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

3b. Packed container (storage + decode bandwidth):
- NEW op `GGML_OP_DSV4_QAT_SET_ROWS` (constructor mirrors ggml_set_rows;
  ~15 registration touchpoints as usual): f32 rows → MXFP4-container view
  with QAT rounding. CUDA kernel reuses the A1 quantizer math, packs
  nibbles + e8m0 scale. CPU ref mirrors.
- `llama-kv-cache-dsv4.cpp:1067`: kv_lid ctor type → GGML_TYPE_MXFP4 when
  `LLAMA_DSV4_LID_CACHE_MXFP4=1`; lid cpy_k routed to the new op.
- Readers:
  - decode: `dsv4_score_decode_kernel` KT=packed path — warp loads 68B/row
    (17 words lane-strided), LUT nibble→f32 × exp2(e−127), existing shuffle
    dot (drop int8 sub-path in packed mode; kernel is latency-bound).
  - prefill: per-layer-per-ubatch staging dequant kernel
    `dsv4_mxfp4_dequant_f16` (cache → pool f16 buffer, ~8MB @32k); wmma/int8
    kernels consume staging unchanged.
  - rescore: packed-exact dequant (bit-identical to write rounding).
- Gate: backend-ops (packed cases for topk + decode), selection set ==
  Phase-3a output exactly; decode kernel µs before/after (expect 2–3x);
  tg64@d131072 and @d524288 re-legs; PPL spot.

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
