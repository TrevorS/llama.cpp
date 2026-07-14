# Buildspec: tiled-FA shared-window fix (split-attention + LSE merge)

Target: recover the fattn small-nb latency wall — tiled FA runs 12 vs 41
TFLOPS because every tile (FA ne3 stream) re-loads the shared raw window
from LPDDR. Ceiling measured by ncu/union-overlap: **+65% pp @ d131k**.

Scouted 2026-07-14. All refs `ds4-flash-experiments`.

## Where the amplification lives

- `deepseek4.cpp:850` — `ggml_repeat_4d(raw_k, hd, 1, n_raw, T)` physically
  replicates the raw window T times; `:852` concats it with the per-tile
  gathered unions into `k_all [hd, 1, n_raw+u_cap, T]`; one FA call (`:896`)
  covers raw+union per tile. Each of the T streams re-reads the same ~53%
  of its KV from distinct addresses (replication defeats L2).
- Kernel side (`fattn-mma-f16.cuh:1815-1821`): K/V pointers are hard-indexed
  `nb13*sequence` / `nb23*sequence` — one uniform stride, no way to express
  "shared for [0,n_raw), per-stream after". Mask ALREADY broadcasts on ne3
  (`sequence % ne33`, `:1818`; assert `ggml.c:5667`) — K/V do not
  (`ggml.c:5658-5659` assert q/k/v ne3 equal).

## Design decision: split-attention (a), NOT kernel ne3-broadcast (b)

(b) would edit the hottest templated kernel (`fattn-mma-f16.cuh`, 101 KB:
pointer math, `KV_max` skip, stream-K `kbc` decomposition) + relax op
asserts + need a two-tensor signature — regression surface spans every
non-DSV4 FA consumer. Rejected.

(a): two FA calls + log-sum-exp merge:
  i.  dense FA over the shared raw window, ne3=1, full occupancy
      (this is plain raw attention — the fast 41-TFLOPS shape);
  ii. tiled FA over ONLY the per-tile disjoint union remainder
      (`k_all` shrinks to `[hd, 1, u_cap, T]` — no repeat_4d, no concat);
  iii. merge the two normalized outputs with per-row LSE stats.

The merge math already exists twice in-tree (`flash_attn_combine_results`
fattn-common.cuh:916-969 for the vec path; `flash_attn_ext_f16_fixup`
:871-911 in-launch for mma) but is NOT exposed across two FA calls.

## The one blocking gap: FA does not emit LSE

`ggml_flash_attn_ext` (ggml.c:5646) outputs only the normalized
`[DV, n_head, n_tokens, ne3]` result; per-row (max, sum) is discarded.
Merge of two independent calls is impossible without it.

Plan: opt-in LSE output — `ggml_flash_attn_ext_with_lse(dst)` flag (mirrors
the existing add_sinks-style op mutators): dst ne[0] = DV+1, kernel writes
the per-row log-sum-exp in the extra slot at the same place the fixup
already has (max_val, rowsum) live. Opt-in ⇒ zero change for every other
FA consumer; upstream-worthy on its own (sequence-parallel / tree
attention both want FA-with-LSE).

New op `ggml_dsv4_fa_merge(a, b)`: out = (ea*va + eb*vb)/(ea+eb) with
ea = exp(lse_a - max(lse_a,lse_b)) etc., per row. Trivially parallel;
same op-registration touchpoints as union/memb.

## Scope

- Files: `ggml/src/ggml.c` (+~30, LSE flag + ne0 handling),
  `ggml/src/ggml-cuda/fattn-common.cuh` + `fattn-mma-f16.cuh` (+~40, write
  LSE in launch_fattn epilogue + fixup), `src/models/deepseek4.cpp`
  (+~40/-15 around :827-896, split into two calls + merge),
  new `dsv4_fa_merge` op (~+120 across ggml.h/ggml.c/cuda/cpu/tests).
- Named units: `ggml_flash_attn_ext`, `launch_fattn`,
  `flash_attn_ext_f16_fixup`, `build_csa_lid_attention`, 1 new op, tests
  for FA-with-LSE + merge (merge of two manual FA halves == single dense
  FA reference, tight fp tolerance).
- Verification: backend-ops new cases; tile-vs-dense greedy/PPL gates
  re-run; pp depth legs d65k/d131k/d262k (expect toward +65% @d131k);
  kernel occupancy re-probe (ncu) on the remainder-only tiled call.
- Risk: public API yes (opt-in FA output variant) · data migration no ·
  cross-module yes (ggml core + cuda + model graph) · reversible yes
  (env-gated alongside LLAMA_DSV4_CSA_TILE) · external blocker no.

## Side benefits

- Removes `repeat_4d` + `concat` materialization of `k_all` (today:
  T×n_raw×hd×2B extra writes+reads per layer per ubatch, plus pool
  pressure at 512k).
- Remainder-only tiled call halves each stream's KV again (u_cap vs
  n_raw+u_cap) — fewer blocks/stream, but the dense call now carries the
  bulk at full occupancy, which is the right trade.
- Mask for the tiled call shrinks to `[u_cap, W, 1, T]` (memb only, no
  raw slice concat).

## Gates

1. backend-ops: FA-with-LSE vs reference (dense == merged halves), merge
   op exact formula tests, existing FA suite untouched-paths green.
2. Tile A/B: split-attention vs current concat path — PPL c32768 trio,
   passkey battery, within-config determinism.
3. Perf legs: pp2048 @ d65k/d131k/d262k/d512k; ncu occupancy probe.
4. Upstream hygiene: LSE flag opt-in, no behavior change when unset
   (full test-backend-ops sweep).
