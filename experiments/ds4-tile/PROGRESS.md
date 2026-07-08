# DS4 prefill parity loop — progress log

Targets (UD-IQ2_XXS, GB10, llama-bench -ngl 999 -fa 1 -mmp 0, best ub):
1. Short ctx: pp2048 >= 419 t/s, tg64 >= 14.4 t/s (ds4.c gb10.csv reference).
2. Long ctx (added by Teej): usable prefill+decode at 512k ctx minimum, ~1M stretch — ds4.c ran near-1M on this box. KV is ~14.1 KB/token (7.2 GiB @ 512k, 14.5 GiB @ 1M) so memory-wise feasible; the blocker is the indexer ctx*ub compute buffer -> task #5 is mandatory. Provisional long-ctx gates until ds4 reference numbers exist at depth: allocates+runs at 512k, coherent output, tg@512k >= ~10 t/s, deep-chunk prefill doesn't collapse (graceful continuation of the <=64k curve). Optionally regenerate ds4 512k reference via ~/Projects/ds4/ds4-bench on a remaining GGUF.

Branch: ds4-tile-prefill (local only, never push).

## Baseline (2026-07-07, build d39e36c + experiments commit 970d3ef)

| config | pp2048 | tg64 |
| --- | --- | --- |
| llama.cpp ub512 | 322.13 | 14.16 (parity within noise) |
| ds4.c target | 419 | 14.36 |

Kernel extraction + A/B harness: PASS at real dims (4096/2048/256/top-6), tile kernels run at ~87 GB/s bandwidth ceiling.
Same-quant gap at equal-ish batch: 1.30x.

## Iteration 1 — big-ub probe (in flight)

Hypothesis: ds4's 419 used chunk 2048; llama.cpp short-ctx pp2048 can run ub 1024/2048 (indexer buffer small at 2k ctx). Measures the free win before code.
RESULTS: ub1024 pp 365.0 / tg 15.93; ub2048 pp 374.8 / tg 15.99. tg target MET (16.0 >= 14.4, verify it holds later). pp gap now 374.8 -> 419 (1.12x), batch scaling saturated (+2.7% for 2x ub).
Conclusion: both #5 (mandatory for depth/512k anyway) and #3 (tile fusion, the remaining short-ctx ~12%) proceed. tg64 anomaly (14.2@ub512 vs 16.0@ub2048) noted — tg should not depend on ub; recheck when re-benching.

## Iteration 2 — Phase A: fused indexer op (agent implementing)

Spec: new fused op replacing the 8-op chain in build_lid_top_k; chunked ctx scores + running top-k; CUDA via ggml_cuda_pool workspace; CPU reference; env-gated (LLAMA_DSV4_FUSED_LID=1); validation = top-k index-set compare + greedy spot check + short PPL; then re-bench short ctx and NEW depth points + 256k/512k allocation test.

RESULTS (lid-fuse agent stalled; main session recovered WIP 22e7db3 and validated):
- Gate 1 backend-ops: 6/6 PASS (CPU vs CUDA, odd n_lid, n_stream=2)
- Gate 2 greedy A/B on real model: token-IDENTICAL
- Gate 3: 256k AND 512k ctx allocate + run with flag on — previously failed to allocate at any ub; the 512k requirement is now met at the allocation level
- Gate 4: pp2048 372.6 (no regression); tg64 13.52 REGRESSION (vs 16.0) -> fixed in ce16c0b by fusing only nt>1 (decode keeps unfused chain; its buffer is tiny even at 1M)
- In flight: re-bench tg after fix + first depth points pp2048@d32768/tg64@d32768 at ub2048 (previously unrunnable)

Status vs targets: tg 16.0 (met, pending re-confirm), pp short 374.8 vs 419 (gap 1.12x -> task #3 tile fusion), 512k alloc proven (full #6 validation pending).

## Iteration 4 — depth profile (main session)

Decode fix confirmed: tg64 short = 15.91 (target met). Depth at ub2048: pp2048@d32768 147.3 (ds4: 347.7), tg64@d32768 11.71 (ds4: 13.0), pp512@d8192 242.4.
nsys pp512@d8192 kernel breakdown: dsv4_score_kernel 22.4% (simple dp -> wmma port needed), k_bin_bcast mul+add 17.7% (25k tiny launches — anomalous, attribute+absorb), mul_mat_q 28.9% (MoE, task #3), flash_attn 10.7% (fine), concat 5.7%.
-> Iteration 5 (lid-wmma agent): wmma scores + elementwise storm cleanup.

## Iteration 5 results (lid-wmma)

Goal: replace the fused op's scalar `dsv4_score_kernel` (22.4% of GPU @ pp512/d8192,
41.5ms avg/launch) with a tensor-core kernel, and attribute/eliminate the k_bin_bcast
storm (op_mul 1150ms + op_add 1080ms cumulative — the #2/#3 GPU consumers at depth).

Profile (pp512@d8192, sqlite): dsv4_score_kernel<half> 68 launches, 2819ms total (#1).
Storm real cost is in 2 big op_add variants (gridX=8192 173ms, gridX=32768 871ms) and
2 big op_mul variants (grid 16x512 305ms, 16x2048 720ms); op_div/reduce_rows are cheap.

- [gate 1] test-backend-ops -o DSV4_LID_TOPK: 10/10 PASS (added 4 wmma-path shapes:
  multi-token-tile, non-128-div n_lid, n_stream=2, F32 k). wmma path (d_idx==128) rounds
  q to fp16 -> ~1e-4 set-mismatch at near-tie boundaries; gated small max_err there,
  scalar path stays strict at 0. Deterministic across runs.

### k_bin_bcast storm attribution (target 2) — OUT OF DSV4-INDEXER SCOPE

Attributed by inverting the binbcast grid->dst-shape mapping (block.x=128 for
contiguous F32; ne0=gridX*256; broadcast operand stops collapse at dim0) and matching
per-layer op counts. Result: n_embd=4096, nt in {512,2048}, n_layer=43.

The 4 big variants are ALL the DeepSeek-V4 hyper-connection (HC) residual mixing in
deepseek4.cpp (build_hc_weighted_sum ~L237-238, build_hc_post ~L345-350), NOT the
lightning indexer:
- op_add gridX=8192  -> [n_embd, nt=512]  contiguous HC add (full collapse: 8192*256 = 4096*512)
- op_add gridX=32768 -> [n_embd, nt=2048] contiguous HC add (32768*256 = 4096*2048; the "4x depth" is coincidence = n_embd*2048)
- op_mul 16x512      -> [n_embd,512]  x [1,nt] broadcast HC weight-mul
- op_mul 16x2048     -> [n_embd,2048] x [1,nt] broadcast HC weight-mul
Per layer: 48 HC muls + 39 HC adds; counts match 2 pp512 evals + partial depth-fill evals
exactly (add:mul ratio 0.813 observed = structural 39:48). ~87 elementwise launches/layer
x 43 layers.

Both brief-named suspects RULED OUT:
- build_top_k_mask ggml_add (L653): operates on [n_csa, nt] masks -> cheap (4096,1,1)
  op_add variant, 42 launches ~1.5ms total. Not the storm.
- unfused indexer relu->mul->sum_rows->add: ABSENT from profile (no relu kernel; fused
  dsv4_score/topk kernels present) -> fused path correctly active for prefill, no bug.

Decision: left as-is this iteration. HC mixing is a distinct DSV4 feature, not absorbable
into ggml_dsv4_lid_topk, and warrants its own fused op / graph-batching workstream with
independent numerics validation (proposed: batch build_hc_weighted_sum as a broadcast-mul
+ hc-axis reduction, and build_hc_post's comb-mix as a batched ggml_mul_mat over nt;
~9-14x launch reduction targeting ~2070ms). Filed for a follow-up iteration.

## Iteration 6 — wmma depth bench (main session, post-crash recovery)

Session crash corrupted .git (11 truncated objects incl. tip commit); recovered by
resetting to 4944408 and re-committing the intact working tree as a215c85.

wmma score kernel depth results (ub2048, fused LID):
- pp512@d8192: 242.4 -> 306.3 +-14.1 (+26%)
- pp2048@d32768: 147.3 -> 298.4 +-0.3 (+103%); gap vs ds4.c 347.7 now 1.17x
- tg64@d32768: 11.94 +-0.39 (was 11.71; target 13.0 — decode-side, wmma n/a)

Anomaly: tg64@d32768 leg failed with "failed to decode prompt batch, res = 1" when run
in the same llama-bench process after the pp2048@d32768 test; standalone rerun clean.
Suspect CUDA pool state carryover between tests; recheck if it recurs.

-> Iteration 6 (moe-tile agent): port experiments/ds4-tile expert-tile kernels into
mul_mat_id dispatch, env-gated LLAMA_DSV4_MOE_TILE=1 (mul_mat_q 28.9% = #1 consumer).
HC storm fusion queued behind it.

## Iteration 6b — MoE tile port BLOCKED on quant types (moe-tile agent)

Type-check gate failed before any code: lifted tile kernels (ds4_tile_kernels.cuh) are
Q4_K-only; model experts are ffn_gate/up_exps IQ2_XXS (42L, IQ2_S x1) and ffn_down_exps
IQ3_XXS (41L, MXFP4 x2). Profile enums confirm: 16=IQ2_XXS (gate/up), 18=IQ3_XXS (down).
ds4.c has an IQ2_XXS gate/up kernel (not lifted) but ZERO IQ3 support — the fused
gate_up->mid->down pipeline cannot close on-device for this quant. Coverage: 0/3.
Decision: Option A (stand down port). Gate/up-only IQ2_XXS bridge filed as future
option (payoff capped at ~14.8% slice, routing bridge hard, numerics unvalidated).
-> moe-tile agent REDIRECTED to HC storm fusion (LLAMA_DSV4_HC_BATCH=1, graph-level
op batching preferred over new CUDA; targets the 17.7% elementwise slice).
Fresh nsys re-rank on wmma build in flight (wmma_depth.nsys-rep).

## Iteration 7 — re-rank on wmma build (pp512@d8192, wmma_depth2.nsys-rep)

pp512@d8192 = 308.2 +-9.5. Ranking: mul_mat_q IQ2_XXS(16) 27.5% / IQ3_XXS(18) 15.1% /
HC storm (bcast mul 10.3 + add 5.6) 15.9% / flash_attn 8.1% / mul_mat_q Q8_0 7.8% /
concat 6.1% / dsv4_score_wmma128 2.5% (scalar was 22.4%; 41.5ms -> 1.9ms avg, 22x).
MoE now ~50% of GPU. Gate/up IQ2_XXS share (27.5%) reopens the bridge option from 6b
at ~2x the payoff originally estimated.
-> Iteration 7: moe-iq2 agent ports ds4's IQ2_XXS gate_up_mid tile kernel + routing
bridge (down stays mul_mat_q); moe-tile agent continues HC batching in parallel
(disjoint files: ggml-cuda vs src/models/deepseek4.cpp).

## Iteration 7b — HC graph-restructure NEGATIVE RESULT (moe-tile agent)

LLAMA_DSV4_HC_BATCH op-composition (weighted_sum 7->3 launches via bcast-mul+sum_rows;
hc_post ~38->~6 via repeat+bcast term1 and k=4 batched mul_mat term2): builds clean,
greedy diverges after ~30 tokens (fp reassociation from mul_mat, expected), but
pp512@d8192 REGRESSES 306.5 -> 282.5 (-7.8%, r=4, off-leg reproduces baseline).
Root cause: box is bandwidth-bound (~87 GB/s); the scalar storm is coalesced
contiguous elementwise, so launch count was never the bottleneck. The batched path
adds traffic (full [n_embd,hc,nt] transpose, 4x repeat materialization, skinny n=4
GEMM with poor utilization). Op composition cannot express the traffic-minimal form.
Decision: fund fused CUDA op instead (LLAMA_DSV4_HC_FUSED): out = x*post + sum_src
res*comb reading each operand once, scalar-loop accumulation order for exact A/B;
regressing HC_BATCH branches to be replaced by it (negative result recorded here).
Also: cross-agent note — new .cu files need cmake reconfigure (ggml-cuda file(GLOB))
or other exes hit undefined refs at link.
