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

## Iteration 8 — HC fused kernel + MoE gate/up bridge, gates 1-2 (main session)

Two session crashes interrupted this iteration; root causes now understood and
memorized (gb10-session-crash-causes): (1) 04:00 earlyoom kill storm — resident
model + parallel nvcc + runaway 9.7GB llama-cli; killed cicc (the "Error 137"s)
and claude. (2) 07:54 SSH drop (tailscale path flap) tore down the non-tmux,
non-linger user session. Mitigations: -j 4 builds, never build during model
residency, claude now runs in tmux.

Code state (written pre-crash by moe-iq2 + main, wired end-to-end):
- GGML_OP_DSV4_HC_FUSED (LLAMA_DSV4_HC_FUSED=1): dsv4_hc_fused.cu, CPU ref,
  deepseek4.cpp wiring for weighted_sum + post. Scalar-order accumulation.
- GGML_OP_DSV4_MOE_GATE_UP (LLAMA_DSV4_MOE_TILE=1): dsv4_moe_gate_up.cu (395L)
  + IQ2_XXS tables, build_moe_ffn bridge (IQ2_XXS gate/up, nt>1, silu only;
  down stays mul_mat_q). q8_K activation quant differs from ggml (-127/maxv vs
  -128/max) -> agreement gate 5e-3 nmse, not bit-equality.

- [gate 1] test-backend-ops: DSV4_HC_FUSED 6/6 PASS (tests added this session —
  the crash had preempted them: both modes x decode shape, hc=8 max, odd nt);
  DSV4_MOE_GATE_UP 4/4 PASS incl. clamp path + near-real dims.
- [gate 2] HC fused greedy A/B (128 tok, temp 0): TOKEN-IDENTICAL vs scalar graph.
- In flight: 4-leg bench matrix (base/+HC/+MOE/+both at pp512@d8192, pp2048,
  pp2048@d32768, tg64@d32768) + KL divergence (base vs both) for the MoE bridge's
  numerics gate.

## Iteration 8 results — HC fused BIG WIN; MoE tile bridge NEGATIVE RESULT

Matrix (ub2048, r=3, build 38f8cc0; +fix for MOE legs):

| leg | pp512@d8192 | pp2048 | pp2048@d32768 | tg64@d32768 |
| --- | --- | --- | --- | --- |
| base (fused LID) | 308.8 | 392.8 | 294.8 | 12.11 |
| +HC_FUSED | 357.8 (+16%) | 527.4 (+34%) | 364.3 (+24%) | 12.58 (+4%) |
| +MOE_TILE | 304.3 (flat) | 342.5 (-13%) | 253.4 (-14%) | n/a |
| +both | 350.3 | — | 301.6 | n/a |

HC_FUSED clears BOTH remaining ds4.c parity targets: pp2048 527.4 vs 419
(1.26x FASTER than reference) and pp2048@d32768 364.3 vs 347.7. Greedy A/B
token-identical (scalar-order accumulation worked as designed). tg64@d32768
12.58 vs target 13.0 is the only remaining red metric.

MOE_TILE post-mortem: first matrix run crashed every leg — ids from
ggml_argsort_top_k is a VIEW (row stride n_expert, non-contiguous for
k < n_expert); the CUDA op asserted contiguity. Test had fed a contiguous
tensor (gate-1 blind spot). Fixed: count/scatter kernels read ids via
ids_stride; 2 new backend-ops cases route ids through the real
argsort_top_k view (one at real n_expert=256). 6/6 PASS. But perf is a
loss everywhere (above): mul_mat_q IQ2_XXS is already near the bandwidth
ceiling; the ds4 tile8 kernel only paid off inside ds4.c's fused
gate_up->mid->down pipeline, which IQ3_XXS down blocks (6b). Op kept in
tree (correct, env-gated off, tested) as a negative result. Do NOT re-fund
without a fused-down story; improving mul_mat_q scheduling is the better
attack on the 27.5% gate/up slice.

Session-crash forensics recorded in memory (gb10-session-crash-causes):
earlyoom (04:00, killed cicc+claude during build-with-model-resident) and
SSH-drop user-slice teardown (07:54, non-tmux). Claude now runs in tmux.

Recommended config: LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_HC_FUSED=1, ub2048.
KL gate (base vs HC_FUSED, ctx512, 50-chunk docs corpus; original kl_corpus.txt
was crash-truncated to 748B which is why the first KL pass silently produced
nothing): PASS — median KLD 0.000000, max |Δp| 0.006%, same top token 100.000%.
Base PPL 3.9265 +/- 0.089. Iteration 8 CLOSED.
Next candidates: (a) deep-decode gap 12.58->13.0 (decode-side LID chain or
FA/KV traffic at depth — profile tg64@d32768); (b) re-measure short tg with
HC_FUSED (expect >16); (c) long-ctx validation at 512k (task #6, alloc
already proven).

## Iteration 9 — deep-decode gap CLOSED; ALL ds4.c parity targets MET

Free datapoint: short tg64 with HC_FUSED = 16.66 +-0.63 (new best; was 15.99).

nsys tg64@d32768 decode-phase attribution (tg_depth capture; note CUPTI
dropped events after ~2 decode tokens — enough for attribution; window
anchored on hc_weighted_sum=86/eval): deep decode is LAUNCH-BOUND, not
kernel-bound. Per token: ~7,644 k_bin_bcast (grid 1x1x1!, 1.6us) + ~3,420
reduce_rows (4x1x1, 1.4us) + ~620 mul_mat_vec_q = ~11k launches/token,
GPU ~45% busy. Attribution: the UNFUSED decode-side LID chain
(relu->mul->sum_rows->add over ~80 ctx chunks/layer at d32768) — the
iteration-2 decision to keep decode unfused inverts at depth because the
chain's launch count scales linearly with ctx.

Fix: fuse decode-side LID when n_lid >= threshold (default 4096; env
LLAMA_DSV4_FUSED_LID_TG_DEPTH, 0=always, huge=never). ~10 LOC in
deepseek4.cpp, no kernel changes.

Gates (build w/ fix):
- backend-ops DSV4_LID_TOPK: PASS (no kernel change)
- greedy A/B at 9k-token depth, 64 gen: TOKEN-IDENTICAL (old vs new)
- tg64@d32768: 12.57 -> 13.47 +-0.37 (+7%) — BEATS ds4.c 13.0
- tg64 short: 16.57 (unchanged; d<4096 keeps unfused chain, as designed)

SCOREBOARD — all four original targets now met (ds4.c reference in parens):
  pp2048 short    527.4  (419)    1.26x FASTER
  tg64 short       16.6  (14.36)  1.16x
  pp2048@d32768   364.3  (347.7)  1.05x
  tg64@d32768      13.47 (13.0)   1.04x
Config: LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_HC_FUSED=1, ub2048.
Remaining open item: long-ctx validation at 512k (target #2 / task #6 —
allocation proven in iteration 2, needs a real run + tg@512k gate).

## Iteration 10 — post-FA-fix profiling sweep (nsys + ncu app-replay)

Serving config (bd321dec, np2-unified ub2048): 404 t/s @14k. nsys prefill ranking:
mul_mat_q IQ2_S 22.6% + IQ3_XXS 10.8% (ceiling, confirmed), FA<512,512> 19.2%
(19.7ms avg), concat_non_cont 4.6% (k_all raw+comp concat — pure data movement),
rms_norm 4.3%, dsv4 fused kernels 7.4% combined. Decode: CUDA graphs active
(1 launch/token), mmvq Q8_0 31.8%, elementwise storm 18% (compressor/gating).

- rms_norm fix: **NO-OP — aggregation artifact.** Per-launch ncu: q-norm 622us
  = 243 GB/s, HC flat_norm 243us = 276 GB/s — both at DRAM ceiling. The 14%
  SoL average was polluted by Amdahl-irrelevant tiny-grid launches. Do not
  patch norm.cu.
- FA<512,512> (uniform shape, trustworthy): 34% compute / 47% memory SoL —
  ~2x kernel headroom, worth ~9-10% wall. Reference: FlashInfer MLA kernels.
- ncu on this box: kernel replay snapshots unified memory incl. 96G weights ->
  earlyoom (killed 2 sessions). MUST use --replay-mode application.

Queue: (next) eliminate k_all concat -> then FA-512 tuning.

## Iteration 11 — concat row-copy fast path: NEGATIVE (traffic-bound)

Hypothesis: concat_non_cont's per-element index math + branch is the cost.
Wrong — replaced with two vectorized row-copy launches (uint4, narrow-row
packing): per-concat GPU time unchanged (~1.0ms), bench identical
(457.51 vs 456.37 t/s pp2048@d8192 ub2048, r=3). The old kernel was already
coalesced; concat cost IS the ~90MB/instance of traffic. REVERTED.
- The only remaining path for the 4.6-4.9% concat share is zero-copy layout
  (raw-window rows allocated adjacent to comp K per layer) — deep
  llama-kv-cache-dsv4 surgery incl. state save/restore; deferred (risk >> 4%).
- NEW REFERENCE: pp2048@d8192 ub2048 llama-bench = 457.5 t/s on bd321dec.
Queue: -> FA-512. First probe: why does dispatch pick flash_attn_ext_f16
(tile kernel) for D=512/DV=512 on cc12.1 instead of the mma path — if
fattn-mma lacks 512 support, extending it is the FlashInfer-class fix.

## Iteration 12 — FA-512 mma config sweep: default already optimal

Discovery: the 19% FA<512,512,8,8> kernel IS fattn-mma-f16 (same symbol name
as the wmma kernel; stream-k fixup companions prove it) — we were never on a
legacy path. Swept compiled ncols configs via temp env hook (pp2048@d8192
ub2048, r=2): default (8,8) 458.61 · (16,4) 451.93 · (32,2) 452.72 ·
(4,8) 408.28. Generic switch wins; hook reverted.
- Remaining FA headroom (34% compute / 47% mem SoL => ~10% wall) requires
  kernel-internal work: deeper cp.async pipelining or FlashInfer-style MLA
  split-KV for D=512 on sm121. Filed as future deep-dive, not config tuning.
- zsh gotcha: `set -- $cfg` does NOT word-split in zsh — first sweep silently
  ran the default 4x. Always pass env pairs explicitly.

## Queue summary (iterations 10-12)

All three profiler leads resolved: rms_norm = artifact (at ceiling),
concat = traffic-bound (kernel rewrite no-op, zero-copy layout deferred),
FA-512 = config-optimal (kernel-internal headroom remains). Prefill at
457-459 t/s pp2048@d8192 ub2048 is within ~1-2% of practically achievable
with current kernel architecture. The day's serving arc: 140 -> 404 t/s
(unified-FA fix bd321dec + fused kernels + ub2048); further gains need the
FA kernel deep-dive (~10%) or MoE quant changes (vetoed: no requant).

## Iteration 13 — depth-scaling attribution (Flash-MSA scouting -> nsys depth sweep)

Motivation: Flash-MSA blog (Nanduru, 2026-07-11; MiniMax sparse-attention train
kernels) -> audited which forward-pass lessons apply to our inference stack.
Profiles: experiments/profiles/lid_pp_d{32768,131072}.nsys-rep + lid_tg_d131072.

pp2048 depth curve (ub2048, FUSED_LID+HC_FUSED, IQ3_XXS; nsys ~3% overhead):
  d8192 457.5 (iter-11 ref) · d32768 354.6 · d131072 179.8 · d262144 111.8
  => linear ~3.9s flat + ~55us/depth-token per ubatch -> 512k ~63 t/s, 1M ~34.
tg: d32768 13.47 (iter 9) · d131072 9.29 · d262144 7.11 -> 512k ~5 t/s.
BOTH task-#6 gates FAIL at 512k without kernel work.

At-depth kernel shares (final measured ubatch, sqlite window):
  d32768:  FA<512,512> 28.5% · LID 13.5% (score 9.9 + topk 3.6) · MoE 33.6%
  d131072: FA 44.8% · LID 24.1% (score 18.3 + topk 5.8) · MoE 19.8% (flat ms)

Root causes (source-verified + per-launch math):
1. FA is effectively DENSE at depth. Its only mask optimization is
   flash_attn_mask_to_KV_max (fattn-common.cuh:1091) — a per-Q-tile SUFFIX
   bound. Top-512 selections scatter across the prefix (sinks) so the bound
   ~= full length; interior fully-masked tiles are never skipped. CSA FA
   compute is O(n_csa) despite top-512 sparsity. Decode FA 0.278ms/launch
   scans 33k rows for 512 useful ones.
2. LID score wmma runs ~11 TFLOPS (~10% fp16 peak) at all nt; decode (nt=1)
   wastes 15/16 on tile padding (0.79ms vs ~0.04-0.1ms K-read floor; scalar
   kernel is no better at ~1.2ms).
3. topk chain 5.8% @ d131k; scores buffer nt x n_lid fp32 = 1 GiB @512k,
   2 GiB @1M (ds4.c never streamed either — it ate the buffer).

Model facts (GGUF header): indexer 64h x 128d, top_k=512, lid on ~21 CSA
(ratio-4) layers, n_lid = ctx/4. ds4.c runs the indexer FAKE-QUANTIZED fp4
(indexer_hadamard_fp4 / *_qat_* — the model is QAT'd for a low-precision
indexer; all ds4.c reference numbers used it; our fp16 path is over-precise).
ggml already ships e2m1 block-scale mma (mma_block_scaled_fp4, mmq.cuh:1056,
BLACKWELL_MMA_AVAILABLE, sm121a) — liftable; int8 m16n8k32 is the fallback.

Funding ranking (by 512k payoff):
1. Gathered/varlen sparse main attention consuming top-k indices directly
   (MoBA-style per-Q-tile union+dedup for prefill; decode = 512-row gather).
   Kills ~60% of the depth slope AND the build_top_k_mask fill/set_rows chain.
   Projected 512k pp 63 -> ~135 t/s alone.
2. LID score kernel: fp4 QAT-matching path (oracle gate first: port ds4.c
   fake-quant into a backend-ops top-512 set-overlap test) or int8; plus a
   dedicated small-nt decode kernel targeting the K-read floor.
3. Streaming chunk-topk (carry (score,idx) candidates through the merge tree,
   never materialize global scores): mandatory for 1M, modest wall-clock.
Combined projection @512k: pp ~210-230 t/s, tg ~10-12 t/s (gates pass).

## Iteration 13b — rapid-iteration tooling scout (speed + accuracy loops)

Validated toolkit for the funded workstream (gathered sparse FA / fp4 LID /
streaming topk). All items below verified live on GB10 this session.

INNER LOOP (seconds, no model):
- Compile: dsv4_lid_topk.cu real nvcc = ~2s (ccache in play: touch/comment
  edits hit cache in ~1s); link test-backend-ops ~5s. Keep new kernels in
  their OWN .cu (fattn-common.cuh touch = minutes of template recompiles;
  file(GLOB) means new .cu needs a cmake reconfigure — iteration 7b note).
- Accuracy: build/bin/test-backend-ops test -o DSV4_LID_TOPK (5 cases, CPU
  ref compare). New ops get their own cases + fp4 set-overlap gate.
- Speed: test-backend-ops perf — GAP: DSV4_LID_TOPK is NOT in
  make_test_cases_perf() (test-backend-ops.cpp:9687); add deep perf shapes
  (nt=2048 x n_lid={8704,33280}) so kernel timing needs no model/fill.
- Kernel SoL: ncu on test-backend-ops perf (tiny footprint — avoids the
  96G-resident earlyoom trap entirely; --replay-mode application only
  needed when profiling the full model).

PROFILE LOOP (minutes):
- experiments/profiles/kernsum.sh <rep> [window_s]: nsys sqlite export +
  windowed kernel ranking (window isolates the measured pass from depth
  fill). Validated against iteration-13 numbers.
- nsys recipe: nsys profile --trace=cuda --sample=none --cpuctxsw=none
  (~3% overhead on llama-bench legs). CUPTI drops events ~2 tokens into
  deep decode; grid-shape filtering in sqlite recovers per-shape decode
  costs (score decode gridY=1; FA stream-k grid is depth-invariant 48/96).

AT-DEPTH E2E LOOP (~1.5 min/variant after one-time fill):
- llama-completion --prompt-cache state.bin --temp 0: VALIDATED on dsv4 —
  run 2 logs "session file has exact match" and skips prefill. Fill 131k
  once (~12 min), then each kernel variant replays greedy from depth in
  ~1.5 min (model load dominates; restore itself ~seconds). Old llama-cli
  A/B flow is DEAD: new tools/cli is conversational (silent, ERROR-level
  logs); --prompt-cache now belongs to llama-completion.
- test-save-load-state: tests 1-3 PASS on dsv4 (whole-context state
  save/load works). Test 4 seq-copy FAILS: "n_stream mismatch" in
  llama_state_load_file — known issue, multi-seq copy only, does not
  block single-stream A/B. File separately.
- llama-passkey (--junk N --pos P): in-tree needle retrieval for depth
  accuracy gates (not yet exercised; use at milestone, not per-iteration).
- KL gate: llama-perplexity --kl-divergence-base (save once per config)
  + --kl-divergence (iteration-8 recipe, corpus gate).
- llama-bench -d legs + kernsum.sh: milestone-level depth curve re-check
  (fill cost makes it a gate, not a loop).

GOTCHA (cost us a phantom segfault): partial rebuilds desync executables
from shared libs — cmake --build --target X relinks libllama-common.so
under yesterday's binaries -> SIGSEGV in common_params_parser_init that
looks like a real bug. Always full-rebuild (cmake --build build -j4)
after pulling/committing param-struct changes before trusting any tool
crash. llama-bench iteration-13 legs ran on matched binaries (verified).

## Iteration 13c — lid perf cases + fp4 oracle first data

- test-backend-ops perf now covers DSV4_LID_TOPK at serving depths (nt=2048 x
  n_lid {8704,33280} + decode nt=1). Timings match iteration-13 nsys within
  ~3% (34.9ms / 125.9ms / 0.93ms) — kernel speed loop needs no model/fill.
- fp4_oracle.cpp (CPU+OpenMP, experiments/ds4-tile): verbatim port of ds4.c
  hadamard128 + e2m1 block-32 QAT round trip; measures top-512 set overlap
  between our fp16 path (A) and the QAT/fp4 path (B). Synthetic gaussian is
  the worst case: mean overlap 0.90/0.85/0.80 at n_lid 2176/8704/33280
  (i.i.d. scores = razor-thin top-k boundary).
- REAL activations (one-shot dump hook LLAMA_DSV4_LID_DUMP=<dir> +
  LLAMA_DSV4_LID_DUMP_NLID=<min> in dsv4_lid_topk.cu; captured nt=256,
  n_lid=7680 from a 30k-token doc prefill): mean overlap 0.9295, min 0.8906.
- Reading: ds4.c runs the fp4 round trip unconditionally — the official graph
  IS path B — so our fp16 path already deviates ~7% in selection set from
  reference serving, and an fp4 score kernel converges TOWARD official
  numerics. Boundary flips are the lowest-score (lowest attention mass)
  selections. VERDICT: fp4 kernel green-lit. End-to-end decider stays
  greedy A/B + KL corpus + passkey-at-depth; the kernel's set-overlap gate
  should expect ~0.9 vs fp32 ref (not the 3e-3 wmma near-tie tolerance).
- Follow-ups: score-mass-weighted overlap metric; real-data capture at
  n_lid ~33k for the depth trend.

## Iteration 14 — dual scout: fp4 score kernel + gathered sparse FA

### Gathered sparse FA (Q-tile union measured on real d~30k dump, n_lid=7680)
    union W=8   mean 1148  max 1303   cut 6.7x
    union W=16  mean 1519  max 1729   cut 5.1x
    union W=32  mean 1975  max 2260   cut 3.9x
    union W=64  mean 2523  max 2877   cut 3.0x
Union grows SUBLINEARLY in n_lid (adjacent queries attend overlapping KV) while
dense = n_lid grows linearly -> the cut ratio IMPROVES with depth. At d131k
(n_lid=33280) expect ~15-20x for W=8 (needs a deeper dump to confirm; trend
monotonic). FA is 44.8% of GPU @d131k and it's the depth-scaling half.

Graph today (deepseek4.cpp:779-784): build_top_k_mask scatters -inf/0 into a
DENSE [n_csa] mask, concat with raw-window mask, build_attn_mha over full
k_all. FA scans every CSA tile; per-query top-512 sparsity is invisible to it.
This is the MoBA problem exactly.

Staging:
- B1 (decode, nt=1, THIS iteration): one query -> gather its 512 CSA rows via
  ggml_get_rows into a [d,512] tensor, FA over 512 + raw window instead of the
  dense scan. Pure graph-level (get_rows + smaller build_attn_mha), no kernel.
  Depth-flat decode CSA attention. ~12ms/token -> <1ms projected.
- B2 (prefill, follow-on): per-W-tile union gather + membership mask over the
  ~1148 gathered rows (not 7680). Needs a varlen/segmented FA variant or the
  MoBA re-param. Bigger; deferred behind B1 + A.

### fp4 score kernel
CPU ref: ggml-cpu/ops.cpp:8337 (relu-weighted-sum + partial_sort, mirrors CUDA).
Device e2m1 dequant: ds4_cuda.cu:4589 (dsv4_e2m1fn_dequant_dev) — switch-table,
even-index tie-break. mma.cuh:1126 mma_block_scaled_fp4 (m16n8k64 e2m1.e2m1,
ue8m0 for MXFP4 / ue4m3 for NVFP4) is BLACKWELL_MMA_AVAILABLE and sm121a builds.

Staging:
- A1 (THIS iteration, numerics only, NO speedup): LLAMA_DSV4_LID_FP4=1 applies
  e2m1 block-32 round-trip to q and k in-op (they arrive already hadamard-
  rotated at deepseek4.cpp:612, so op does e2m1 ONLY — matches oracle
  --pre-rotated) before the existing wmma/scalar score kernels. CPU ref mirrors
  it. Purpose: validate device e2m1 path + set the accuracy bar via greedy A/B
  + KL on the resident model. Oracle says 0.93 set overlap vs current fp16, and
  since ds4.c applies the round-trip unconditionally this path is CLOSER to the
  official graph, not further.
- A2 (follow-on, the actual speedup): real block-scaled fp4 mma score kernel
  (e2m1-pack q/k, ue8m0 scales, mma_block_scaled_fp4). ~2-4x on the compute-
  bound nt=2048 path (34.9ms measured); fp4 K storage also cuts the memory-
  bound decode path (0.93ms @ 8.7GB/s). Gated on A1 accuracy PASS.

## Iteration 14a — A1 fp4 fake-quant path DONE; accuracy PASS, A2 green-lit

Implemented LLAMA_DSV4_LID_FP4=1: e2m1 block-32 fake-quant of q and k inside
the fused lid_topk op (dsv4_fp4_quant_kernel + dsv4_e2m1_dequant device funcs;
CPU mirror dsv4_fp4_quant_row_cpu in ops.cpp). Numerics-only, no speedup.

Gates:
- test-backend-ops DSV4_LID_TOPK: non-fp4 12/12 OK (no regression); fp4 path
  (LLAMA_DSV4_LID_FP4=1) all OK — device e2m1 == CPU e2m1 exactly.
- Shallow PPL (-c 2048): 3.5139 identical both ways — EXPECTED, n_lid<=top_k so
  selection is a no-op (everything selected); not a real test.
- Greedy A/B at ~10k depth on an instruction-less doc dump: DIVERGED (English
  vs Chinese continuation). Knife's-edge ambiguous prompt; first-token flip
  cascades. Weak signal.
- DEEP PPL (-c 16384, selection active, n_lid up to 4096 >> top_k 512):
    fp4 off 2.8744 +/- 0.0316
    fp4 on  2.8751 +/- 0.0317   delta +0.0007 (+0.024%, ~sigma/45)
  Statistically identical. 7% of selected tokens differ (0.93 overlap) but
  they're low-attention-mass -> no quality cost. Confirms the oracle theory.

VERDICT: fp4 indexer is quality-safe (model is QAT'd for it). A2 (real
block-scaled fp4 mma score kernel, the actual ~2-4x speedup) is GREEN-LIT.
A1 stays in tree env-gated off as the numerics scaffold + accuracy harness.

## Iteration 14b — B1 decode-side gathered CSA attention (env-gated)

LLAMA_DSV4_CSA_GATHER=1 (deepseek4.cpp build_csa_lid_attention): for nt_s==1
(decode) and n_csa>n_top_k, replace the dense [n_csa] top-k mask + FA-over-all-
CSA with a ggml_get_rows gather of the n_top_k selected CSA rows, then FA over
just those + the raw window. Per-stream batched gather (a=[hd,n_csa,n_stream,1]
indexed by top_k=[n_top_k,n_stream,1,1]); gathered K cast to raw_k type; zero
mask for the selected block (valid by construction at depth). Prefill (nt_s>1)
keeps the mask path — per-token index divergence within a tile needs the union
path (B2, deferred).

Gates:
- Builds clean; decode greedy A/B (gather off vs on, ~16k depth): first ~25
  tokens BIT-IDENTICAL, then fp16 FA reassociation drift (gather does FA over
  n_top_k+window tiles vs mask path's ~4700 — same selected set + softmax
  denominator, different accumulation order). Wrong-cell bug would flip token
  #1, not #26 -> gather is correct; drift is benign fp non-assoc (same class
  as HC-fused / fp4). NOTE: PPL can't test this path (PPL is all nt>1 = mask
  path); generation-based validation only.
- tg64 @ d32768: 12.32 -> 13.17 (+6.9%, r=2). Win is CAPPED because gather
  only fixes CSA FA, not the decode indexer score (16.5ms/tok @d131k, that's
  A2). d131072 gather-on leg OOM'd mid-fill (earlyoom) before recording; off
  leg = 9.40. Re-measure deeper win at a conservative depth (d65536) later.

CAUTION: d131k tg bench (131072-ctx KV alloc) triggered earlyoom this session.
Keep depth benches <= d65536 unless watching free mem; matches
gb10-session-crash-causes (never provoke earlyoom).

### B1 depth trend confirmed (d65536, conservative depth, no OOM)
    tg64 gather off/on:  d32768 12.32->13.17 (+6.9%)   d65536 10.78->11.67 (+8.3%)
Win GROWS with depth (dense CSA FA scan scales linearly; gather pinned at
n_top_k=512) -> confirms depth-independence. d131k+ win larger but capped by
the decode indexer score kernel (A2). 101G free throughout the d65536 pair.

## Iteration 14c — A2 PIVOT: score kernel is L1/smem-bound, not compute-bound

ncu on dsv4_score_wmma128 (perf case nt=2048 n_lid=8704, --replay-mode
application, no model -> no earlyoom):
    Memory throughput   81.4%   <- BOUND
    L1/TEX throughput   80.4%   <- real limiter (smem reads by wmma)
    Compute (SM)        14.1%   <- mma 86% idle
    L2                   7.25%  (K reused from L1)
    Achieved occupancy  32.7%   (smem-limited, ~2 CTA/SM)
    Duration            29.7ms

=> fp4/int8 ON THE MMA gives ~0 (tensor cores already idle). The lever is
SMEM DATA WIDTH: storing K in smem as int8/fp4 (8/4-bit vs 16-bit fp16) cuts
the L1/smem traffic that bounds the kernel AND shrinks smem footprint to raise
occupancy. The block-scaled fp4/int8 mma is the vehicle (reads narrow data
straight from smem, no dequant), but the WIN is memory, not throughput.

A2 plan (revised): int8 mma score kernel first (LLAMA_DSV4_LID_INT8) — halves
K smem width, low risk, standard ggml mma:: tiles, projected ~1.5-1.8x since
L1 is the bound and K dominates smem (b_sh 128x128 vs a_sh 16x128). fp4 mma
follow-on doubles the smem win again (~2.5-3x) at the cost of the interleaved-
nibble + e8m0 packing (quantize_mmq_mxfp4 is the reference). Epilogue is scalar
(14% compute) so watch it becoming the new bound after the memory fix.
Earlier "fp4 for tensor throughput" framing was WRONG — corrected here.

## Iteration 14d — A2 int8 dp4a score kernel DONE (first cut, +3.2% prefill)

dsv4_score_int8_kernel (LLAMA_DSV4_LID_INT8=1): K quantized int8 in smem
(16KB vs fp16 32KB) + per-head int8 q, dp4a dots, per-row scales applied after
int32 accum. Attacks the L1/smem-bandwidth bound (14c), no exotic ldmatrix.

Gates:
- test-backend-ops DSV4_LID_TOPK: 12/12 OK (int8 vs fp32 CPU ref; set-mismatch
  <=0.7% so bumped the d_idx==128 tolerance to 1.2e-2 for the int8 env — int8
  error 0.5% << fp4's 7% which was already PPL-neutral, so quality-safe).
  Non-int8 regression 12/12 unchanged.
- ncu int8 vs wmma: kernel 29.66 -> 21.76ms (1.36x). Still L1-bound (88% mem,
  88% L1) but occupancy 32.7->49.7%, compute 14->43%. K still dominates smem
  reads -> fp4 (quarter K) is the next increment for more.
- perf: nt2048/n_lid8704 34.97->28.09ms (1.24x); n_lid33280 126.5->102.9;
  decode nt1 0.926->0.769 (1.20x).
- e2e: pp2048@d32768 336.3->347.1 (+3.2%, grows with depth; score is 24% @d131k
  vs 13.5% @d32k). tg@d32768 12.54->12.71 (int8 alone +1.4% — nt=1 still pads to
  a 16-token tile, 15/16 wasted; dedicated decode kernel is a separate fix).
  int8+GATHER tg@d32768 13.33 (+6.3% over baseline — stacks with B1).

A2 next increments: (1) fp4 mma score kernel (quarter K smem; interleaved-
nibble + e8m0 packing per quantize_mmq_mxfp4; still L1-bound so real headroom);
(2) dedicated nt=1 decode score kernel (kill the 15/16 tile-padding waste).

## Iteration 15 — int4 score kernel NEGATIVE RESULT; fp4-mma not worth it

Probed a 4-bit-K-in-smem dp4a kernel (LLAMA_DSV4_LID_INT4): K stored 8KB (vs
int8 16KB), unsigned-nibble + bias-correction (sum(q*k) = dp4a(q,k+8) - 8*qsum),
spread nibbles -> int8x4 for dp4a. Two packings tried:
- uint16 reads (4 dims/read): 47.0ms
- uint32 reads (8 dims/read, halved read count): 45.6ms
Both ~1.6x SLOWER than int8 (28.1ms) and slower than wmma (35.0ms). Since int4
moves HALF the K bytes yet runs slower, it is compute-bound on the nibble
UNPACK (spread ~10 ALU ops x 32/dot x 8 comps x 64 heads). dp4a needs int8
operands, so any sub-8-bit storage pays an unpack tax that exceeds the L1
saving. Accuracy also coarse (~7% set-mismatch, like fp4).

=> The only unpack-free 4-bit path is the native block-scaled fp4 MMA (reads
e2m1 straight into tensor cores via ldmatrix). But ncu (14c) showed compute is
IDLE and the bound is L1 — freeing the mma doesn't help, and ldmatrix reads are
word-granular so the 4-bit L1 saving is not guaranteed. Verdict: fp4 mma is NOT
worth the high-risk packing effort; int8 dp4a is the score-kernel sweet spot.
int4 code reverted (kept only as this documented negative result).
Reallocating the fp4-mma budget to the decode kernel + B2 (higher value).

## Iteration 16 — dedicated decode score kernel (LLAMA_DSV4_LID_DEC) +5.2%

Probe 1 (cheap): route decode to the existing scalar kernel -> SLOWER (1304 vs
930us) — only 65 blocks, GPU starved. Decode bottleneck = parallelism + the
15/16 tile-padding waste, not the kernel math.

dsv4_score_decode_kernel (nt_s==1): warp-per-comp, q (all heads) int8-quantized
into smem once and reused, each warp streams one comp's K coalesced (4 dims/
lane), int8 dot per head via warp-shuffle reduce, grid-strided (cap 512 blocks)
for high occupancy. No token padding.

Gates:
- test-backend-ops: added 2 d_idx=128 nt_s=1 cases (n_stream 1 & 2); DEC path OK
  (int8 tolerance 1.2e-2). Baseline wmma also OK on the new cases.
- perf nt=1 n_lid=33280: wmma 935us -> int8 773 -> DEC 470us (2.0x vs wmma,
  1.6x vs int8).
- e2e tg@d32768: baseline 12.56 -> DEC 13.21 (+5.2%, vs int8's +1.4%);
  DEC+GATHER 13.91 (+10.8% over baseline). Grows with depth (score is a bigger
  decode share at d131k). 470us still >> the ~30us memory-ideal K read, so more
  headroom remains, but 2x banked.

## Iteration 17b — B2 refined design (shapes probed): tractable via 2 bitmap ops

Runtime shapes (il=2, ~10k prefill): head_dim=512 (MLA), raw_k=[512,1,2304,1]
(n_raw=2304 bounded window), csa_k=[512,1,4096,1] (n_csa=ctx/4), and CRUCIALLY
prefill CSA is called with nt_s<=16 (already chunked into natural W<=16 tiles).

This collapses B2 from a fused-FA kernel into a MINIMAL edit of the existing
dense path. Per call (nt_s<=16 tokens = one union tile):
  1. union op: top_k[512,nt_s] -> union_idx[U] (sorted unique union, padded)
  2. gather:  get_rows(csa_k, union_idx) -> [512, U]   (shared across nt_s)
  3. k_all = concat(raw_k[512,2304], union_csa[512,U])  (raw reused as-is)
  4. memb op: top_k + union_idx -> membership[U, nt_s] (0 if selected else -inf)
  5. mask  = concat(raw_mask[2304,nt_s], membership[U,nt_s])
  6. FA(q, k_all, mask)  -- SAME call as dense, just U instead of n_csa
Steps 3/5/6 already exist in build_csa_lid_attention; only 1,2,4 are new, and 2
is plain get_rows. So B2 = 2 small bitmap ops (union_idx, membership).

Why not the alternatives (ruled out this iteration):
- tile-skip FA: selections scatter across all K-tiles -> nothing fully empty
  (this is exactly why the dense mask FA is already dense, iter 13).
- per-token gather (q=1 batched FA): FA runs at ~1/16 tile efficiency -> a 12x
  data cut becomes ~0.8x wall. Union (shared W=16 K) keeps Q-tiles efficient.
- raw window (2304) is shared+dense, can't cheaply separate from gathered CSA
  in ggml FA (no LSE merge) -> keep raw in k_all, gather only CSA.

U per call: nt_s=16 union ~500-1500 (grows sublinearly); cut vs n_csa grows with
depth (n_csa=32768 @d131k, U~1500 -> ~20x on the CSA portion; raw 2304 fixed).
Building op 1 (union_idx) next.

## Iteration 18 — B2 union path INTEGRATED + validated (+14.6% prefill)

Two ggml ops (dsv4_lid_union, dsv4_lid_memb; CPU+CUDA, smem bitmap, backend-ops
5/5 each incl overflow + deep n_csa=32768) + get_rows gather wired into
build_csa_lid_attention (LLAMA_DSV4_CSA_UNION, nt_s>1, cap LLAMA_DSV4_CSA_UNION_CAP
default 2048).

Gates:
- CORRECTNESS proven: with cap>=n_csa (no overflow, u_max=n_csa) greedy A/B vs
  dense is TOKEN-IDENTICAL -> union+gather+membership reproduces the dense
  computation exactly. So the logic is correct; cap=2048 divergence is purely
  the overflow-drop approximation (union>cap drops highest-index cells).
- PERF: pp2048@d32768 dense 332.6 -> union(cap2048) 381.1 = +14.6%. Should grow
  with depth (CSA cut ~ n_csa/cap; n_csa=32768 @d131k).
- REMAINING: quality gate for the cap=2048 overflow approximation (deep PPL vs
  dense) — crashed/oomd before capturing; rerun at safe depth. And depth-trend
  perf at d65536.
CAUTION: deep PPL (c16384 long corpus) likely triggered the OOM this session.

## Iteration 18b — CORRECTION: B2 integration has a bug; reverted (ops kept)

The +14.6% integration (afdc029ef) was committed with a FALSE correctness claim.
The "cap8192 token-identical" proof was INVALID: n_csa~4096 < cap 8192 made the
gate `n_csa > cap` false, so the union path never activated — cap8192 silently
ran dense. Proper test (env LLAMA_DSV4_CSA_UNION_FULL forcing u_max=n_csa, so
NO overflow -> must equal dense if correct): full-union DIFFERS from dense and
emits garbage ("[end of file]", NaN-like). So the get_rows+concat+memb+FA
integration is BUGGY. Dense verified deterministic (run1==run2), so the diff is
a real bug not FA noise.

Likely cause: the dense path adds the base CSA mask (inp_csa.kq_mask) inside
build_top_k_mask (ggml_add at deepseek4.cpp ~L726); the membership op omits it,
so a top-k-selected cell that is CSA-causally-masked (or padding) gets attended
-> corrupt softmax. Fix: gather+add the base CSA mask per union cell (or fold it
into the memb op), then re-verify full-union == dense before any perf/cap work.

Status: integration reverted to ops-only (c0e091d68). union_idx + membership
ops remain VALIDATED in isolation (backend-ops 5/5 each, CPU==CUDA incl overflow
+ deep n_csa=32768) and committed. The +14.6% perf datapoint is real but on
INCORRECT output — do not trust until the integration bug is fixed. Clear repro:
LLAMA_DSV4_CSA_UNION=1 LLAMA_DSV4_CSA_UNION_FULL=1 greedy A/B vs dense.

## Iteration 18c — B2 debug: bug isolated to CUMULATIVE (single-layer correct)

Extensive bisection of the full-union (u_max=n_csa, no overflow, must==dense)
garbage. Every component VERIFIED CORRECT:
- memb values: zeros_tok0/1024/2047 = 512 each (exactly n_top_k). Correct.
- gather/memb ALIGNMENT: union_idx[rank(c)]==c confirmed on device. Correct.
- shapes: raw_mask/memb ne match (both [.,nt_s,1,n_stream] F16). Correct.
RULED OUT: ggml_cast F32->F16 (Mode A + cast still works); base CSA mask (dense
is coherent WITHOUT the build_top_k_mask base add); FA KV_max opt (disabled,
still garbage); -inf vs finite -50000 (both garbage); memory corruption (short
prompt / small n_csa where memb is all-zero is coherent); CUDA graphs (prefill
doesn't use them).

KEY ISOLATION: union on a SINGLE layer (LLAMA_DSV4_UNION_IL=2 or 4) is COHERENT
and close to dense. Union on ALL layers -> garbage. So each layer's op is
correct; the failure is CUMULATIVE across layers. Also: Mode A (compaction +
ZEROS mask = attend-all-union) works on all layers; only compaction + the
SPARSE membership mask fails cumulatively. Real correctness structure is sound
(1-layer ~= dense); the bug is a subtle multi-layer interaction — leading
theories: (a) peaked-attention numerical drift amplifying across 21 layers
(reordered union-K vs dense cell-order K), (b) an allocator/compute-buffer
reuse issue when many union subgraphs coexist. Next: per-layer output diff
(eval-callback) dense vs union, or compute-sanitizer, to find where NaN first
appears.

Status: integration kept env-gated OFF (LLAMA_DSV4_CSA_UNION, default off);
ops validated + committed. Do NOT enable until the cumulative bug is fixed.

## Iteration 18d — B2 debug CRACKED (partial) + design reframe

Two findings from deep bisection of the union garbage:

1. CUMULATIVE BUG FIXED (memb rewrite). Root cause: the original memb kernel
   REBUILT its own bitmap and computed rank independently of the union op's
   emit order. Despite passing token-0 spot checks, the two bitmaps could
   disagree on some patterns in deeper layers -> gather/mask misalignment that
   compounded across 21 layers into garbage (single-layer was fine). Rewrote
   memb to use union_idx DIRECTLY via binary search (parallel, TPB=8 grid) ->
   gather order and mask are guaranteed consistent. Also ~10x faster. Backend-
   ops 5/5. Cap-mode is now COHERENT (was garbage).

2. DESIGN REFRAME (the real blocker). The CSA attention is called with
   nt_s=2048 (the FULL ubatch), NOT nt_s<=16 as the shape-probe (iter 17b)
   suggested. Instrumented union sizes: at nt_s=2048, union_cnt = 1881-2048 of
   n_csa=2048 (92-100%). So the whole-ubatch union covers ~ALL cells -> ZERO
   compaction. The +14.6% (iter 18) was from cap-TRUNCATION (drop cells beyond
   cap by lowest-index) = a coarse, wrong approximation, not tight per-tile
   union. B2's compaction only exists at small W (iter 17: W=16 -> 5x). So a
   correct+fast B2 MUST sub-tile the nt_s=2048 call into W<=16 groups, each with
   its own union + gathered K, via a BATCHED FA over nt_s/W tiles. That's the
   harder design scoped in iter 17 (per-tile gather + batched flash_attn_ext).

3. Residual exact-mode bug: full-union (u_max=n_csa) still garbage while cap
   (u_max<union, no padding) is coherent. Isolated to the PADDING slots
   (union_idx padded with n_csa-1 when u_max>union_cnt): masking those all-(-inf)
   duplicate columns breaks FA (attending them via zeros mask is fine). Dense
   has scattered all-(-inf) cells and works; the duplicate-contiguous padding
   is the trigger. Unresolved; avoided in cap mode (padding-free by
   construction since union>cap there).

STATUS: memb rewrite is a real fix worth keeping (ops faster + cap-mode
coherent). Integration stays env-gated OFF: it gives no compaction at nt_s=2048
and cap-truncation is a poor approximation. Correct B2 = per-W-tile batched FA
(next focused build). Ops (union_idx + memb) validated + reusable for it.

## Iteration 19 — B2 PER-TILE built end-to-end (tiles = virtual streams)

Probes first (all model-free, test-backend-ops):
- FA EXONERATED: test_fa_pad (trailing fully-masked duplicate KV columns = the
  union padding pattern; mid-tile start, nb=2048, sinks, batched nr3=128) —
  18/18 OK on CUDA. The iter-18d "padding breaks FA" attribution was wrong;
  the residual full-union bug is in the (now superseded) whole-batch model
  path, NOT the FA kernel. Learned: CUDA FA requires total kv 256-aligned
  (n_raw 2304 + u_cap 2048 = 4352 OK).
- PERF GO/NO-GO (hsk=hsv=512, nh=1, nr2=64, sinks, real geometry):
    d32k  dense kv=10496 nb=2048: 69.0ms 40.8 TFLOPS
    d32k  tiled kv=3328  T=128  : 74.1ms 12.1 TFLOPS  -> 0.93x LOSES
    d131k dense kv=35072 nb=2048: 230.0ms 40.9 TFLOPS
    d131k tiled kv=3840  T=128  : 85.1ms  12.1 TFLOPS -> 2.70x
    d131k tiled kv=4352  T=128  : 95.9ms             -> 2.40x
    d131k tiled kv=5376  T=32 (W=64): 117.7ms        -> 1.95x
  Tiled FA runs at ~12 TFLOPS vs 41 dense (small-nb mma config; flat across
  W=16/64) -> depth-gate required; break-even ~n_csa 16k. FUTURE LEVER:
  close the 12->41 TFLOPS gap (fattn ncols config for small nb) => tiled
  would win everywhere and ~9x at d131k.
- PRECEDENT (../quartile attn.rs): attend_var_m_shared = the SAME design
  (shared gathered KV union + per-row additive mask into it, sinks in the
  softmax max), validated there; pads with finite -1e30 (not -inf); its docs
  note tiled-online reassociation is NOT bit-exact — expected, matches A/B.

Build (per-tile union via dim-3 "virtual streams" — build_attn_mha derives
n_stream from k->ne[3] and splits Q/reassembles out for free; token order is
already tile-major):
- ggml_dsv4_lid_union(+W): per-tile unions [u_max, T, 1, ns], T=ceil(nt_s/W),
  partial last tile OK; memb reads W from uni op_params -> per-tile membership
  (same out shape; reshape to [u_max, W, 1, T] is free). CUDA grid (T, ns);
  memb reloads smem union on tile crossing (never for W%8==0). CPU refs match.
  backend-ops 20/20 (W=0 back-compat + W=16/8, partial, overflow, streams, deep).
- graph (LLAMA_DSV4_CSA_TILE=W, default off): flat-ids get_rows gather of all
  per-tile unions, repeat_4d raw window across tiles, concat dim2 ->
  k_all [hd,1,n_raw+u_cap,T]; mask = reshape(raw_mask)||memb [.,W,1,T].
  Gates: n_stream==1, nt_s%W==0, n_csa>=LLAMA_DSV4_CSA_TILE_MIN (default
  12288), u_cap(LLAMA_DSV4_CSA_TILE_UCAP, default 2048) <n_csa, 256-alignment,
  raw_mask ne1==nt_s. Replaces the whole-batch union branch (cap-truncation
  mode retired).
- tooling: experiments/ds4-tile/ab.sh — greedy prefill-path A/B vs banked
  baseline (prompt-cache would skip prefill; full reprocess ~50s/leg).

Gates so far:
- ab.sh tile16 (TILE_MIN=2048, 15388-tok prompt): output COHERENT, first
  stretch token-identical then plausible divergence; rerun == rerun
  (DETERMINISTIC). Divergence = fp reassociation of the reduced/reordered
  KV set (quartile precedent) — bit-exactness not expected for per-tile.
- perf @ shallow (n_csa<=3847, forced on): 281.8 vs 292.0 t/s dense — the
  expected below-gate loss; real gate is d65536+.
PENDING: pp2048@d65536 dense vs tile (running), d131k leg (-r 1, OOM-careful),
union overflow stats at depth (u_cap 2048 headroom), PPL quality gate at safe
depth.

## Iteration 19b — bench crash post-mortem + union stats instrumentation

- First d65536 bench pair CRASHED (both legs, incl pure dense): CUDA abort at
  a relu launch during the first fill. Cause: bench legs launched WITHOUT
  LLAMA_DSV4_FUSED_LID=1 -> the unfused indexer chain's O(n_ctx x nt x n_head)
  relu intermediate (~8.6GB @d65536) OOMed the pool. Operator error, not a
  regression; all depth benches MUST carry LLAMA_DSV4_FUSED_LID=1.
- Added LLAMA_DSV4_UNION_STATS=1: union kernel optionally popcounts the FULL
  bitmap per tile (exact union size incl cells beyond u_max) into a pool
  buffer; host prints per-call "US n_csa W T u_max cnt min/mean/max over".
  Measures true overflow magnitude, not just at-capacity. Zero cost when off
  (nullptr stats arg). Smoke-tested via backend-ops (random-index test unions
  sit near max, as expected for uniform data).

## Iteration 19c — d65536 GATE + union overflow MEASURED (u_cap 2048 is not exact)

pp2048@d65536 (fused lid, -r 2): dense 232.3 -> tile16(u_cap2048) 271.5 = +16.8%.
First correct-construction B2 win at depth (v1/v2 bench crashes were missing
LLAMA_DSV4_FUSED_LID=1 -> unfused relu OOM; plus zsh `VAR=1 eval` drops the env).

UNION_STATS sweep (W=16, TILE_MIN=2048, n_csa 2560->16896, 21 calls x 128 tiles):
  n_csa  mean  max   over(u_max=2048)
  2560   1366  2180  1%
  4096   1631  2916  27%
  8192   2030  3892  48%
  16384  2446  4942  60%
Union mean grows ~ +250 per n_csa doubling (sublinear, as designed) but the
u_cap=2048 overflow is SYSTEMATIC at depth, not rare: 60% of tiles drop cells
at d65k. Drops are highest-index (most recent) selected cells — partially
shadowed by the 2304 raw window, but semantic until proven otherwise
(ds4.c contract comment: dropped selected rows = semantic loss class).
=> +16.8% is PROVISIONAL pending a quality gate.
Exactness at d65k needs u_cap ~5376 -> kv_tile 7680; at the 12-vs-41 TFLOPS
small-nb FA penalty that erases the d65k win (survives only d131k+). The
strategic lever is the fattn small-nb config gap (ncols1<=16 starves K-byte
amortization): closing half of it makes EXACT per-tile B2 win from d32k up.
Running: u_cap 3072/4096 @d65536 + dense/2048/4096 @d131072 sweep legs.

NOTE (2026-07-13, Teej): quant of record is UD-IQ3_XXS (the serving quant),
not the UD-IQ2_XXS the campaign targets were baselined on. Structural results
(ops, union stats, FA ms deltas, the small-nb fattn lever) transfer; headline
% gains do NOT (IQ3's slower MoE shrinks FA's share -> smaller % for the same
FA savings). All DECISION gates from here run on IQ3: dense-vs-tile d65536,
stats confirm, quality gates. ab.sh default switched to IQ3 (AB_MODEL to
override). IQ3 is 96GB resident: d131k legs need a headroom check first
(earlyoom line; see gb10 crash memory). A/B baselines must be re-banked on
IQ3 before any variant comparison (old IQ2 outputs are not comparable).

## Iteration 19d — u_cap/depth sweep complete (IQ2): honest caps keep the win; d131k +37/+32%

pp2048 (IQ2, fused lid): u_cap curve is SHALLOW —
  d65536:  dense 232.3 | 2048: 271.5 (+16.8%) | 3072: 265.9 (+14.5%) | 4096: 260.6 (+12.2%)
  d131072: dense 166.4 | 2048: 228.3 (+37.2%) |                        4096: 220.2 (+32.3%)
Doubling u_cap 2048->4096 costs ~1/4 of the win at both depths (FA is diluted
in total pp; the backend-ops-probe pessimism priced FA as the whole pipeline).
u_cap 4096 covers mean+tail at d65k (only ~5k-tail tiles truncate, few %) and
keeps +12.2% @d65k / +32.3% @d131k. d131k pp stable with fused lid (no OOM).
QUANT SWITCH: decision legs now on UD-IQ3_XXS (serving quant; Teej).
Running: IQ3 dense/2048/4096 + stats sweep @d65536.

## Iteration 19e — IQ3 decision legs: same story as IQ2; u_cap default -> 4096

pp2048@d65536 (UD-IQ3_XXS 95.9GB, fused lid): dense 225.3 | tile16 u_cap2048
263.2 (+16.8%) | u_cap4096 252.7 (+12.2%). Same percentages as IQ2 to a tenth:
the FA savings and MoE slowdown scale together at this depth.
IQ3 union stats == IQ2 within noise (mean 2410 vs 2446 @n_csa16384, over 59%
vs 60%) -> indexer selection clustering barely changes with expert quant; all IQ2
structural findings transfer.
DECISION: u_cap default 4096 (covers mean+bulk of tail; tail few % of tiles
still truncate at d65k+). Committed. Tile path remains opt-in
(LLAMA_DSV4_CSA_TILE=16) until quality gates pass.
Running: IQ3 ab.sh re-bank + tile A/B, IQ3 d131k dense/4096 pair.
Next gates: passkey-at-depth (plant beyond raw window, the dropped-cell victim
region) + PPL at safe ctx, both IQ3.

## Iteration 19f — IQ3 decision legs COMPLETE; A/B false alarm root-caused

pp2048 IQ3 (fused lid): d65536 dense 225.3 | cap2048 263.2 (+16.8%) | cap4096
252.7 (+12.2%). d131072 dense 163.1 | cap4096 215.4 (+32.1%) — matches IQ2's
+32.3% at the same depth. Union stats match IQ2. No stability issues at 96GB.

A/B false alarm: IQ3 tile16-vs-baseline "DIVERGES" was NOT the tile path —
UNION_STATS proved the gate never opened (padded n_csa <= 4096 == default
u_cap on the 15k prompt; ne[0] is the PADDED mask width, cells pad to 2048
multiples). Cross-diffing exposed the real cause: baseline != baseline2 —
DENSE ITSELF varies across process launches (first run right after a bench
cycle; kernel/algorithm selection shifts with free-memory state). Runs in
similar machine state agree (baseline2 == tile16 == tile16c).
=> GATE POLICY: cross-process transcript identity is NOT a valid gate on this
box. Valid gates: within-config determinism (proven IQ2+IQ3), coherence,
PPL/KL, passkey-at-depth. ab.sh prompt (15k) needs UCAP=2048 override or a
>16k prompt to exercise the tile path at the 4096 default.
Tile-ACTIVE IQ3 A/B (cap2048+stats): 63 US lines, coherent output. Banked.

STATE: B2 per-tile SHIPS opt-in (LLAMA_DSV4_CSA_TILE=16, u_cap 4096,
TILE_MIN 12288): IQ3 +12.2% @d65k, +32.1% @d131k, correct-construction.
REMAINING gates to default-on: passkey-at-depth + PPL (safe ctx), both IQ3.
NEXT LEVER: fattn small-nb column config (12->41 TFLOPS headroom) -> exact
u_cap affordable everywhere + bigger wins; then 256k/512k depth points.

## Iteration 19g — PPL gate PASSES (both caps); scout of follow-up levers

PPL c32768 IQ3 (docs corpus, fused lid): dense 2.4348 +/- 0.0226 | tile16
cap4096 2.4371 (+0.1 sigma) | cap2048 2.4343 (-0.02 sigma). Statistically
identical even at cap2048's ~48% tile truncation rate. Caveat (ds4.c contract
comment): PPL can miss tail retrieval -> passkey-at-depth is the decisive
gate (v2 running; v1 was harness error — 185KB prompt overran c49152 at
~3.6 chars/tok for markdown, stderr was discarded. v2: 150KB + stderr kept).

Scout results (follow-up levers, ordered by payoff):
1. fattn small-nb stalls (12->41 TFLOPS): dispatch picks the SAME config for
   dense and tiled (ncols2=8 via gqa>4, ncols1=64/8=8, 2048 blocks both) —
   NOT column starvation. DRAM ruled out by arithmetic (640MB tile K ~2.3ms
   vs 85ms measured). Hypothesis: short-inner-loop pipeline drain (kv 4352
   vs 18688/block; prologue/softmax-rescale/epilogue amortize 4.3x worse).
   Probe: ncu --replay-mode application on the backend-ops tiled perf case
   (model-free — GB10 ncu hazard doesn't apply). Candidate fix: fatter
   ncols1=16,ncols2=8 instantiation for small-Q ne3-batched shapes.
2. 512k run gate (task #6): iter-13 "FAIL without kernel work" is now
   plausibly unblocked (B2 +32% @d131k, FA share ~60% @512k -> pp proj
   ~90-110 t/s; B1+DEC already flattened decode FA). Plan: d262144 rehearsal
   -> d524288 pp -r1 -> tg64@512k vs >=10 t/s gate -> coherence run.
   Memory ~105-108GB of 121.
3. Streaming chunk-topk (iter-13 item 3): remaining LID lever, ~6% @d131k,
   grows with depth.
4. Overflow policy: truncation drops NEWEST cells; keep-newest variant is a
   ~3-line kernel change — decide from passkey v2 position sensitivity.

## Iteration 19h — passkey gate PASSES; B2 per-tile DEFAULT ON

Passkey v2 (42151-tok prompt, 5 keys at 10/40/70/90/97%, IQ3, greedy):
- tile16 cap4096: 5/5 exact. tile16 cap2048: 5/5 exact — including delta@90%
  (outside raw window, in the truncation-victim region at ~50% tile overflow).
- dense first run emitted instant EOS (0 answer tokens); dense RERUN: 5/5.
  Same cross-process greedy variance as 19f (one-ulp flip at token 1) — dense
  scored as pass on the rerun; variance precedent now observed twice.
ALL GATES PASS: PPL identical (both caps), passkey 5/5 (all configs),
within-config determinism, coherence. Truncation direction (drop-newest)
empirically harmless at these depths -> lever #4 (keep-newest) deprioritized.

FLIPPED DEFAULT: LLAMA_DSV4_CSA_TILE now defaults to 16 (0 disables); u_cap
4096, TILE_MIN 12288 (self-gates off shallow). Shallow default-path sanity OK.
Net effect at defaults: pp unchanged <=d49k, +12.2% @d65k, +32.1% @d131k (IQ3).

## Iteration 20 — three-lever scout: probes + ncu profiles (root causes found)

LEVER 1 (fattn small-nb, 12->41 TFLOPS) — ROOT CAUSE FOUND: LPDDR latency
wall, not config, not kv length, not bandwidth.
- Discriminating probes (kv=4352, total Q=2048 fixed): T=1/nb=2048 43.8
  TFLOPS | T=4/nb=512 35.9 | T=16/nb=128 12.2 | T=32..128 12.2-12.3.
  Same kernel every case (nsys: flash_attn_ext_f16<512,512,8,8>, no combine
  pass), same 2048 blocks, same per-block work. Collapse is keyed to the
  NUMBER OF DISTINCT K STREAMS (ne3), cliff between T=4 and T=16.
- ncu (application replay, model-free): both cases 8 warps/SM occupancy (1
  block/SM, no latency-hiding headroom). Dense: SM 36%/L2 47%, scoreboard
  6.1 cyc. Tiled: SM 11%/mem 15%/L2 20% — ALL idle — scoreboard 21.0 cyc
  (49%) + CTA barrier 15.4 cyc (36%). Dense's 2048 blocks stream ONE K in
  near-lockstep = L2-broadcast amortization; tiled has 16 blocks per K
  stream (~3 streams interleaved across 48 SMs) = raw LPDDR5X latency,
  nothing hides it. Bandwidth explicitly NOT the limit (640MB=2.3ms vs 96ms).
- CEILING: tiled FA at dense TFLOPS -> d131k FA 85->25ms -> pp +32% becomes
  ~+65%. Fix candidates: (a) K ne3-broadcast in FA (raw window = 53% of each
  tile's kv is IDENTICAL across tiles; serving it from one shared ne3=1
  tensor restores L2-broadcast for half the traffic; also an upstream-useful
  feature = shared-prefix batched decode; needs ggml assert relax + mma
  pointer math + LSE-combine story), (b) deeper K prefetch/2 CTAs per SM
  (register-wall at DKQ=512, low odds).
  Scope (a): ggml.c asserts (+10), fattn-mma-f16.cuh K/V ne3-broadcast
  (+60/-10), fattn.cu plumbing (+20), backend-ops cases (+30); verification:
  FA suite + tiled perf probes + e2e depth bench. Risk: upstream-facing
  kernel, reversible, no data migration.

LEVER 2 (512k run gate): battery RUNNING (bench_512k.log): d262144 tile
(default) + dense pp2048 -r1, d524288 tile pp -r1, d524288 tg64 with
GATHER+DEC+INT8. Memory budget ~105GB/121 at 512k. Gate: tg >= 10 t/s,
pp continues the depth curve.

LEVER 3 (streaming chunk-topk): READ COMPLETE. Current design: chunk kernel
(SORT_N=4096 bitonic) emits INDEX-ONLY candidates; every merge level
RE-READS scores via random 4B gathers (row[idx], uncoalesced) — LPDDR
latency again, worst at decode (few blocks). Iter-13 item 3 = carry
(score,idx) pairs through the candidate tree -> merges become pure-smem.
Selection semantics identical (same values, same idx tie-break) -> existing
backend-ops cases are the correctness gate; DSV4_LID_TOPK perf cases (deep
shapes) are the perf gate, runnable model-free after the 512k battery.
Scope: dsv4_lid_topk.cu chunk/merge/single kernels + scratch sizing (pairs
= 8B, 2x scratch), ~+80/-40 LOC, 1 file. Ceiling: topk 5.8% of pp @d131k,
bigger share of decode + at 512k (32 chunks, 2 merge levels of 4096-wide
random gathers).

## Iteration 20b — 512k battery: pp gates PASS (+62% @262k), tg 9.64 vs >=10

IQ3, fused lid, tile default-on:
  pp2048 @ d262144: dense 103.1 -> tile 166.9 (+61.9%; FA share grows with
  depth exactly as the d131k->d262k projection said)
  pp2048 @ d524288: 115.7 t/s — allocates, runs, no OOM (peak fits ~110GB
  free after); ~2x the iter-13 projection (63). Depth curve tile-on:
  252.7 @65k / 215.4 @131k / 166.9 @262k / 115.7 @512k — graceful.
  tg64 @ d524288 (GATHER+DEC+INT8): 9.64 vs the >=10 gate — 3.6% short.
VERDICT: task-#6 pp gates PASS at 512k; tg gate MARGINAL FAIL. Lever 3
(streaming chunk-topk: index-only candidates force random score gathers in
every merge level; worst at deep decode) is sized right to close the tg gap
(~+80/-40 LOC, 1 file, model-free gates). Lever 1 (fattn ne3-broadcast)
remains the pp prize (+32 -> ~+65% @d131k).

## Iteration 20c — streaming chunk-topk: NEGATIVE RESULT (reverted)

Built the (score,idx)-pair candidate tree (chunk emits pairs, merges become
pure-smem, no score re-gathers; 48/48 backend-ops across default/int8/dec/fp4).
Perf: NO measurable change at ANY shape —
  n_lid=8704  nt=2048: 34.96 -> 34.94 ms
  n_lid=33280 nt=2048: 125.85 -> 125.77 ms
  n_lid=33280 nt=1:    926.0 -> 923.8 us
The iter-13 "carry pairs through the merge tree" hypothesis was wrong about
the bottleneck: the merge tree is BITONIC-COMPUTE-bound (4096-wide sorts),
not gather-bound — at decode the gathers are only ~chunks*top_k ~ 4.6k random
loads, trivial. Reverted (2x scratch for zero gain; int4 precedent).
IMPLICATION for tg@512k (9.64 vs >=10): the 3.6% gap will NOT come from topk.
Next candidates: profile tg at d524288 (nsys) for the real decode breakdown;
DEC score kernel headroom (470us @33k n_lid vs ~30us memory-ideal, scales
~4x at 512k -> ~1.8% of the 103.7ms token budget); FA-decode/B1 path share.
