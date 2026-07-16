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

## Iteration 20d — tg@512k decode breakdown (nsys, 2 tokens before CUPTI drop)

Composition of GPU-busy at d524288 decode (gate envs GATHER+DEC+INT8):
  q8_0 matvecs 20.6% | dsv4_score_decode 18.9% (1.13ms x21 = 23.7ms/tok;
  linear-ish from 470us@33k n_lid) | ELEMENTWISE STORM ~30% (bin_bcast
  add 10.5% + mul 8.2% + div 3.4% + reduce_rows 3.6% + tiny cpys; ~16k
  launches/token at 1-2us) | MoE IQ matvecs ~17% | FA decode 2.5% (B1
  depth-flat, working) | topk <2.5% (20c negative result confirmed in situ).

Two tg levers, quantified:
1. score_decode kernel: 18.9% share, ~9x from memory-ideal (~120us vs
   1.13ms at n_lid=131072); realistic 2x -> ~9% tg. Iter-16 noted the
   headroom; at 512k it's now the top custom kernel.
2. decode micro-kernel storm: ~30% share across ~5k graph ops/token —
   attribution needed (graph node dump at decode: what emits ~257 adds
   /layer/token? suspects: top-k mask add chain remnants, HC scalar chain,
   B1 gather mask build). If it's a fusable chain, this is the bigger lever.
Either alone likely covers the 3.6% gap to the >=10 t/s gate.
Trace: scratchpad tg_d524288.nsys-rep (CUPTI drops decode events ~2 tokens
in — known GB10 limitation, shares valid).

## Iteration 21 — Sinkhorn FUSED (HC mode 2) + HC-fused default ON (graph -74%)

Storm attribution (20d) -> root cause: build_hc_sinkhorn's unrolled chain
(softmax + eps + 20 col/row normalization iterations = ~85 nodes of 1-2us
launches on a [8,8,nt] tensor), x2 per layer — AND the existing HC fused
weighted_sum/post kernels (token-identical, iter gate 2) were still ENV-OPT-IN:
every bench this session ran the scalar HC chains. (Campaign refs carried
LLAMA_DSV4_HC_FUSED=1 — iter-13 d131k dense pp 179.8 vs yesterday's 163.1
reconciles. Ratios were valid; absolutes were handicapped.)

Built: ggml_dsv4_hc_sinkhorn (GGML_OP_DSV4_HC_FUSED mode 2) — one kernel
(block per token, [hc,hc] in smem, serial per-lane sums matching the CPU ref
order) replaces the whole chain incl softmax. CPU ref + CUDA + tests 10/10
(incl nt=2048 prefill width). Fixed supports_op null-src[1] segv (mode 2 has
a single src). HC fused now DEFAULT ON (LLAMA_DSV4_HC_FUSED=0 disables).

Decode graph: 21183 -> 5439 nodes (-74%); SUM_ROWS 3439->85, DIV 3397->43,
ADD 5780->703. Coherent output. Payoff battery running: pp d65k/d131k refs +
tg64 d131k + tg64 d524288 (the >=10 gate that sat at 9.64 with scalar HC).

## Iteration 21b — scouts: two-pass rescore, mxfp4 LID storage, MMQ-DS4 (false lead)

SCOUT 1 — two-pass margin rescore (B -> A for LID_INT8/DEC):
- Pass 1: int8 scores (current kernels) -> top-(512+m) candidates (SORT_N 4096
  handles 576+ fine, merge_group stays >= 2).
- Pass 2: fp32 rescore of ONLY the candidates (warp per candidate, ~60-LOC
  kernel; 576 cells x 64 heads x 128 dims — ~4% of pass-1 flops at prefill,
  trivial at decode) -> exact top-512 via dsv4_topk_single.
- m from measurement: extend the fp4 oracle to rank-displacement of true
  top-512 under int8 ranking (LID_DUMP data); expect m<=32 at 0.5% score err;
  use 64 with safety. Optional debug-mode runtime violation check.
- Result: selection set == fp32 set provably (given bound) -> class A;
  unlocks default-on for both int8 flags with no selection caveat.

SCOUT 2 — mxfp4 LID storage (numerics + storage + decode bandwidth):
- KEY SIMPLIFIER: GGML_TYPE_MXFP4 already exists (e2m1 block-32 + e8m0
  scale = EXACTLY the QAT format), and kv_lid is an ordinary llama_kv_cache
  taking type_k at ctor (llama-kv-cache-dsv4.cpp:1067) -> storage flip is a
  ctor arg + env, NOT a new type. Hadamard is already applied pre-cache in
  the graph, so quantize-on-write == the official QAT numeric.
- Missing pieces: (a) f32->MXFP4 set_rows/cpy CUDA (+~50 LOC; we already
  have the e2m1 quantizer in dsv4_lid_topk.cu, just pack instead of
  fake-quant); (b) score-kernel input: DECODE (DEC kernel) reads packed
  directly — 68B vs 256B per row attacks the 18.9%-of-decode, 9x-off-ideal
  score_decode directly; PREFILL avoids the iter-15 unpack tax entirely via
  a per-ubatch dequant-to-f16 staging pass (n_lid x 128 x 2B ~ 8MB @32k
  ~0.03ms) — wmma/int8 kernels unchanged. NO regression path.
- Cache: 3.76x smaller LID cache (68 vs 256B/row) — matters at 512k-1M.
SCOUT 3 — MMQ_Q8_1_DS_LAYOUT_DS4: FALSE LEAD. "DS4"/"D4"/"D2S6" are q8_1
scale-layout names (delta/sum variants picked per WEIGHT quant), nothing
DeepSeek. No upstream DS4 activation quant exists. MoE activation numerics
(ours MMQ q8_1 vs ds4.c Q8_K) remain a benign 8-bit-vs-8-bit difference.

ENDGAME (composition): mxfp4 storage + int8 first-pass + QAT-numeric rescore
= official-graph-EXACT selection (class A vs the model's trained semantics)
at int8 speed with 3.76x smaller LID cache. Build order: oracle displacement
measurement -> rescore (small) -> mxfp4 storage (medium).

## Iteration 21c — composition scout (official-exact selection) + promotion sweep

THE COMPOSITION (env LLAMA_DSV4_LID_EXACT=1, one coherent pipeline):
  write:  indexer K (post-hadamard f32) --[OUR QAT quantizer]--> MXFP4 blocks
          in kv_lid (ctor type arg). CRITICAL: ggml's quantize_row_mxfp4 does
          NOT match QAT (scale floor(log2 amax)-2 vs QAT ceil(log2(amax/6));
          first-wins vs even-index tie-break) -> custom set_rows write kernel
          (we already own the exact e2m1+ue8m0 quantizer from A1 / ds4.c
          parity). ggml MXFP4 is used as a CONTAINER only (same block layout:
          1 e8m0 byte + 16 nibble bytes per 32).
  q side: A1 QAT fake-quant on q (exists).
  pass 1: int8 ranking for speed —
          prefill: per-ubatch dequant-to-f16 staging (~8MB, ~0.03ms) then the
          UNCHANGED wmma/int8 kernels (sidesteps iter-15 unpack tax);
          decode: DEC kernel reads packed rows directly (68B vs 256B -> the
          18.9%-of-decode, 9x-off-ideal score_decode gets a ~3.8x byte cut;
          kernel is latency-bound so nibble ALU is free).
          Output: top-(512+m) candidates.
  pass 2: fp32 rescore of candidates on EXACT QAT values (dequant matches
          ds4.c bit-for-bit) + mask; exact top-512, lower-index tie-break
          (matches ds4.c indexed_topk). ~60-LOC kernel + dsv4_topk_single.
  m:      measured via oracle rank-displacement (LID_DUMP data), ship 2x
          margin; optional debug runtime violation counter.
  CLAIM SCOPE: SELECTION becomes class A vs the official DeepSeek graph
  (the only official-mandated low-precision numeric). Attention over the
  selected set stays llama.cpp numerics (as does ds4.c's own f16 comp cache).
  Perf: decode score 2-3x -> ~+10-14% tg @512k; prefill ~neutral (+rescore
  ~4% of score, -bytes); lid cache 704MB -> 188MB @512k.
  Scope: set_rows-mxfp4 write (+50), DEC packed reader (+60), staging pass
  (+30), rescore kernel (+60), oracle ext (+40), plumbing/env (+30), tests
  (+40). All in dsv4_lid_topk.cu / llama-kv-cache-dsv4.cpp / deepseek4.cpp.

PROMOTION SWEEP (other B/C -> A candidates):
1. HC sinkhorn C->A (CHEAP, DO with the composition): the fused kernel's
   softmax sum is serial; the old graph's ggml soft_max used a warp-tree
   order. Replicate that tree order in the fused kernel (+~20 LOC) ->
   token-identical vs the pre-fusion graph; HC_FUSED returns to pure A.
2. CSA_TILE tail-truncation B->pure-C (CONFIG ONLY): "exact profile"
   UCAP >= measured max union (5376 @d65k, 256-aligned OK; deeper depths
   need a bigger cap — p100 grows with n_csa, measure at 512k). Costs a few
   % of the tile win; leaves only irreducible FA reassociation (C).
3. CSA_GATHER C->A: NOT practical — would require an exact-order decode FA
   matching the dense kernel's reduction order (infeasible) or defining a
   new canonical order (changes dense too). ds4.c pays a dedicated
   'deterministic' reference kernel for this; not worth it at 2.5% decode
   share. Stays C with determinism gates.
4. MoE activation quant (q8_1 vs official bf16): upstream-wide MMQ design,
   shared by all models, ds4.c deviates equally (Q8_K). Out of scope.
5. Already A: MTP_FUSED_DRAFT (ids-identical), spec decode (target-exact),
   FUSED_LID, HC modes 0/1.

## Iteration 21d — payoff battery complete: 512k tg GATE PASSES (11.08 >= 10)

Full battery (IQ3, tile+HC-fused defaults, gather+dec+int8 for tg):
  pp2048 @ d65536:  303.3 (campaign best; was 252.7 tile-only, 225.3 dense)
  pp2048 @ d131072: 245.1 (+50.2% vs yesterday's dense 163.1)
  tg64   @ d131072: 14.29 (iter-13 baseline was 9.29 -> +54%)
  tg64   @ d524288: 11.08 >= 10 — TASK-#6 tg GATE PASSES (was 9.64 with
  scalar HC; sinkhorn fusion + HC defaults = +15% decode at 512k).
TASK-#6 (512k) STATUS: allocation PASS, pp PASS (115.7 pre-HC-flip, will be
higher now), tg PASS (11.08). Remaining formality: one long-completion
coherence run at 512k. The iter-13 verdict ("BOTH gates FAIL without kernel
work") is now fully reversed by B2 tile + B1 gather + DEC + HC/sinkhorn fusion.
Next: BUILDSPEC-lid-exact.md phases 0-4.

## Iteration 22 — BUILDSPEC phases 0-2: exact selection LANDED (class A, 36/36 @ zero tolerance)

P0 (oracle m): --int8-displacement mode added to fp4_oracle. Synthetic
(n_lid 33k): m p50=15 p99=36 p100=36. REAL DATA (fresh dump, 256 tokens,
n_lid=16384 from a 127k-token prefill): m p50=7 p99=24 p100=26 — default
m=64 (LLAMA_DSV4_LID_RESCORE_M) has 2.5x headroom over worst observed.
Bonus real-data: fp16-vs-QAT overlap 0.9145 at depth; oracle union stats
corroborate the W-tile design (W=16 mean 1412 max 1821 @n_lid 16k).

P1 (sinkhorn C->A): DESCOPED deliberately — matching the legacy soft_max
block-reduce tree is template-shape archaeology for transcript-compat with
a graph that no longer runs. Fused sinkhorn's serial order is canonical
(deterministic, CPU-ref-matched, full gate suite + 512k battery passed).

P2 (two-pass rescore): LANDED. LLAMA_DSV4_LID_EXACT=1 implies the QAT path
(q/k materialized as f32 QAT values — already the case under LID_FP4);
pass 1 = any score kernel (wmma/int8/dec) to top-(512+m); pass 2 =
dsv4_lid_rescore_kernel: serial-order fp32 rescore (bitwise the CPU
reference = official ds4.c order), 1024-wide bitonic, exact top-512
(desc, idx tie-break). backend-ops: 36/36 at ZERO tolerance (set AND
order) across all three pass-1 variants. Selection is now class A vs the
official QAT graph with int8 pass-1 speed. e2e determinism + d65536 perf
spot running.

## Iteration 22b — P2 e2e gates: deterministic + coherent; exact-mode price -28%

e2e (15k prompt, EXACT+INT8): run1 == run2 bit-identical, output coherent.
pp2048@d65536: 217.4 vs 303.3 default = -28.3%. Attribution: the A1 fp4 path
re-materializes ALL of k as f32 QAT every layer-call (8.4MB writes @n_lid16k
+ score kernels forced onto f32 K reads, 2x the f16 bytes) — the documented
"numerics-only" design, now priced at depth. P3a (QAT-at-write: quantize k
once when the cache row is written, store F16-of-QAT, kernels keep the fast
f16 path; needs a small e2m1 round-trip ggml op for the graph-side write)
removes ~all of it — predicted residual: q-side QAT (tiny) + rescore (~4%
of score). LID_EXACT stays OPT-IN until P3a lands.
STATE: buildspec P0 done, P1 descoped, P2 landed+gated (class A, opt-in),
P3a next (new op GGML_OP_DSV4_FP4_RT + graph insert at lid cache write),
then P3b (mxfp4 container), P4 (default flips + gates.sh).

## Iteration 22c — P3a scout (QAT-at-write): single write site, one small op

WRITE SITE: exactly ONE (deepseek4.cpp ~1189): the lid state-compress output
is hadamard-rotated then cpy_k'd into kv_lid with state_write_idxs — both
prefill and decode flow through it. P3a insert is 3 lines between the
hadamard and the cpy_k. ONE reader (deepseek4.cpp:630 -> the fused/unfused
score path) — blast radius is the selection pipeline only.

NEW OP: GGML_OP_DSV4_FP4_RT — unary f32->f32 e2m1 block-32 QAT round-trip.
CUDA kernel already exists (dsv4_fp4_quant_kernel, reuse with identity
strides); CPU helpers already in ops.cpp. ~15 registration touchpoints.

NUMERICS (the important discovery): f16 storage of QAT values is EXACT —
e2m1 has a 2-bit mantissa and power-of-2 scales, so f16(QAT(f32 x)) ==
QAT(f32 x) bit-for-bit in range (subnormal edge only for amax<7e-38 rows,
which are all-zero rows that never get selected). That makes at-write
STRICTLY MORE official than the current A1 order (QAT(f16(x)) — quantizing
from already-f16-rounded cache values). ds4.c stores QAT(f32 x) in fp32; we
get the same values in half the bytes.

ENV: LLAMA_DSV4_LID_QAT_WRITE=1, read by three places: the graph (insert the
write op), the CUDA fused op (skip k-side re-quant, keep the f16 fast path,
rescore gets an f16-strided-K variant), the CPU ref (skip k-side sim).
Cache content changes => flag is process-lifetime + fresh context (statics
make this automatic); session save/restore untouched (still F16 values).

SCOPE: ggml.h/.c +30 (op), ops.cpp +27, dsv4_lid_topk.cu +60/-10 (host op
skip logic + rescore f16-K template), ggml-cuda.cu +6, deepseek4.cpp +12,
tests +30. Verification: FP4_RT backend-ops cases; LID_TOPK zero-tolerance
under EXACT+QAT_WRITE; e2e determinism; pp2048@d65536 re-leg (predict
~300 t/s, i.e. exact-mode cost collapses from -28.3% to ~1-2%).
Risk: public API no · cache-content migration (flag=process-lifetime) ·
cross-module no · reversible yes · external blocker no.

## Iteration 23 — P3a landed after a two-round rescore-kernel fight

P3a build: GGML_OP_DSV4_FP4_RT (op count 103->104; CUDA reuses the A1 quant
kernel, CPU reuses the row helper), graph insert at the single lid write site
(LLAMA_DSV4_LID_QAT_WRITE), fused-op k-side re-quant skip (f16 fast paths
kept), CPU-ref mirror. FP4_RT 3/3 exact; LID_TOPK 36/36 zero-tolerance
(EXACT+QAT_WRITE x {plain, int8, dec}).

THE SURPRISE: qat_write alone did NOT fix the -28% (217->213 e2e; op-level
359->250ms). nsys: dsv4_lid_rescore_kernel cost 2.8x the ENTIRE score kernel.
Two real bugs in my kernel, found by measurement not guessing:
1. K rows re-read from global 64x (once per head), uncoalesced 2B loads
   -> smem wave staging (each row read once, coalesced): 250->192ms.
2. Serial FP dependency chains with zero ILP (128-deep dot x 64 heads)
   -> 8 heads interleaved as independent chains (bitwise order preserved:
   each head's chain stays ascending-d, heads combined ascending): 192->124ms.
   (n_head=4 tail case caught by zero-tolerance test: predication guard
   regressed ILP 124->137 -> split hot/tail paths -> 127ms.)
3. Decode was still 1 block on 1 SM (+822us/layer = tg-gate killer at x21):
   split into two phases — parallel score chunks (grid nt x ceil(cand/128))
   + tiny sort-only finale -> decode 3039 -> 974us (+200us/layer vs baseline
   774, ~+4% decode at 512k — tg gate holds ~10.6).

Final op-level (n_lid=33280): prefill 103.8 -> 126.7ms (+22%, ~+2.8% e2e);
decode 774 -> 974us. Final e2e legs running.

## Iteration 23b — P3a CLOSED: all gates pass; -7.1% e2e price; NONDET was environmental

Final e2e: pp2048@d65536 exact+qat_write = 281.9 vs 303.3 default (-7.1%,
from -28.3%; matches op math: 22ms x21 layers per 6.75s ubatch. NOTE the
earlier "+2.8% e2e" prediction forgot the x21 — op deltas are PER LAYER).
Determinism: initial back-to-back pair split at token 3 — attribution probe
in settled memory: EXACT pair MATCH, DENSE pair MATCH -> the split was the
documented GB10 cross-process machine-state variance (3rd observation; the
NONDET run fired right after a bench exit). Op-level determinism stands on
the zero-tolerance suite (the stronger gate).
P3a DONE: official-exact selection at -7.1% pp / +4%-ish decode at 512k.
Residual trims for P3b era: packed mxfp4 reads (bandwidth), possibly reusing
pass-1 candidate scores in phase B. Next: P4 default flips + gates.sh, P3b.

## Iteration 24 — P4a: four defaults flipped ON + gates.sh battery

FLIPPED (one commit, FLAGS.md same commit): FUSED_LID, CSA_GATHER, LID_INT8,
LID_DEC — all now `=0` disables. The fast serving profile IS the default;
no env needed. test-backend-ops tolerance selection mirrors the new
defaults (unset => int8-class 1.2e-2 gate on d_idx==128 topk cases).

gates.sh (experiments/ds4-tile): one-command battery. quick = backend-ops
sweep (6 DSV4 ops + EXACT zero-tolerance leg) + coherence + determinism;
std adds PPL c32768 trio (defaults/exact/conservative, 2-sigma gate) +
passkey 5-depth battery; full adds soft-gated depth legs (pp@d65536 >= 285,
tg@d131072 >= 13.5). Corpus/prompts derive from immutable base-commit blobs
(git show e3546c794:...) so the battery does not drift with the tree.
Determinism stage: one retry on split, settled-pair rule per the
machine-state variance findings.

GATES (quick, post-flip): 7/7 ops PASS (incl EXACT zero-tolerance over the
now-default int8 pass-1), coherence PASS (ggml.c continuation, clean),
determinism PASS (run1 == run2, no retry needed). std/full battery deferred
to P4b where it gates the EXACT-default decision on post-P3b numbers.

## Iteration 25 — P3b-i LANDED: packed MXFP4 lid container

NEW OP GGML_OP_DSV4_QAT_SET_ROWS (COUNT 104->105): set_rows into an MXFP4
container with OUR QAT rounding folded into the scatter (stock set_rows has
no MXFP4 dst on CUDA and stock rounding on CPU). Packing: e = s+127 with
TRUE-level nibble table makes dequant d = 2^(e-127) == the QAT scale, so
dequant(pack(x)) == dsv4_fp4_rt(x) bit-exact.

Container: LLAMA_DSV4_LID_CACHE_MXFP4=1 flips ONLY the lid ctor type
(llama-kv-cache-dsv4.cpp — type_k is shared by all 4 sub-caches, lid
decoupled); write path cpy_k_qat (kv-cache + both context wrappers);
DSV4_K_CACHE_STATE_VER 1->2. Read side: staged dequant into the existing
k_force_f32 pool buffer — ALL score kernels keep their float dispatch,
zero kernel changes. Unfused decode shortcut blocked under packed cache.

ZERO-TOLERANCE CATCH: rescore dispatch's `qat_write && !f16` arm read
k->data as raw f32 — under packed cache that reinterprets 17B blocks as
floats. Op suite flagged 11%/19% selection mismatch before any model run;
fix = staged buffer wins whenever it exists (k_force_f32 guard).

GATES (all PASS): QAT_SET_ROWS 3/3 zero-tolerance; lid_topk packed cases
x4 default + EXACT profile zero-tolerance; gates.sh quick under packed
profile (7/7 ops, coherence, determinism); engagement verified in logs
(type = mxfp4, lid KV buffer 2.79 MiB @c8192 vs ~10.5 f16 = the predicted
3.76x); prompt-cache save/restore round-trip "exact match" + identical
continuation. PPL spot folds into the P4b std battery.

NEXT (P3b-ii): native packed readers — decode kernel 68B/row (claws back
the exact-mode +200us/layer decode price), then exact-price re-leg for the
P4b default decision.

## Iteration 26 — P3b-ii: native packed readers + exact-price re-legs (+ box death #7)

PACKED READERS: dsv4_score_decode_kernel + dsv4_lid_rescore_score_kernel
gained PACKED template arms reading the 68B rows directly (byte strides;
decode skips whole-cache staging entirely — at decode the staging would
COST more traffic than f16). Prefill staging switched f32 -> F16
(f16-of-QAT bit-exact) recovering +3.9% on the exact leg. 18/18 op cases,
zero FAILs under default AND exact env.

BOX DEATH #7 (12:42): the first legs run used ONE llama-bench with
-p 2048 -n 64 -d 65536,131072 (4 tests). Journal shows avail 21->16.7G
during fill then stops cold — no OOM-kill line (UMA spike outran earlyoom
and the kernel logger; hard reset). Mechanism: context teardown/recreate
between tests transiently overlaps two deep contexts. RULE (memory
#7): one llama-bench test per process; single -p XOR -n, single -d.

RE-LEGS (same boot, single-test processes, settle-waits):
| profile        | pp2048@d65536      | tg64@d131072 |
| default (f16)  | 315.1              | 14.31 (ref 14.29 ✓) |
| packed         | 303.7-308.4 (-2~3.6%) | 14.33 (neutral) |
| packed+EXACT   | 285.2 (-9.5%)      | 13.63 (-4.8%) |
Packed+EXACT absolute BEATS P3a's f16+EXACT absolute (281.9, prior boot)
while shrinking the lid cache 3.76x (2.79 MiB @c8192 confirmed in logs;
~520MB saved at 512k). Packed decode reader is latency-neutral at d131k
(decode dominated by FA/MoE there; win should grow at 512k n_lid=131072).

P4b DECISION (per pre-agreed criteria: flip only if <=3% pp and
tg-neutral): -9.5%/-4.8% is material -> LID_EXACT + QAT_WRITE stay OPT-IN
as the official-exact validation/reference profile. LID_CACHE_MXFP4 also
stays opt-in (prefill -2~3.6% for 3.76x memory; add it to the 512k
serving profile where memory is the binding constraint). Defaults remain
the P4a fast profile. gates.sh quick PASS on the final build (packed
exact profile: 7/7 ops zero-tolerance, coherence, determinism).

## Iteration 27, step 1 — FA-with-LSE: op flag + tail layout + CPU reference

Toward BUILDSPEC-fattn-lse-merge (split-attention: dense raw-window FA +
remainder-only tiled FA + LSE merge; kills the repeat_4d raw replication
at deepseek4.cpp:850; ceiling +65% pp @d131k).

LANDED (this commit): ggml_flash_attn_ext_with_lse — result gains one
tail ne3-slice holding lse[h, iq, s] = M + log(S) at element offset
DV*n_head*n_q*ne3, idx (s*n_q + iq)*n_head + h; flag = op_params i32[4];
requires DV >= ne3 (T<=128 tiles at ub2048 vs DV 512). KEY DESIGN FACT:
CUDA kernels compute ALL dst offsets from Q/K dims (launch_fattn never
passes KQV->ne), so the tail is invisible to existing kernels — zero
hot-path dst-stride changes. CPU: one-chunk path writes the tail;
split-kv and tiled paths force-disabled under the flag (loud, not
silent). Gate: full FLASH_ATTN_EXT backend-ops sweep with flag unset —
green (no behavior change).

REMAINING: CUDA meta plumbing (mma epilogue emission + stream-K fixup
writeback + finalize kernel + supports_op mma-only), dsv4_fa_merge op,
deepseek4 graph split, merged-halves==dense reference tests, depth legs
(single-test per process — crash rule #7).

## Iteration 27 (CLOSED) — split-attention + LSE merge LANDED, default ON

Steps 2-4 of BUILDSPEC-fattn-lse-merge, on top of step 1 above.

CUDA FA-with-LSE (3e4c79585): DESIGN SIMPLIFICATION vs buildspec — no
finalize kernel, no forced meta alloc. Every output row is finalized in
exactly one place already holding the combined (max, rowsum): fully-owned
tiles write lse = max + log(rowsum) straight from the mma epilogue
(np==1 both cols_per_warp variants + np>1 combine block, gated
!needs_fixup && !is_fixup); seam tiles write it from the stream-k fixup
kernels (uniform + general) next to their `*dst = dst_val/rowsum`.
fattn_kernel_t grew a trailing write_lse i32 (vec/tile/wmma UNUSED).
supports_op: LSE requires the mma kernel, else CPU fallback. SINKS FOLD
INTO THE LSE for free — the kernel merges sinks into (max, rowsum)
before the meta/tail writes on every path (epilogue, both fixups, CPU
after its sink block), so the initial sinks-rejection gate was lifted.
LSE is invariant to FATTN_KQ_MAX_OFFSET (stored pair is consistent).
Gates: 11 LSE cases (incl kv=8192 stream-k seams, MLA 576/512 V-view,
ne3=2 multi-seq, 3 sink cases) + full FA sweep 2908/2908 flag-unset.

GGML_OP_DSV4_FA_MERGE (c16f6e91d + shape relax): out = (ea*a + eb*b)/
(ea+eb), ea = exp(lse_a - max(lse_a, lse_b)); both-masked rows -> zeros.
Halves may slice rows differently ([DV,H,nt,1+1] dense vs [DV,H,W,T+1]
tiled share one memory layout) — constructor requires equal DV and equal
row count only. BUG CAUGHT BY GATES: first battery aborted in
graph_reserve because the constructor still had ggml_are_same_shape (the
op-level tests only used same-shape halves). Fixed + 2 mixed-shape
backend-ops cases (5/5).

deepseek4 graph split (LLAMA_DSV4_FA_SPLIT, default ON): tile branch now
runs (i) dense raw-window FA over all queries (ne3=1, full occupancy,
sink attached here — its LSE carries the sink exactly once) + (ii)
remainder-only tiled FA over per-tile unions (k_all shrinks to
[hd,1,u_cap,T]) + (iii) dsv4_fa_merge. repeat_4d + concat GONE. Guard
T <= hd (LSE tail must fit) falls back to concat path.

GATES (fa-split battery, run dir gates-runs/fa-split-20260714-161213):
smoke 12k ok; determinism @42k on1==on2 PASS; A/B split vs concat
BYTE-IDENTICAL (64 greedy tokens @42k); passkey 5/5 @~42k; PPL c32768
tile-forced ON 1.1884 +/- 0.00595 vs OFF 1.1873 +/- 0.00593 (well within
2 sigma); tg64@d131072 14.27 (ref 14.29 — decode untouched, neutral).

PERF (single-test benches, same boot):
| leg            | split ON | split OFF | delta |
| pp2048@d65536  | 321.14   | 299.91    | +7.1% |
| pp2048@d131072 | 270.24   | 254.74    | +6.1% |
Both are new absolute records for these legs (prior: 315.1 @d65k,
245 @d131k in the B2 table). CEILING REVISIT: +65% (ncu union-overlap)
did NOT materialize — the replication traffic is gone, but the
remainder-only tiled call (u_cap=4096 KV x W=16 queries per tile) still
sits on the small-nb latency wall and is now the binding FA constraint.
Next lever if pursued: batch the remainder tiles wider (fold T into
ncols) or KV_max-style skip inside the union call; the dense half is
already at the fast 41-TFLOPS shape. FA-with-LSE + merge remain
upstream-worthy standalone (sequence-parallel / tree attention).

## Iteration 28 — remainder-batching scout: NEGATIVE, W=16/U4096 stays

Scouted the post-split remainder-FA lever (iter 27 follow-up). Code scout
(agent) + zero-code W/UCAP sweep (gates-runs/w-sweep-20260714-173621).

Code scout findings (fattn dispatch + mma):
- Remainder instance is (DKQ 576, DV 512, ncols1 4, ncols2 16); ncols2=16
  is pinned by gqa 64 -> ncols1 CANNOT widen for any W (W only raises
  iter_j). nbatch_fa 32, iter_k 128 @u_cap 4096. Not block-starved:
  ntiles_dst 2048 == dense.
- KV skip is SUFFIX-ONLY (flash_attn_mask_to_KV_max backward scan -> upper
  bound; no interior/lower bound, no in-kernel all-inf chunk early-out).
- G-fold (concat G unions per stream + block-diag mask): NO DRAM win —
  same disjoint bytes remerged; needs a new KV-min interval scan just to
  break even. REJECTED. KV-min alone (~30-LOC mirror kernel + ~8 plumb
  sites + 5 LOC kbc; hazard: decouple needs_fixup from mask-raised
  kb0_start) only pays INSIDE a G-fold -> shelved with it.

Sweep (pp2048@d65536, split ON, same boot, single-test + settle):
| w16-u4096 (default) | 334.6 | baseline |
| w32-u4096           | 326.4 | -2.5% |
| w64-u4096           | 321.4 | -3.9% |
| w32-u6144           | 316.0 | -5.6% |
| w16-u2048 (control) | 339.4 | +1.4% (accuracy-invalid: ~60% of tiles
                                truncate; control only) |

WHY W-widening loses (scout model corrected): union padding is fully
masked, so KV_max clips each stream to its ACTUAL union size (~2400 mean
@W16). Effective FA compute is therefore ∝ union size, and unions GROW
with W (each token computes against more cells it doesn't need) — that
swamps the halved window-count DRAM term, which the u2048 control bounds
at ~1.4% total. Monotonic: every widening leg lost; u6144 (less
truncation = more real compute) lost most.

CLOSES the remainder-batching lever: W=16/U4096 is the best measured
config and stays default; kernel
ncols pinned; G-fold rejected; KV-min shelved. Remaining post-split FA
headroom at d65536 is ~1-4% (u2048 control bound) — not worth kernel
work. Depth-scaling attribution (lid indexer share grows with n_csa)
is the better hunting ground for the next pp lever.

## Iteration 29 — post-split nsys/ncu attribution: lid indexer is the new #1

Fresh profiles on the split-attention build (83767ecd1 era), replacing the
2026-07-12 pre-split attribution. Run dir experiments/profiles/
postsplit-20260714-182048 (local; 221MB, gitignored). nsys legs pp2048 at
d65536 (335.2 t/s w/ ~3% overhead) and d131072 (270.2); kernsum windowed
to the measured pass. ncu (model-free, backend-ops perf cases at the
exact production shapes; NOTE: ncu on this stack only profiles the first
cold launches — --launch-skip lands in the PDL region and intercepts
nothing, GGML_CUDA_PDL=0 does not help; use --launch-count without skip).

Kernel shares (top groups):
| group                     | d65536 | d131072 |
| lid (score_int8+topk+mrg) | 17.3%  | 30.1%   |
| FA (both halves, one sym) | 21.4%  | 18.3%   |
| MoE mmq (+quantize)       | 35.7%  | 31.2%   |
| concat raw+comp K         |  3.4%  |  2.8%   |
| gather/cpy/merge/hc/norm  | ~11%   | ~10%    |
FA absolute is depth-FLAT post-split (1298ms -> 1380ms window share) —
the split killed FA depth-scaling; pre-split FA was 44.8% @d131k, now
18.3%. dsv4_fa_merge costs 0.9-1.2%. The lid indexer DOUBLES d65k->d131k
(score 36.8ms -> 73.3ms/launch, linear in n_lid) and is now the dominant
depth-scaling term. Empirical FA instance for BOTH halves is
flash_attn_ext_f16<512,512,8,8> (hd=512 — k_rot hadamard keeps rope out;
the iter-28 scout's 576/(4,16) dispatch derivation was wrong).

ncu SoL, production shapes (grid 48 blocks, 8 warps/SM, occ 16.7%):
| shape                          | compute | memory | duration(locked) |
| dense raw 2304kv x 2048q       | 37.6%   | 49.4%  | 15.6ms |
| remainder 4096kv x 16q x T128  | 21.6%   | 27.9%  | 48.2ms |
Remainder is latency-bound (both pipes <30%), ~1.7x worse per unit work
than dense; it is ~75% of FA time. Fixing it to dense efficiency would
save only ~5-6% wall @d131k (FA is already small). Historical 34/47
dense SoL reproduced (49/38 here) — methodology consistent.

CONCLUSION: next pp-at-depth campaign target is the lid indexer (30% and
scaling; score kernel is L1-bound per iter 14), not FA. Second-order:
MoE mmq is the flat-share ceiling as before. Box caveat: power/thermal
capping active during runs (clocksnap now records this; see
83767ecd1) — shares are relative and robust, absolutes carry ±4%.

## Iteration 30 — lid traffic campaign scout (4 levers ranked, 1 struck)

Shallow legs (gates-runs artifacts local): pp2048 d8192 466.8 / d32768
361.3 (nsys ~3% overhead). Attribution table + all four design-scout
verdicts consolidated in BUILDSPEC-lid-traffic.md. Highlights:
- shallow ctx is MoE-bound (~49% @d8192); lid only matters d65k+.
- FA share PEAKS at d32768 (29.4%) — the dense-CSA gap below TILE_MIN;
  TILE_MIN=4096 probe under split: +1.5% @d32768 (weak positive,
  parked).
- 4-bit-K-in-smem lever STRUCK before rebuild: scout traced it to the
  iter-15 LLAMA_DSV4_LID_INT4 measured negative (1.6x slower; dp4a
  demands int8 operands, unpack tax on the same int pipe).
- SORT_N 8192 blocker found in advance: 64KB static smem exceeds the
  48KB compile cap — needs the 3 topk kernels on dynamic extern shared
  (fits the 99KB carveout).
- Fusion scout: A-safe bit-identity achievable (verbatim bitonic +
  quant + ascending-h accum); footprint is a WASH (cand-val tree
  replaces scores); win is ~0.5GB/layer traffic; K L2-residency under
  1-token blocks is the deciding risk (ncu gate).
- fp4-mma scout: register-resident-K schedule is the ONE structure the
  iter-14c/15 negatives never tested; MLA 1-K-per-64-heads makes
  B-resident real (16x smem-read cut); numerics = LID_FP4's accepted
  class; needs LID_CACHE_MXFP4; explicit ncu kill criterion.
Build order: (SORT_N + f16 scores + int8-K pre-quant) -> fused
chunk-topk prototype -> fp4-mma probe. Ceiling ~10-15% pp @d131k.

### Iteration 30 addendum (2026-07-15) — detail scout, all steps

Five parallel implementation scouts (code-only, no GPU) produced
BUILDPLAN-lid-traffic.md — line-precise hunks for step 1 and settled
designs for steps 2-3. Corrections found vs BUILDSPEC: pre-quant
buffer is 16.5 MiB/stream not 4.1MB (int8 = 128 B x n_lid); the
iter-29 "K 8.4MB" was the MXFP4-packed size, int8 K is 16.8 MB @131k
(fused L2 margin tighter than assumed); merge re-reads the score
matrix at EVERY tree level (:888) — a third traffic term, killed only
by a candidate-value tree; fp4-mma regs ~50 @Nc=16 not ~40. Verified
clean: pre-quant bit-identity (K-row-local amax, no block coupling);
scores-buffer consumer trace (topk only); container nibbles ARE
hardware e2m1 + raw ue8m0 = transform-free mma operands; supports_op
auto-tracks the SORT_N macro. New blocker logged: fp4-mma staging
bypass breaks EXACT pass-2's k_f16_d source — must keep staging alive
under EXACT or use the inline-dequant arm. Step-1 gate upgraded: the
EXACT oracle displacement re-run is load-bearing (f16 pass-1
displacement vs m=64 window, prior p100=36).

## Iteration 31 — lid traffic step 1: f16 scores + int8-K pre-quant land; SORT_N 8192 measured negative

Built the step-1 trio from BUILDPLAN-lid-traffic. Op-level A/B
(test-backend-ops perf, same boot) split the verdict:
- SORT_N 8192: +13% op @n_lid 8704, +7% @17000, +12% @33280-decode.
  The launch-count argument missed that bitonic work grows N*log^2 N:
  fewer-but-bigger chunks add ~30% total sort work, more than the
  halved merge launches save. REVERTED to 4096; kept the dynamic-smem
  + score_t template conversion (32KB dyn @4096, no cost measured,
  and step 2 builds on it).
- f16 score store (d128) + global int8-K pre-quant @4096: -0.6..-1.3%
  op across the four deep shapes. KEPT (strictly-less-work, A-safe).
Serving legs (same-boot A/B, r=3, clocksnap; SW Power Cap seen on
this boot): pp2048@d65536 326.2 vs base 326.3 (flat); @d131072 270.5
+/-1.4 vs 268.0 +/-0.5 (+0.9%, marginal).

Why the projected ~5% @d131k was wrong, in order of importance:
1. ATTRIBUTION BUG (mine): n_lid is the COMPRESSED lid length
   (ratio 4) — @d131k n_lid~33k, not 131k. The 33280 perf case IS the
   d131k shape; BUILDSPEC's original 268MB/8.4MB figures were right
   and my BUILDPLAN "correction" to 16.8MB is retracted there.
2. The merge's row[idx] gather is RANDOM-access: f16 doesn't cut
   gather sectors (2B vs 4B in the same 32B sector). The gather dies
   only via a candidate-VALUE tree (store (idx, half val) at chunk
   emit, merge over stored vals) — that is step 2's merge change and
   it works UNFUSED: promoted to sub-step 2a, do-first.
3. The lid op at these shapes is score-kernel-dominated; topk was
   9.7% of GPU-busy, and only its sequential-read fraction responds
   to f16.
EXACT oracle re-run with f16 pass-1 (--f16-scores added to
fp4_oracle): displacement p100 51 vs 50 baseline @33k — m=64 holds
with 13 ranks headroom (note: baseline seed-2 was already 50, not the
previously quoted 36). Op gates: default / EXACT+QAT_WRITE / INT8=0 /
DEC=0 / FP4 all green, incl 4 new eval cases (8193, 17000, 70000
tree-merge, d128 17000 f16-multi-chunk) — found and closed a gap
where no eval case exercised the chunk/merge kernels above the old
boundaries. Step-2 delta scout re-anchored the fused design to the
post-step-1 tree (cand-val stores HALF via dsv4_lds for byte-identity
with the f16-ranked unfused path; own launch wrapper inside the d128
branch; scores_h guarded on !fused) — folded into BUILDPLAN.

## Iteration 32 — 2a/2b detail scout + production-shape decomposition

Measurements first (test-backend-ops perf + nsys/ncu, committed build):
- d131k serving shape (n_lid 33280, nt 2048) @top_k=512: op 102ms =
  score 71.2 (69.9%) + chunk 24.1 (23.6%) + merge 6.6 (6.5%).
- chunk kernel ncu: SM 77.8% / L1 78.0% / L2 0.5% — smem bitonic
  COMPUTE bound; its 136MB sequential matrix read is ~0.7ms of 24ms.
- @top_k=2048 (PRODUCTION indexer_top_k; new perf case added): op
  121.7ms = score 71.1 (58.5%) + merge 26.4 (21.7%, 4 launches at
  merge_group=2) + chunk 24.0 (19.8%). TOPK TOTAL 50.4ms = 41.5% of
  the op (~10-12% wall @d131k) — the deep merge tree at production
  top_k doubles the topk cost vs the 512-case.
Scout verdicts:
- 2b FUSION: KILLED by arithmetic. Unique saving <1.7% of op (matrix
  write <1ms + chunk read 0.7ms; merge gathers already die with the
  value tree); cost = losing the score kernel's 16-token smem-K
  amortization (K L2 traffic 545MB -> 8.7GB, 16x) on the 58-70% score
  side + 1.75x occupancy loss. No 512k niche (scales together, stays
  ~1.7%). Alternative running-topk-in-score-kernel fusion is
  geometrically blocked: 8:1 selectivity means a 128-comp tile can
  contribute up to all 128 scores — no safe pre-selection below the
  full matrix. Same disposition as SORT_N 8192.
- 2a(A) partial bitonic top-k (dsv4_bitonic_topk: sort K-blocks, fold
  + resort halving rounds; exact-set by strict-total-order argument):
  CE-count ratio 0.70 @K=512 (~7% op) but 0.93 @K=2048 (~1.6-3% op)
  — phase-1 block sort dominates at K=N/2. +58/-3, bit-identical,
  reusable by all 3 topk kernels.
- 2a(B) candidate-value tree: merge gather is ~0.7ms of 6.6/26.4ms —
  ~1% op. Demoted; decide after (A).
OPEN QUESTION the scouts surfaced: at production K=2048 the right
lever is a selection algorithm whose work does not blow up at K=N/2
— candidates: radix-select on the half-score bits (bit-exactness
needs tie handling: strictly-greater + idx-asc fill at threshold), or
SORT_N=8192 REVISITED with partial-K (the earlier 8192 negative was
FULL-sort N*log^2 N; partial pays N*log^2 K + folds, and bigger N
cuts chunks 9->5 and merge launches 4->2 at K=2048). Both unscouted.
Prize: topk 50.4ms is 41.5% of the lid op at serving shape.

## Iteration 33 — 2a(A) partial bitonic top-k LANDED

dsv4_bitonic_topk<SORT_N>(vals, idxs, K): phase 1 sorts K-blocks
descending (k==K stage direction-forced), phase 2 fold+resort halving
rounds; strict-total-order argument => top-K set identical to full
sort, output order identical (sorted desc). Drop-in at all 3 topk
call sites, K = next_pow2(top_k), full-sort fallback at K>=SORT_N.
Gates: all 5 op profiles green incl EXACT 0.0 and scalar 0.0.
Op perf (vs step-1 build): -7.5..-8.9% @top_k=512 shapes, -1.6%
@production top_k=2048 (as predicted: phase-1 block sort dominates at
K=N/2). Cumulative op vs pre-campaign baseline @33280x2048x512:
104.0 -> 95.0ms (-8.7%).

## Iteration 34 — selection-design round: RADIX-SELECT wins; step 3 build-ready

Two design scouts against fe6043410/d89d2849b:
- RADIX-SELECT on f16 score bits (STEP 2c in BUILDPLAN): one kernel
  per token replaces chunk+merge on the d128/half path. Tie story
  SOLVED: canonicalize -0 (0x8000->0) so key-eq == f32-value-eq,
  then strictly-greater + lowest-index ordered-scan fill at ==T is
  dsv4_better exactly (atomic-append is wrong — nondeterministic).
  Expected topk 50.4 -> ~5ms, op -37.5%, ~9-11% serving wall.
  SORT_N-8192-with-partial-K REJECTED by CE math (+4.6% total at
  production K=2048; occupancy 2->1 blocks/SM); 16384 dead (128KB >
  99KB carveout).
- STEP 3 fp4-mma detail scout: build-ready, and the hardest part
  evaporated — container rows are literal block_mxfp4 arrays and
  Blackwell mmq consumes qs verbatim (mmq.cuh:934 memcpy), so no
  nibble repack anywhere. EXACT via packed-direct inline-dequant
  rescore; class-B test gate needs an env-gated max_err branch
  (~8e-2 smoke) with the EXACT 0.0 run as the true proof; kill gate
  vs int8 71.1ms @33280x2048. After radix lands, score = 93% of the
  lid op.
Standing bench matrix (18 legs, cooldowns, base+mtp) running on
fe6043410 — results in the matrix-p5b profile dir; first legs:
pp2048@d0 518.7 (NEW RECORD, prior ~505), tg128@d0 17.9.

## Iteration 35 — STEP 2c radix top-k LANDED (gates green pre-crash)

dsv4_topk_radix_kernel: 2x8-bit MSB histogram threshold + ordered
ballot/popc compaction (lowest-index fill at ==T, -0 canonicalized)
+ runtime bitonic on the top_k selected; routed via
dsv4_topk_try_radix (half scores, nt>=16, n_lid>top_k), env
LLAMA_DSV4_LID_RADIX default ON. Gates (all pre-crash, first
compile): 6 op profiles green incl EXACT 0.0 (radix drives pass-1
over tie-heavy e2m1 scores), FP4, and RADIX=0 revert. Op perf:
33280x2048@2048 119.8 -> 75.0ms (-37.4%, scout predicted -37.5%);
@512 95.0 -> 73.7 (-22%); decode unchanged (bitonic by design).
Cumulative op vs campaign start: 121.7 -> 75.0 (-38.4%); topk chain
50.4 -> ~3.9ms. Serving same-boot: pp@d65536 342.2 vs 334.3 (+2.4%).
BOX HARD-HUNG during the pp@d131072 radix leg (22:14, journal stops,
no OOM/Xid logged; avail steady 15.7G) — new crash signature, cause
unattributed (radix weak suspect: d65k leg + 98 op-perf runs at the
exact d131k shape were clean; sustained-load thermal/power lockup at
least as likely). d131k A/B pending on the new boot.

## Iteration 36 — standing matrix + MTP records + dtype audit + crash recovery

Matrix (bench legs boot A on fe6043410; completion/MTP boot B on
radix 47e0599dc; boot effect ~+6% on boot B — all deltas quoted
same-boot):
- llama-bench pp2048: d0 518.7 / d16k 426.3 / d32k 356.7 / d65k
  334.3 / d131k 269.3. tg128: 17.88 / 16.61 / 16.36 / 15.88 / 14.88.
- Radix serving A/B: pp@d65536 334.3 -> 342.2 (+2.4%, boot A pair);
  pp@d131072 287.0 -> 302.7 (+5.5%, boot B pair). Records: 518.7@d0,
  342.2@d65k, 302.7@d131k.
- Completion tg (base): short 17.15 / 32k 16.13 / 65k 15.64 / 131k
  14.97 — only -13% short->131k.
- MTP (llama-server, draft-mtp mxfp4, defaults): 24.3 t/s @1.2k
  (+42% vs base), 27.6 @8k (+61%) — NEW RECORDS (prior 21.6).
  Coherence: 4 base outputs clean greedy continuations (one
  single-token case typo @65k); MTP-vs-base greedy diverges ~token
  20 then both coherent — expected class (verify runs nt=4 batches
  vs base nt=1; fp reassociation flips near-ties; same class as the
  CSA_GATHER precedent).
- TOOLING TRAP (cost one OOM'd session + two multi-GB files):
  llama-cli in batch mode (-f, -no-cnv, --no-display-prompt) runs
  away writing ~70MB/s to stdout (56GB + 22GB .out files, disk hit
  98-99%). BANNED from scripts. llama-completion lacks spec args
  (-md gated to CLI/SERVER/SPECULATIVE examples) — MTP measurements
  go through llama-server /completion (timings in response JSON).
- Box crash #8 addendum: the wedge-correlated d131k radix leg ran
  clean on reboot (same-boot A/B above) — supports sustained-load
  thermal lockup over a radix bug; pattern recorded in memory.
- Dtype audit (3 scouts: HF/paper, ds4.c, ours) -> DTYPES.md +
  DTYPES-ds4c.md. Headlines: paper's own serving quantizes index
  scores to BF16 (our f16 is finer); official attn softmax is BF16
  (ours f32); hadamard is canonical-but-quantizer-only (invisible in
  the BF16 HF reference); our one real lossier-than-official spot is
  experts at ~2.5-3bpw (size-forced; ds4.c is lossier still); the
  UD-IQ3_XXS mix is IQ2_S/IQ3_XXS (NOT IQ2_XXS — the custom moe
  tile kernel doesn't even engage). fp4-mma (step 3) would move the
  indexer multiply onto the official grid.

## Iteration 37 — step 3 fp4-mma fully de-risked (scout + oracle, no GPU)

Implementation scout vs 14f47d82c + a CPU oracle extension closed
every open question; STEP 3 in BUILDPLAN updated to ready-to-type:
- Q-pack: PER-HEAD (all-heads prologue killed — its 4.5KB smem
  claim dropped the M=16 token dim; real 64KB -> 1 block/SM).
  smem ~3.2KB total vs int8's 18.9KB.
- K gather: direct loads from the 68B rows FAULT (qs at byte
  offset 1 -> always misaligned); resolved mmq-style: byte-copy to
  aligned smem + load_generic into resident B-regs, once per
  block, amortized over 64 heads.
- EXACT pass-2 routing fix pinned (:1680 predicate) — without it
  pass-2 silently reads block bytes as float.
- Oracle --fp4-mma-displacement (new mode, committed): fp4-mma
  pass-1 m-need p50=0 p100<=4 all shapes/seeds vs int8 p100 up to
  51 under identical f16-store numerics -> m=64 EXACT window has
  ~16x headroom, no RESCORE_M bump. fp4-mma is not just faster
  than int8 pass-1 — it ranks on the truth grid.
- Container tax re-derived with fp4-mma ON: staging + pre-quant
  terms disappear, lid-K read drops 3.76x -> LID_CACHE_MXFP4
  flips net-neutral-to-positive; default flip gated on the kill
  gate + EXACT combo 0.0 + both-arms serving A/B + 512k tg + PPL.
- Cost model: ideal ~5ms, realistic 15-40ms vs int8 71.1ms @prod
  shape; kill = l1tex>=sm AND >=71.1ms, or reg spill >64.
Next session: type the kernel (build order in BUILDPLAN; ~+185/-4
one file + test tolerance branch + MXFP4 perf case). Server was
resident all through — zero GPU touched.

## Iteration 38 — STEP 3 fp4-mma LANDED, MEASURED WIN (op -47.8%)

dsv4_score_fp4mma_kernel: register-resident-K block-scaled fp4
tensor-core indexer scoring (mma.sync kind::mxf4 m16n8k64), env
LLAMA_DSV4_LID_FP4_MMA default OFF, Blackwell+packed-container gated.
Reads 68B block_mxfp4 rows direct (staging + int8 pre-quant bypassed);
K resident in B-registers across the 64-head loop; q packed to e2m1
per head into the block_mxfp4 layout; f16 score store -> radix.

MEASURED @33280x2048x2048 (d131k serving shape), full op:
- int8 F16-K reference: 73.9ms
- int8 MXFP4-K (staged): 74.3ms
- fp4-mma MXFP4-K: 38.8ms (stable 38.6-38.9 x3) = -47.8%
Score kernel ~71 -> ~35ms: the register-resident-K bet CONVERTED the
L1 bound the iter-14/15 negatives were stuck on (they only tested
smem-resident K). Kill gate was >=71.1ms -> passed by ~2x, no ncu
adjudication needed.

Correctness: 5 gate profiles 22/22 incl fp4-mma+EXACT+CACHE_MXFP4 at
ZERO tolerance (pass-1 lands true top-k in the m=64 window, oracle
p100<=4; pass-2 packed-direct rescore is bit-exact). Standalone
class-B smoke gate 1.1e-1 (observed 0.056 on the 2048x4 case).

Debug: one bug caught by compute-sanitizer — launched a flat 256-thread
block, but the ggml_cuda_mma ldmatrix/get_i/get_j primitives use
threadIdx.x AS the warp lane (0-31). Fixed to dim3(32,8):
threadIdx.x=lane, threadIdx.y=warp. Clean after that.

Container flip: with fp4-mma the LID_CACHE_MXFP4 staging+prequant
terms disappear; int8-on-container (74.3) ~= int8-on-f16 (73.9), so
the container is ~free at prefill and fp4-mma is pure win on top.
Cumulative lid op vs campaign start: 121.7 -> 38.8ms (-68%). Dtype:
fp4-mma puts the indexer multiply on the OFFICIAL e2m1 grid — faster
AND more canonical than the int8 default. Pending: on-model serving
A/B (both arms CACHE_MXFP4=1), PPL/coherence, then default-flip
decision for the 512k serving profile.

## Iteration 39 — GGML_CUDA_POWER duty-cycle governor (pattern-#8 counter)

Context: 5th box lockup 2026-07-15 20:37 (journal-stop, mem flat, no
OOM/kill/Xid — pattern #8) under interactive llama-server load. Root
class: sustained draw engages firmware SW Power Cap which never
clears. Studied competing fork Entrpi/ds4 (whose "2x prefill" is
llama.cpp's own MMQ stack vendored back — 11k LOC verbatim); the
stealable idea was upstream antirez/ds4 --power N (ds4.c:10583): EWMA
work time per unit, host-sleep work*(100-N)/N -> duty cycle N%. Their
Spark data: 85% FASTER than 100% sustained (18.99 vs 17.52 t/s)
because the firmware throttle never engages.

Our version (ggml-cuda.cu, env-gated, default off, all binaries):
- GGML_CUDA_POWER=N: cudaEvent pair around each graph_compute;
  elapsed consumed at the NEXT compute (already drained by caller's
  output sync — zero added syncs); sleep min(work*(100-N)/N, 5s)
  before submit; <0.05ms skipped.
- GGML_CUDA_POWER_ADAPT=1 (+_MIN, default 60): dlopen libnvidia-ml
  poller (500ms) on ClocksEventReasons; distress mask 0xE4 (SwPowerCap
  | Sw/HwThermal | HwPowerBrake); engage -> floor duty, restore after
  10s clear. Better than ds4's open-loop: pays the tax only in
  distress. NVML init verified clean on GB10.

Gates (2026-07-15): default-off parity pp2048@d0 520.58 +/- 1.60 (vs
523 standing, in-noise); duty accuracy POWER=85 -> 444.69 = 0.854x
(85 requested).

SURVIVAL TEST PASSED (2026-07-15 21:28-21:46, attended): two
consecutive pp2048@d131072 same boot with POWER=85 — leg 1 262.16,
leg 2 264.04, both exit 0. Leg 2 is the leg that wedged the box 3/3
times before (plus 2 more wedges in other forms); first-ever double
completion. No degradation leg-to-leg (leg 2 slightly faster). Cost:
264 vs 301-303 full-power single-shot = 0.87x ~= the duty tax.
Mechanism confirmed by boundary clocksnaps
(experiments/profiles/power-survival/): firmware cumulative slowdown
counters DID advance during the legs (SW thermal 2.1s->15.2s, HW
thermal 0.5s->4.1s across leg 2) — transient throttling happened but
never latched; the per-graph idle gaps give the power subsystem
recovery air. n=1 survival vs 3/3 prior repro — treat as strong but
keep POWER=85 mandatory on deep legs until more runs accumulate.

Telemetry finding + fix: "SW Power Cap: Active" reads at IDLE on
GB10 (normal power mgmt, 208MHz parked; clears under healthy load).
The v1 adapt mask (0xE4) would false-engage whenever the GPU idled.
Fixed: hard bits (0xE0 thermal/power-brake) unconditional; SwPowerCap
(0x4) counts only when nvmlDeviceGetUtilizationRates reports gpu>=50%
— capped-while-working is distress, capped-while-idle is sleep.
