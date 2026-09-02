# Deviations from upstream llama.cpp

Every difference between this branch and upstream, and why it exists. If a
deviation is not listed here it should not exist — delete it or add a row.

Base: upstream `a7cc83bba`. 46 commits, 74 files, `↑`8895 `↓`468.

## Standing rules

1. **A deviation must earn its keep.** If disabling it does not cost
   throughput, acceptance or correctness at a shape we actually serve, it
   should be removed rather than documented.
2. **Measure at production depth.** Our serving traffic runs 15–40k-token
   prefills. Several optimisations read *dead flat* at a 27-token prompt and
   then pay 5–8% at 15k. A shallow benchmark is not evidence for removal.
3. **Prefer upstream.** Where upstream ships an equivalent, take theirs even
   if ours works — the rebase tax compounds and their version gets reviewed.

## Upstreamable as-is

Standalone commits, no DS4-specific dependencies. Cherry-pick straight into a PR.

| Commit | Change |
| --- | --- |
| `08bfb1cad` | `ggml-backend`: one `synchronize` per split instead of one per input, when events are unused |
| `d6b0e7f90` | `llama-quant`: a same-type `--tensor-type` pin is a byte copy and never needs an imatrix |
| `786798dcb` | `llama-batch`: `reserve`+append instead of a zero-filling `resize` — hundreds of MB on wide-embd draft ubatches |
| `b50485abd` | `perplexity`: honor `parse_special` so chat-templated corpora tokenize to real control ids |

## Ours, no upstream equivalent

**`ae25da950` — fused DSV4 ops.** Seven ops with CPU references. The
lightning-indexer is sparse (score+topk fused) where upstream's is dense; the
hyper-connection fusion replaces an op composition that measured **−7.8%** with
one that is **+16–34%**. Carries the MXFP4 indexer container and the fp4
tensor-core score path, both depth-gated at runtime — sustained fp4-mma at
d131k shapes hard-locks GB10 even under duty cycling.

**`6d434e4dc` — per-row LSE for flash attention.** DSV4 splits an attention row
across launches and merges partials, which needs each partial's log-sum-exp.
Upstream's FA writes only the result. Opt-in tail slice, stream-k path only,
`supports_op` rejects everything else.

**`98b5474e9` — GB10 duty cycling + shape-keyed graph cache.** This box has
hard-locked 13 times; the SoC is a shared CPU/GPU budget and the firmware latch
is burst-sensitive, so averaging power is insufficient — `GRANULARITY=layer`
bounds the burst window. Serving runs at `GGML_CUDA_POWER=85`. The graph cache
keys on shape to stop capture blowup across a speculative decoder's varying
ubatch widths.

**`27952d3ab` — DSV4 model + composite KV cache.** Raw SWA plus three
compressed block caches sharing a compressor ring. Adopts upstream's `n_rs_seq`
rollback ring, sized from `draft.n_max`. Keeps an MXFP4-container v1-compat
*reject* — a v1 f16 snapshot would otherwise be reinterpreted byte-for-byte and
read as garbage.

**`a1e6f5be2` — disk prompt cache (DKV2).** Upstream's prompt cache is RAM-only
and dies with the process. Adds a disk tier with eager store (survives SIGKILL),
shutdown spill, exact eviction reserve, and truncated-file guards.

Two host-RAM bounds, both fixing defects in the above rather than in upstream:
the eager store gated only on the *disk* budget while allocating a full state
blob on the **host**, and the pending-spill queue capped entry *count* (4) for
entries whose size is unbounded and grows with the conversation (~350 MiB at 36k
tokens, ~2.3 GiB at 384k — so four of them is ~9 GiB). On a UMA box where the
model already owns most of RAM, that is the largest allocation the server makes,
and the OOM killer reaping it takes the whole server — losing the very
conversation the store existed to protect. Now: a byte bound on the queue, and
`LLAMA_SERVER_STORE_MIN_FREE_MB` (default 3072) skipping any store that would
leave less than that free. Persistence is best-effort by design — a skipped
store costs a re-prefill.

The margin has two failure modes and must sit between them: below the OOM
killer's trigger it does not protect, above the under-load available-memory floor
it blocks *every* store and silently disables the disk cache. On this box
(earlyoom `-m 2,1` ⇒ 2492 MiB) that window is ~1.5 GiB at `-c 262144` and only
~250 MiB at `-c 393216`, which is why 384k is not a supported serving config
here. Nothing needs it: the DS4-Flash template gates max reasoning on
`reasoning_effort` alone and never references context length.

**`4c0980758` — prefill guard.** Refuses a cold bulk re-prefill with 503.
Agentic traffic at 0% cache reuse drove 33 back-to-back full re-prefills and
wedged the machine at every duty level tried. Cache reuse is unaffected.

**`8ca8f1921` — refusal-direction ablation.** Arditi projection
(`cur' = cur − scale·⟨cur,dir⟩·dir`) as an alternative to additive control
vectors.

**`641c61a8c` — `reasoning_effort` passthrough.** It was parsed and dropped, so
a template branching on it never saw it.

**`45703be51` — revert of upstream `257813839` (TENSOR_READ_LAZY handling).** It
routes every lazily-read tensor to the generic CPU buffer type, which moves our
28.8 GB `per_layer_token_embd` out of `CPU_Mapped`. Measured cost: **73% of
llama-bench prefill** (812.70 -> 222.70 t/s) with the GPU at 31.5% busy and 22-38 W,
clock unthrottled. The measurement stands; the explanation this entry used to give
("the GPU reads the mmap'd pages directly, the CPU buffer puts the gather on the
wrong device") is **wrong** and was corrected on 2026-09-01: `ggml-cuda.cu`
hardcodes `integrated = false` on non-HIP, `GET_ROWS` has op-batch-size 0 so it is
never offloaded, and the PLE gather runs on the CPU backend either way — on ONE
thread (`ggml-cpu.c` pins `GET_ROWS` to `n_tasks = 1`), demand-faulting the mmap one
row at a time. The live server's main thread carries every one of its ~700k major
faults. Why the generic buffer is 4x worse than `CPU_Mapped` has not been
re-diagnosed. See the PLE cold-prefill item in the 2026-09-01 decision.

**`ec4d1e89f` — decode is exempt from the power duty cycle.** The firmware cap the
throttle exists to dodge is a *prefill* effect; taxing decode too cost ~21% of tg
for no thermal benefit. Spans shorter than `GGML_CUDA_POWER_EXEMPT_MS` (250) accrue
no debt. Prefill still pays in full — verified by the `[cuda-power]` telemetry
reporting 85.0% effective duty and zero exempt spans on a pp4096 run.

**Superseded by upstream at the 2026-09-01 rebase onto `3466812d1`, dropped from
the branch:** `37ac8c297` (qwen4exp in `llm_arch_supports_rs_rollback`; the
`LLAMA_QWEN4EXP_RS_ROLLBACK` env toggle went with it) and `f2580c704` (per-token
conv-state snapshots for the ring) are upstream `0eadefebd` (#28123) line for line;
`86b5fdc36`/`12030fe86` (windowed n-gram history scan, `LLAMA_KV_NGRAM_WINDOW`) is
subsumed by upstream `b356fa262` (#28040), which makes the lookup O(log n) through a
per-sequence position set.

**`53eb047b6` — argsort prefers the capture-safe radix sort.** Upstream treats
`DeviceSegmentedRadixSort` as the constrained fallback; on GB10 it is the faster
path (0.729 s across 20376 launches vs 1.344 s across 984 for `DeviceSegmentedSort`
in one capture).

**`3b0aef182` — no more rescanning from zero.** `split_simple()` searched for the
first unused token from index 0 on every call over a batch sized by the whole
prompt; `llm_graph_input_ple::set_input()` heap-allocated a 3-element vector inside
its per-token loop. Both are O(n^2) in prompt length.

**`77bb6a95b` — QSA attention gathers the selected cells at depth.** The masked
scan read every K/V row of the cache in each of the 12 QSA layers per decode token
(3.2 GB at 131k, an 11.8 ms floor before the kernel's own overhead; the CUDA mask
skip is a suffix bound gated on `Q->ne[1] >= 1024`). `build_qsa_gather` gathers
the 2051 selected rows per query, each query on the flash-attention stream axis
with its own window and its mask gathered at the same cells, padded to 2304 rows
with a -inf tail so the vector and GQA kernels stay eligible. Engages when
`4*n_tps*width < n_kv`; `LLAMA_QWEN4EXP_QSA_GATHER=0` forces the scan, `=2` the
gather. Shape after #27977, which upstream never resubmitted.

**`fa7e7cf2c` — pooled indexer keys are cached in the indexer cache's V plane.**
Every graph recomputed every block key (gather all raw keys, add, norm, rope over
all blocks, x12 layers): 18-21 ms per token at 131k. The V plane, which scoring
never reads, is retyped F32 with `idx_dim/ratio` per cell so a block's cells are
one key row, and a graph pools only the blocks this ubatch wrote or whose row
holds something else, or everything after a structural change (non-suffix
`seq_rm`, `seq_cp`, shifts, restores, a stream with several sequences, 2d
positions). Bit-identical to recomputing, checked by `test-qsa-pool-cache` on the
generated model. The plane shrinks from 512 to 128 B per cell (1.5 GiB to 384 MiB
at 262k). `LLAMA_QSA_POOL_CACHE=0` disables. The host `set_input_qsa` scan is
still O(n_kv) per ubatch.

**`35aee0e4c` — F32 weights stay on the mat-vec kernel up to batch 8.** Upstream
hands F32 x F32 to cuBLAS sgemm from `ne11 == 4`, or to the TF32 mma path when
`ne01 % 32 == 0`. On qwen4exp that is 216 latency-bound cuBLAS launches per
speculative verify step (HC inject, shared-expert gate, delta-net alpha/beta) and
a router computing its logits with 10-bit mantissas at widths 4..8 only. Same rule
the mmvf batch-invariance pin applies, now on by default.

**`aa5f5dda9` — the server stops accepting draft tokens past the first EOG.**
Tokens chained past an end-of-generation stayed in the KV and recurrent state; a
recurrent model cannot trim them on the next turn, so that turn re-prefilled the
whole previous answer (upstream #28049). Not seen biting in our logs (the
`<|im_end|>\n<|im_start|>` continuation extends the prefix), fixed anyway.

**`d3ca42258` / `732b007f1` — the n-gram table gather runs on every thread, with the pages
advised ahead.** The PLE gather never touches the GPU: `integrated` is hardcoded
false on CUDA, `GET_ROWS` is never offloaded, and ggml-cpu pinned it to one task,
so one thread demand-faulted the 28 GB mmap one 90-byte row at a time (every one
of the live server's ~700k major faults sat on its main thread). A gather of
4096 rows or more now uses all threads, and `set_input` hands the ubatch's pages
to the kernel with `MADV_WILLNEED` first (`process_madvise` in batches, plain
`madvise` fallback). Same bytes out; cold real-text prefill is the target, warm
prefill and decode unchanged. `LLAMA_QWEN4EXP_PLE_PREFETCH=0` disables the
advice. Upstream #28136 reaches the same place with a pread worker pool.

**`6bd7c7bd2` — the MTP draft attends sparse over its own indexer cache.** The
reference MTP block is a QSA layer; ours attended dense over a plain KV cache, so
at depth the draft saw the whole context where it was trained on a 2051-cell
selection. The draft context now gets the hybrid wrapper with the indexer cache
and a recurrent cache with no layers (which lets a sequence be cut anywhere, so
rollback keeps working), and `graph_mtp` runs the trunk's QSA path. The draft
file already carries the indexer tensors. `LLAMA_QWEN4EXP_MTP_QSA=0` restores the
dense draft. Acceptance at depth is the measurement that decides the default.

**`7706dc7b4` — QSA selects blocks first, then cells among them.** The single-stage
selection expanded block scores to every cell and sorted n_kv entries per query,
a cost and a surface (n_kv x n_tokens f32 at prefill) that grew with the context.
Two stages: top-k of top_k/ratio + 2 blocks, gather their cells from a per-stream
block table (the spare block carries the tail, pads marked -inf), then the final
width over ~2056 candidates. Same selected set, checked by the new `.topk` output
of `test-qsa-pool-cache`. Engaged at n_kv >= 8192 on single-sequence causal
streams; `LLAMA_QSA_TWO_STAGE=0` disables, `=2` forces.

**`54b6a4da0` — SSE pings cover the wait for the first result.** The stream sent
nothing until the first token, so a multi-minute prefill at depth looked like a dead
connection to the client: all 28 cancels in the 2026-09-01 log fired at 292-300 s,
each one a 134k-token re-prefill thrown away (26% of that day's wall). The first
result now gets one ping interval; past it the headers go out and the existing
`--sse-ping-interval` loop pings until it arrives. See "qwen4exp serving" for the
checkpoint window that removes the re-prefill itself.

**`93e512b57` — the pooled window is gated on real 2d positions.** `is_pos_2d()` is
`n_pos >= 3`, true for every text batch of an mrope model, so the first GPU run of
the pooled cache repooled everything on every graph and measured a null
(10.73 vs 11.08 t/s at d131072). Only tokens whose position rows disagree are
images. `LLAMA_QSA_POOL_TRACE=1` prints every window decision; the CPU model shows
prefill windows of 8 and decode windows of 1 after the fix. Lesson: an env-gated
fast path needs a trace of *which path ran* before its first benchmark.

## Measured: what pays and what does not

Ablation at two shapes, same config, one server boot per leg. Acceptance was
byte-identical in every leg except `MOE_GATE_FUSE`.

| Flag | shallow (27 tok) | depth (15,106 tok) | verdict |
| --- | --- | --- | --- |
| `LID_SHORTCUT` | **−6.0%** off | — | keep |
| `MOE_GATE_FUSE` | **−5.7%** off | — | keep (also changes numerics) |
| `CSA_TILE` + `FA_SPLIT` + `LID_RADIX` + `FP4_MMA` | flat | **−7.6% pp / −5.3% tg** off | keep — this is why rule 2 exists |
| `UNION_GATHER` | flat | flat | **removed** (`a5d74026d`) |
| `FUSE_INGEST` | flat | flat | **parked**, see below |
| `GRAPH_SHAPE_KEY` | flat | flat | keep — bounds cache growth under agentic churn, which no single-prompt benchmark exercises |
| `QWEN4EXP_RS_ROLLBACK` | flat | **+45.9% tg** (28k) | keep — the sharpest rule-2 case yet |
| `KV_NGRAM_WINDOW` | — | +4.5% tg / +3.6% pp (28k, single leg) | keep — asymptotic, but see the floor below |
| `BATCH_USED_LB` | — | **+6.75% tg** (28k, 19 reps) | keep |
| `CUDA_ARGSORT_RADIX` | — | +2.13% (at the floor) | keep on GPU-time evidence, not on this |
| `CUDA_POWER_EXEMPT_MS` | +21.4% tg | — | keep |

`LLAMA_DSV4_MOE_GATE_FUSE=0` changes acceptance (0.6545 → 0.5977) and output
length. **It is not a clean A/B control** — do not reach for it to isolate an
unrelated bug.

### This rig cannot resolve better than ~2%

Measured, not assumed: two legs of the **identical** configuration, 19 repeats
each, came out 44.99 and 45.92 — a **2.07%** spread. The within-leg standard error
is 0.18%, so per-rep error bars understate the real uncertainty by an order of
magnitude. The noise has a timescale longer than one leg's bench window, which is
why more repeats inside a leg do not help; only more interleaved legs do, at ~10
min each.

Consequences for rule 1: an effect under ~2% cannot be shown or refuted by A/B
here, so such a change must be justified by mechanism or by direct instrumentation
(a kernel-time delta from a profile, an asymptotic argument) rather than by a
throughput number. Two traps that cost real time this session: a 4-repeat window
reports a mean that looks precise to 0.3% while sitting up to 1.3% off the true
mean. Gate on `draft_n` / `draft_n_accepted`, which were byte-identical across
every leg of every A/B here.

An earlier version of this section also claimed that completion text is not
run-to-run deterministic at temperature 0, and used that to argue text-hash
equality could never gate a change. **That was wrong**, and it was wrong in the
expensive direction: it retired the one check that would have caught the
speculative-decoding divergence below. `test-spec-decode-exactness` measures it
directly — single-token decode, chunked decode, chunked decode with rollback,
and all six checkpoint save/restore combinations each reproduce bit-for-bit
across fresh contexts on both qwen4exp and Ministral-3B, `max|dlogit| = 0`. What
is *not* stable is the same prompt run twice through the speculative server, and
that is a property of the server loop, not of the model math.

## Known debt

**The MTP draft-coverage warning is over-eager, and cost me a wrong entry here.**
`common_speculative_impl_draft_mtp::begin` (`common/speculative.cpp:1746-1757`)
warns when `ctx_dft`'s position is behind the prompt, and it fired 11 times
during the 2026-08-30 quality A/B (`ctx_dft pos_max=29 < N-1=370`). I recorded
that as a real coverage gap that collapsed acceptance on those requests. **It is
not**, on the evidence available:

- `llama_decode[N] returned` appears **0** times in that server's log, and
  `inconsistent sequence positions` **0** times, both of which are `LOG_ERR` and
  would be visible at default verbosity. So the draft never failed to seed and
  drafting was never disabled.
- Speculation was healthy on the same server: tg 32-37 t/s against 25-27
  no-spec, and `diverge.py` measured acceptance 0.74-0.88 on it.

The likely reading is that `begin()` measures coverage before the ingest
pipeline has finished, so the predicate is right and the conclusion in its own
message ("Drafts may degrade") is not. Worth retexting or moving the check, not
worth chasing as a data-loss bug.

**Open lead, untested:** on an ordinary prefix-cache hit the first ingested draft
row may be paired with `pending_h` still holding the *previous* request's last
`h` row (`common/speculative.cpp:1804-1808`), because the checkpoint path that
would repair it (`common_speculative_set_state`) is gated on
`pos_min >= pos_min_thold`, which is false for a non-SWA target on a plain prefix
hit. If real this would degrade drafts on *every* warm request rather than some.
Not measured.


**`FUSE_INGEST` should be removed.** Measured flat at both depths, so it is
unnecessary complexity by rule 1. Retained only because unpicking it touches
five pieces of cross-round driver state (`pending_g_last`, `verify_tok`,
`verify_g`, `fuse_end`, `keep_pos`) across 8 sites, and the failure mode is
wrong draft tokens rather than a crash. Do it as its own change, validated
against the depth harness where acceptance (0.8316) is far more sensitive to
draft quality than the shallow one.

## Gates

| Gate | Value |
| --- | --- |
| PPL | 6.0735 ± 0.10675 — **bit-identical** to the pre-adoption baseline, same model and command |
| DSpark acceptance | 178/280 = 0.6357 at c16384 n=2, output byte-identical across legs |
| DSpark acceptance at depth | 79/95 = 0.8316 at 15,106 tokens |
| backend-ops | 7 DSV4 ops, 2/2 backends each |
| server tests | disk-cache 3/3 |

**The PPL gate command**, which was not written down and cost a wrong run to
recover (a code corpus at c32768 scores 1.16, nowhere near this gate):

```
llama-perplexity -m <UD-IQ3_XXS shard 1> -f ~/models/datasets/wiki.test.raw \
  -c 512 -b 2048 -ngl 999 -fa on --no-mmap --chunks 100
```

wikitext, `n_ctx=512`, 100 chunks, `n_seq=4` (implied by 2048/512). The same
protocol produced the older 5.0291/5.0177 anchors on the previous quant — the
number moved because the **model** changed to UD-IQ3_XXS-v2, not the command.

The 5.0177 PPL figure in older notes is **stale** — it predates ~90 upstream
commits. Never report a PPL delta without rebuilding the baseline.

**Acceptance has no standalone harness.** It is read from the server's
per-request `draft_n` / `draft_n_accepted` timings (see `gb10-thermal/soak.sh`
for the extraction), but the prompt set behind the 0.6357 and 0.8316 figures is
not recorded anywhere, so those two gates are not currently reproducible from
the repo alone. Write the probe down the next time it is run.

`ds4-tile/FLAGS.md` says text-hash equality is not a valid equivalence check for
a spec flag, on the grounds that speculation preserves the target distribution
so the text matches whatever the draft proposed. Read it narrowly: it is a fair
warning that a text match does not prove a *draft-side* flag did anything, and
`draft_n`/`accepted` remain the right gate for that. It is not a licence to skip
the text comparison, because the premise fails here — see **Speculative decoding
is not token-exact** below.

### Re-validated on the 2026-08-10 rebase onto `030ebb558`

| Gate | Result |
| --- | --- |
| PPL | 6.0735 ± 0.10675 — **exact match**, value and error bar |
| backend-ops | 13236/13236, 2/2 backends |
| DSV4 ops | 226/226 across the nine fused ops |
| DSpark acceptance | **not run** — no recorded prompt set (see above) |

One `CONV_TRANSPOSE_1D` case failed on the first full backend-ops pass
(`ERR 4.8e-5 > 1e-7`, `ne_input=[1,7,1,1]`) and passed on the re-run, and passes
in isolation on both this tree and pristine `030ebb558`. We touch no conv code.
Treated as upstream flakiness, not a rebase regression — but if it recurs, it is
a real data race worth reporting upstream rather than a threshold to relax.

### Re-validated on the 2026-08-30 rebase onto `a7cc83bba`

21 upstream commits inherited. **Zero conflicts**, but 13 of our 45 commits took
merged content rather than applying byte-identically — a clean rebase message is
not the same as an unchanged tree, so the full gate set was run rather than a spot
check.

| Gate | Result |
| --- | --- |
| PPL | **6.0735 ± 0.10675 — exact match**, value and error bar |
| backend-ops | 13919/13919, 2/2 backends (was 13765; upstream added 154 cases) |
| DSV4 + HC ops | 208/208, 2/2 backends |
| `test-llama-archs -a qwen4exp` | OK, GB10 8.87e-08 / CPU 0.00e+00 |
| qwen depth (28k, MTP n6) | 45.18 t/s, acceptance 0.8240 (pre-rebase 43.48 / 0.8240) |

**One break the gates could not have caught.** Upstream `bebc9350e` (#27969)
renamed `--tensor-read-lazy` to `--lazy-mode` / `-lzm`, so the server refused to
start with `error: invalid argument`. Nothing in backend-ops, the arch test or the
PPL gate passes that flag — it is a *serving recipe* dependency, and the only thing
that surfaces it is actually starting the server the way production does. Worth
keeping in the post-rebase checklist: **boot the real recipe, not just the gates.**

Picked up in this rebase and directly relevant: `f1793c1c4` (CUDA: fast
`mm_ids_helper` path for any `n_expert_used`). We were excluded from the fast path
by the old `warp_size % n_expert_used == 0` gate — 32 % 10 = 2 — and the commit
instantiates `launch_mm_ids_helper<10>` for exactly our config. The kernel goes
**4.229 ms -> 0.607 ms per call, 11.9% -> 1.9%** of prefill, worth **+9.5%**
end-to-end on pure upstream (741.98 -> 812.70).

### The 21 commits also carried a 73% prefill regression, bisected

`llama-bench pp4096` fell from ~675 to ~210 after the rebase. Bisected on **pure
upstream** with none of our code in the tree, five steps:

| commit | pp4096 |
| --- | --- |
| `c589f0ed1` (old base) | 747.62 |
| `0b5be7e4a` | 741.98 |
| `f1793c1c4` mm_ids fast path | **812.70** |
| `257813839` TENSOR_READ_LAZY | **222.70** |
| `a7cc83bba` (tip) | 211.67 |
| `a7cc83bba` + revert | **813.86** |

Reverted in `45703be51`; our tree went ~210 -> **754.42**, i.e. **+11.8% over the
pre-rebase 673.88** once the mm_ids win is no longer masked.

**The serving path never saw it.** `serve-qwen` passes `-lm mmap --lazy-mode on`
explicitly and measured 570-580 t/s prefill both before and after the revert;
llama-bench uses the default `auto` and took the full hit. That is worth
remembering as a general hazard: a microbenchmark and the server can diverge by 4x
on the same tree, so **a regression in one is not evidence about the other, in
either direction.**

### Re-validated 2026-08-30, after touching shared code for qwen4exp

Not a rebase. The qwen4exp decode work landed changes in code DS4 also runs
(`get_prev_tokens` in `llama-kv-cache.cpp`, `split_simple` in `llama-batch.cpp`,
the argsort dispatch, and the duty-cycle debt path), none of which qwen4exp's own
gates cover. DS4 was re-gated on its own model to catch that.

| Gate | Result |
| --- | --- |
| PPL | **6.0735 ± 0.10675 — exact match**, value and error bar |
| backend-ops | 13765/13765, 2/2 backends (was 13752; upstream added 13 cases) |
| DSV4 + HC ops | 208/208, 2/2 backends |
| DSpark acceptance | **not run** — still no recorded prompt set |

The PPL match matters more than usual here: `split_simple` and `get_prev_tokens`
are on every model's prefill path, and the batch cursor carries an invariant that
`split_equal` can violate. An exact PPL on a model that exercises different split
and cache paths than qwen4exp is the evidence that neither change altered
behaviour.

### Re-validated on the 2026-08-27 rebase onto `6fdd0ac89`

301 upstream commits inherited. Conflicts in 6 of our 31 commits, 15 hunks
across 9 files.

| Gate | Result |
| --- | --- |
| PPL | 6.0735 ± 0.10675 — **exact match**, value and error bar |
| PPL (older UD-IQ3_XXS quant) | 5.0291 ± 0.07884 — exact match to *that* model's anchor |
| backend-ops | 13752/13752, 2/2 backends |
| DSV4 ops | 229 cases across the ten fused ops |
| DSpark acceptance | **not run** — still no recorded prompt set |

Two exact PPL matches on two different quants is what licenses the three
upstream adoptions below: each changes the DSV4 graph, and neither number moved.

**Upstream has converged onto much of this work.** `common_speculative_impl_draft_dflash`
now carries `is_dspark`, `sample_from_anchor` and `mask_token_id` upstream. Their
DSpark block still lacks the Markov-head chaining, which is the defect that collapses
a block to its anchor (26.0% / 1.8% / 0% / 0% / 0% per-position), so ours is kept.
Upstream also added `sample_from_anchor`, read from `dflash.sample_from_anchor`; our
0731 draft GGUF does not set that key, so it defaults true and the anchor semantics
match what this fork did unconditionally. **If a future draft sets it false, the
block layout changes under us** — check that key before trusting an acceptance number.

Adopted from upstream, shrinking the diff:

- **`ggml_rope_set_offset()`** (`873e5d8e3`) replaces the view/rope/concat partial-rope
  dance in `build_compressed_kv_reduce_finish`. It sets `op_params[15]` on the rope
  node so rotation starts at `n_embd_head_nope`; one concat, two views and a cont drop out.
- **Single-gather overlap read.** Upstream reads all `2*ratio*n_blocks` rows with one
  `ggml_get_rows` and views prev/cur out of it, instead of splitting the index tensor
  and gathering twice.
- **`need_embd` removal** (`f785fc9ea`). This was flagged as the rebase's semantic
  hazard — silent acceptance collapse if the drivers stopped getting target embeddings.
  It is **inert**: every impl in this fork returned `need_embd() -> false`, and
  `server_slot::need_embd_nextn()` had **no caller**. The drivers enable what they need
  directly in their constructors (`llama_set_embeddings_nextn(ctx_tgt/ctx_dft, ...)`).
  The whole vestigial chain — both virtuals, both free functions, the header decls and
  the slot accessor — is now gone, and this fork matches upstream exactly there.

Kept against upstream, with the reason:

- **`state_read` stays unconditional.** Upstream guards the block-cache reads with
  `if (!partial_only)`; our writer emits them in partial mode too via `raw_flags`.
  Adopting their guard would desync the reader from our own on-disk format.
  Their per-seq `clear_compressed(seq_id, ...)` helper is a genuine improvement over
  our `kv_csa->clear(true)` (which clears *all* sequences) — **queued, not taken**,
  because changing restore scope mid-rebase is not gateable by PPL.
- **`reasoning_effort` passthrough** is now upstream's, better than ours (it erases the
  kwarg on `"none"`). Our deviation is down to one condition: an explicit
  `chat_template_kwargs` entry still wins over the OAI field.

One adaptation was forced: upstream changed `json` to `common_json`, whose iterator is
not random-access, so `std::sort` over a `json` array no longer compiles. The slot-saves
listing now sorts a plain vector and builds the array afterwards.

## qwen4exp serving

`serve-qwen` (in `~/bin`) is the validated recipe; every value in it was measured.
The three that are not obvious:

- **`--ctx-checkpoints 8 --checkpoint-min-step 16384 --cache-ram 0`** (2026-09-01 log
  mine). pi compacts at 75% of the context and re-sends a prefix cut at the last user
  message, ~56k behind the head; four checkpoints ~8-11k apart reach back ~35-40k, so
  every such turn re-prefilled from the system prompt and pi's ~300 s timeout cancelled
  it, eight cycles in 175 min. Eight checkpoints at >= 16k spacing reach ~131k for ~3 GB
  host RAM at 200k (a checkpoint is ~1.35 KiB per token of depth). The host prompt cache
  cannot hold a > 180k session (131k = 5.8 GB) and only serves slot switching, so it is
  off. The log also shows acceptance FLAT with depth (0.78) with drafted length median
  3.6, so `n_max 6` is not binding and `p_min 0.7` is.
- **`-lm mmap --lazy-mode on` is mandatory.** The PLE is marked
  `TENSOR_READ_LAZY`, but `-lm auto` resolves to `none` on GB10, so the default
  loads all 26.82 GiB resident and lazy never fires. Confirm by host memory, not
  by log line — lazy leaves ~50 GiB available, resident leaves ~24 GiB.
- **The draft's head is Q5_K, not Q8_0.** `output.weight` was 675 MB of the
  draft's 827 MB per step and 12.6% of all decode GPU time; Q5_K is 437 MB for
  +8% tg at flat acceptance. Free of quality risk by construction — the target
  verifies every token, so a weaker draft costs acceptance, never correctness.
  **Q4_K is slower than Q5_K** despite reading 22% fewer bytes; do not assume
  `mul_mat_vec_q` is monotonic down the quant ladder.
- **The recurrent rollback ring is load-bearing at depth.** `qwen4exp` was missing
  from `llm_arch_supports_rs_rollback`, which silently forced a full state snapshot
  to host on every speculative cycle. Flat at 30-token prompts, **+45.9% tg and
  +16.6% acceptance at 28k** (`37ac8c297`, now upstream `0eadefebd`). Rule 2 in the
  flesh — shallow would have argued for deleting it.
- **Backend sampling is a loss here, and now we know why.** CUDA 13.0 ships CCCL
  3.0.1; `top-k.cu` needs CCCL >= 3.2 for a real `DeviceTopK`, so below that
  `ggml_top_k` falls back to "argsort + copy" — a top-k of 10 becomes a **full sort
  of the 248,320-wide logit row**. Target-side `-bs` measured ~15% down; the draft
  defaults to backend sampling and `--no-spec-draft-backend-sampling` is worth
  +0.7% at 28k. **Both should be revisited if the toolkit's CCCL crosses 3.2** —
  the verdict is about a missing kernel, not about backend sampling as an idea.
- **`--spec-draft-n-max 6`.** The optimum was 1 only because `graph_mtp`
  published `t_h_nextn` before applying `inp_out_ids` (fixed in `278b29bc7`);
  n=8 regresses because expert traffic does not amortise across the verify
  batch — each token routes to its own 10 of 512 experts.

Cumulative against the pre-`ec4d1e89f` config: **35.71 → 44.45 t/s (+24.5%)**,
acceptance flat at ~0.75.

### Speculative decoding is not token-exact, and cannot be made so

Greedy speculative decoding is supposed to reproduce the non-speculative output
token for token. On qwen4exp it does not: every prompt in `qwen-evals/diverge.py`
diverges from the no-draft answer. The working theory for a long time was that
some piece of per-token recurrent state was not being rolled back. It was wrong.

`test-spec-decode-exactness` compares each decode path against a single-token
baseline and against a second run of itself. On UD-IQ4_XS, prefill 8, 21 tokens,
chunk 7, `n_rs_seq` 6:

| phase | max abs logit delta | argmax flips |
| --- | --- | --- |
| `[det]` every path twice | 0 | 0/21 |
| `[ckpt]` 6 save/restore combinations | 0 | 0/21 |
| `[shape]` chunked vs one-at-a-time | 4.62 | 2/21 |
| `[roll]` chunk, partial accept, roll back | 4.62 | 2/21 |

`[roll]` matching `[shape]` bit for bit, at both acceptance patterns and every
rollback depth 1..6, is the whole answer: **rollback contributes nothing.** The
divergence is that the target computes different logits for the same token
depending on how many tokens sit beside it in the ubatch, and speculation is
precisely a change of that shape — one token per decode without a draft, `1 +
n_draft` with one.

The chain, from `SPEC_TRACE=1`:

- the first `MUL_MAT` already moves, by 3.8e-06 on a `[320 x 1]` row
- through layer 0 everything stays small: `linear_attn_out` 6e-08,
  `ffn_moe_logits` 9.5e-07, `ffn_moe_probs` 7e-10
- from layer 6 the `ffn_moe_argsort` rows differ in their **top 8 entries**, by
  as much as 363. With 512 experts and top-10 there are near-ties everywhere, so
  a 1e-06 nudge to the router picks a different expert set
- a different expert set is not a rounding difference, and the rest of the stack
  compounds it into whole logits

Chunk 1 is exact; chunk 2 already reaches 4.29 — a hard step, not accumulation.
It is also not ours and not qwen4exp's: dense Ministral-3B on the same test
gives 0.22 (Q4_K_M) and 0.38 (Q8_0). The env-gated changes of our own that
touch these paths (`GGML_CUDA_ARGSORT_RADIX`, `LLAMA_BATCH_USED_LB`; the former
`LLAMA_KV_NGRAM_WINDOW` scan is now upstream's #28040) are each bit-identical when
disabled.

(An earlier note here said "forcing MMQ changes nothing". Disregard it —
`GGML_CUDA_FORCE_MMQ` and `GGML_CUDA_FORCE_CUBLAS` are `#ifdef` build macros,
not environment variables, so setting them in the environment did nothing at
all. The run was a no-op, not a control.)

So exactness is not reachable by fixing *bookkeeping*, and `7d71c3a90` — which
did fix a real conv-state bug and moved three of the five diverge.py prompts a
long way — was never going to finish the job.

### It diverges on everything and costs nothing measurable

Since the divergence is real and not going away by itself, the question becomes
whether it costs anything. Same eval, same seed, same prompts, one server each
(`qwen-evals/run_eval.py --reps 3 --seed 4242 --reasoning low`), 7 items x 3 reps
x 2 phases per arm:

| | no-spec | spec |
| --- | --- | --- |
| OPEN | 18/21 (86%) | 20/21 (95%) |
| CHOICE | 18/21 (86%) | 19/21 (90%) |
| identical replies | — | **0 / 42** |

Paired, which is the statistic that matters: **spec better on 3, worse on 0, same
on 39.** Not evidence that speculation *helps* — three wins and no losses out of
42 is a sign test at p = 0.125 — but it is a clean absence of loss. Reply length
is flat too (4354 vs 4109 characters mean), so it is not trading quality for
brevity. `diverge.py` on the same servers puts the first differing token at
133-323 across four of five prompts with acceptance 0.74-0.88; the fifth
(`count`, "count from 1 to 60") is **token-identical**, which is the control you
would want — a task whose top-2 gap is enormous never flips.

One thing blunts this test and should be fixed before leaning on it harder: the
eval sits at its ceiling on six of seven items, so it can only show a loss. (The
draft-coverage warning also fired during it, but that warning was later shown to
be over-eager — see **Known debt** — so it does not qualify the result.)

So the practical position is: speculation produces **a different sample of the
same quality**, at ~+30% decode on this workload. What it does not produce is
the model's own no-draft answer.

### Batch invariance achieved at the kernel level, and what it bought

Five sources, every one of them a switch on the **width-1** path, all now behind
build flags that default off:

| flag | what it pins |
| --- | --- |
| `GGML_CUDA_MMVF_BATCH_INVARIANT` | F32/F16 mat-vec family (`mmvf.cu:830/850`) |
| `GGML_CUDA_MMVQ_BATCH_INVARIANT` | quantized mat-vec geometry (`mmvq.cu:419/1060`) |
| `GGML_CUDA_MMID_BATCH_INVARIANT` | `mul_mat_id`'s family switch (`mmvq.cu:1026`, fused-MoE refusal `ggml-cuda.cu:1851`) |
| `GGML_CUDA_FATTN_BATCH_INVARIANT` | FA family + `ncols1` (`fattn.cu:27/482`); umbrella — also implies the KV-split pin below |
| `GGML_CUDA_FATTN_KVSPLIT_INVARIANT` | FA KV work-split: one block per output tile, sequential KV walk (`fattn-common.cuh:1143/1184/1215`); defaults to the umbrella flag, force `0` for a control isolating it; gated on width <= 8 so prefill keeps stream-k |
| `GGML_CUDA_DISABLE_FUSION=1` (env) | fusion — runtime env read at `ggml-cuda.cu:3420`, mat-vec forms gated on `ne1 == 1` |

(The first four are raw preprocessor macros, not CMake options — set them via
`-DCMAKE_CUDA_FLAGS="-DGGML_CUDA_MMVF_BATCH_INVARIANT=1 …"`. The fusion knob is a
runtime env var and must accompany every pinned invocation.)

With all five, `test-spec-decode-exactness` reports **`max|dlogit| = 0`, 0/21
flips** at chunk 2 and chunk 7 with `-fa on`. A 7-token verify batch is bitwise
identical to single-token decode.

**Why every partial attempt read as a null.** Each pin removes one source while
the others keep injecting ~1e-06, and a 512-expert top-10 router turns any
perturbation into a different expert set. There was never going to be a gradual
descent — it is all-or-nothing, and the four "nulls" (including the `mul_mat_id`
pin, which *raised* the number from 2.84 to 4.50) were real progress the
measurement could not show. Do not read a single-pin null as evidence against
that pin.

**Cost: 8.1% of width-1 decode**, tg128 25.58 -> 23.52 on llama-bench, prefill
flat (the pins only touch widths <= 8). Compare SGLang's ~34% for the same
guarantee on dense models. Two oddities recorded rather than explained:
`pp512` carries +/-13% error bars and reads far below our 754 t/s, so treat it as
a did-not-regress guard only; and disabling fusion measures *faster* than leaving
it on with the pins (23.52 vs 21.97), which is the opposite of the prediction.

### End to end it is better, not exact -- and the reason is n_kv

Server built with all four pins, `GGML_CUDA_DISABLE_FUSION=1`, `diverge.py`
no-draft vs speculative, 1200 tokens:

| prompt | first diff before | first diff now | acceptance |
| --- | --- | --- | --- |
| hard | 323 | 388 | 0.85 |
| code | 300 | **938** | 0.78 |
| prose | 206 | **859** | 0.76 |
| math | 133 | **identical** | 0.84 |
| count | identical | identical | 0.92 |

Two of five token-identical, the rest diverging 1.2-3.1x later, and acceptance up
across the board (0.76-0.92 from 0.74-0.88) — consistent numerics make the draft
agree with the target more often.

**The harness cannot see what is left.** It runs at a fixed `n_kv` of 256; the
server's grows. FA's stream-k partition is a function of the **KV length**, not
the token count (`fattn-mma-f16.cuh:1820`, `iter_k = ceil(ne11 / nbatch_fa)`), so
a solo decode at position p sees `n_kv = pad256(p+1)` while the verify batch sees
`pad256(p+7)`. When those fall in different 256-buckets the split and the combine
order differ. That fits the evidence exactly: divergence now appears only after
hundreds of tokens, at whichever crossing lands differently, and the two short
prompts never hit one. (The next paragraph's "needs real kernel work" prediction
turned out wrong — the one-block-per-tile predicate was enough.)

### The KV-split pin validates — and the breach needs saturation, not a crossing

2026-08-31, `experiments/batch-invariance/run_sweep.sh`: three arms (plain / all
pins / control = pins with `-DGGML_CUDA_FATTN_KVSPLIT_INVARIANT=0`) x five
prefills, chunk 7, 64 tokens. Prefills 250/506/762 make the width-1 baseline and
the first chunk land in different 256-buckets (`pad256(prefill+1) !=
pad256(prefill+chunk)`); 8 and 260 are same-bucket controls. `[shape]`/`[roll]`
max abs logit delta:

| arm | p8 | p250 | p260 | p506 | p762 |
| --- | --- | --- | --- | --- | --- |
| plain | 5.43 | 11.16 | 11.26 | 10.72 | 10.30 |
| control (no KV-split pin) | 0 | 0 | 0 | 0 | **8.39** |
| all pins | 0 | 0 | 0 | 0 | **0** |

Three findings. **The pin works**: the pinned arm is exact at every prefill,
including the one that breaches the control. **The control isolates it**: the
only difference between the two pinned arms is the KV-split macro, so the p762
breach is that split and nothing else. **A bucket crossing alone is not
sufficient**: the control stayed exact at 250 and 506, where the prediction said
it would break. Read off the dispatch, the reason is that below saturation
stream-k hands every block exactly one KV unit at both lengths
(`nblocks = ntiles_KV * ntiles_dst` while that is `<= max_blocks`), the chunk's
extra units are fully masked and contribute exact zeros, and the real-key
partials group identically. Only once `ntiles_KV * ntiles_dst` exceeds
`max_blocks` (~48 here, first reached at the 768 -> 1024 crossing) do the block
boundaries start cutting inside the real range at length-dependent points. So
the divergence window opens at `n_kv` ~768 on this config and recurs at every
crossing after — which also fits the e2e first-diff positions once prompt
lengths are added in. (This grouping mechanism is dispatch-arithmetic, not
instrumented; the five cells are the measurement.)

The width gate added with the macro split (`fattn-common.cuh:1143`) means
prefill keeps stream-k; the arm equality above is between width-1 and width-7
schedules, which both sit under the gate.

### The cost, decomposed by depth — and where the old 8.1% actually lived

Same three arms, `bench_cost.sh` (tg128, r=10, all arms `GGML_CUDA_DISABLE_FUSION=1`):

| tg128 | plain | ctl (4 pins) | all pins (+KV-split) | total vs plain |
| --- | --- | --- | --- | --- |
| d0 | 23.61 | 20.66 | 23.14 | −2.0% |
| d4096 | 22.30 | 19.99 | 21.03 | −5.7% |
| d16384 | 19.96 | 18.24 | 17.20 | −13.8% |
| d32768 | 18.20 | 16.87 | 13.60 | **−25.3%** |

- **The KV-split pin is *faster* than stream-k at shallow depth** (+12% at d0 over
  ctl — one block per tile skips the fixup, exactly the short-context case the
  code comment at `fattn-common.cuh:1179` describes) and crosses over around
  d8-16k, reaching −19% pin-vs-ctl at d32k. The exactness bill is depth-rented,
  not flat.
- **The four dispatch pins alone cost −7 to −12%** (ctl vs plain), *shrinking*
  with depth as FA's share grows.
- **Prefill did not regress** under the width gate: `pp2048` 253-291 pinned vs
  226-252 plain across all depths (error bars ±16-41, treat as did-not-regress).
- **The old "8.1% pin cost" was mostly fusion-off.** Warm legs on the plain
  binary put fusion ON at 24.7-25.3 vs OFF at 23.0-23.6 (−5 to −8%; the one
  20.66 fusion-on leg was the cold first leg of an ABBA and should be
  discarded), and today's plain fusion-off 23.61 vs all-pins fusion-off 23.14
  is −2%: the recorded 25.58 → 23.52 collapse was fusion-off plus pins, not
  pins alone. The earlier oddity ("disabling fusion measures faster than
  leaving it on with the pins") was therefore a pin×fusion interaction, not a
  fork-wide fusion regression.

**The follow-up that matters:** fusion is only a divergence source because the
fused mat-vec kernels exist solely at `ne1 == 1`. Relaxing that gate to
`ne1 <= 8` and multi-launching the fused width-1 kernel once per column (the
same column-local trick the runtime mmvq pin uses) would make fusion itself
batch-invariant — reclaiming ~8 of the ~10 shallow-depth points that exactness
currently costs. Unbuilt; recorded here as the next lever.

### One binary, one env var: the runtime toggle

`GGML_CUDA_BATCH_INVARIANT` ([TAG_BATCH_INVARIANT_RUNTIME], `common.cuh`) turns
the pins on at run time in a default build: `1`/`all` for everything including
fusion-off, or a comma list of `mmvf,mmvq,mmid,fattn,fattn_kvsplit,nofuse` for
per-pin controls. Read once per process (stable across CUDA graph captures);
the build macros still force their pin regardless. Every pin is a host-side
dispatch decision except mmvq's width-2..8 geometry, which the toggle handles
by launching the *tuned* width-1 kernel once per column — column arithmetic is
column-local end to end, so each launch is bit-identical to a solo decode of
that column. That makes the runtime mmvq pin *stronger* than the build macro
(no detuning to the plain variant) but not bitwise-comparable to a compiled-pin
build; the invariance property (chunked == solo within one binary+env) is the
one that holds, and the one that matters.

Verified on `build-rt` (default build), prefill 762 (the breaching cell), chunk
7: env unset → `[shape]` 8.62 (plain behavior); `=1` → `max|dlogit| = 0`
everywhere, with the `nofuse` bit doing fusion's job (no `GGML_CUDA_DISABLE_FUSION`
set); `=mmvf,mmvq,mmid,fattn,nofuse` (all but the KV-split bit) → `[shape]`
8.75, reproducing the control arm's breach signature per-bit. Serving-cost
numbers should still come from compiled-pin builds — the mmvq multi-launch adds
~10^3 launches per ungraphed verify step — but for exactness work the
two-build-tree snapshot dance is retired.

**Serving cost, measured cleanly.** Same five prompts, `n_predict` 600,
`cache_prompt` off, ABBA within each config, both arms snapshotting the whole
`build/bin` tree and running under their own `LD_LIBRARY_PATH`:

| arm | legs | tg mean |
| --- | --- | --- |
| plain / no-spec | 21.86, 22.27 | **22.07** |
| pinned / no-spec | 20.21, 20.44 | **20.33** |
| plain / spec | 34.48 | 1 leg |
| pinned / spec | 32.21 | 1 leg |

**No-spec costs 7.9%**, which is the solid number — arms do not overlap, ordering
is consistent across the ABBA pairs, and it independently reproduces
llama-bench's 8.1%. Spec reads -6.6% but on one leg each with a 26-46 t/s
per-request spread, so treat it as provisional. The speculative gain survives
either way: **+56% plain, +58% pinned**. Acceptance is slightly *higher* pinned
(0.7400 vs 0.7245), matching the e2e run — steadier numerics make the draft agree
with the target more often.

**A harness trap worth not repeating.** `llama-server` is a 72 KB launcher; the
pins live in `libggml-cuda.so`, which it loads at run time. Snapshotting only the
executable makes both arms load the *same* library, and the A/B then reports a
confident four-arm null that never varied anything. The script now copies the
whole `bin` directory per arm and `cmp`s the two `libggml-cuda.so` before
measuring, aborting if they match. A clean
serving A/B needs the same workload both ways.

### The mechanism has a name, and the kernel is ours to fix

This is **batch invariance**, and it is well documented. Thinking Machines'
*Defeating Nondeterminism in LLM Inference* names the same three offenders —
matmul (data-parallel vs split-K by batch dimension), RMSNorm, and attention
(split-KV by query count) — and reports ~20% on matmul and vLLM 26 s -> 42-55 s
for 1000 sequences to remove it. SGLang shipped it for dense models at a
measured 34% average slowdown, explicitly **not** covering MoE. llama.cpp has an
open PR (#16016, since 2025-09-15, unmerged) for a CUDA deterministic mode, and
it covers FP16/BF16 matmul only — not the quantized path we are on.

Upstream issue **#25618 is our bug**, open since 2026-07-13 with 19 comments and
still `bug-unconfirmed`: "Speculative decoding (draft-mtp / draft-dspark): greedy
output diverges from vanilla on quantized targets". Independent reproductions on
Metal, Vulkan and ROCm; the reporter's BF16 target matched vanilla 10/10 while
Q4_K_M matched 9/10. Ankk98 took the Vulkan side apart into six path mismatches,
every one of them "the N=1 kernel is not the N>1 kernel", and fixed each by
forcing a single path — including one that is exactly ours: *"parallel verify
used integer-dot MMVQ (NUM_COLS > 1) while sequential recheck used NUM_COLS == 1.
Those paths disagree."*

On CUDA/GB10 the dispatch does **not** switch families — `ggml_cuda_should_use_mmvq`
returns true up to `MMVQ_MAX_BATCH_SIZE` = 8, so a 1-token decode and a 7-token
verify both land in `mul_mat_vec_q`. What differs is the geometry inside it:

| | `nwarps` | `rows_per_block` | variants available |
| --- | --- | --- | --- |
| `ncols_dst` = 1 | 4 (8 with our GB10 `halve_iters`) | 1 | plain, `small_k`, `halve_iters` |
| `ncols_dst` = 5..8 | 2 | 2 | plain only |

Different `nwarps` means a differently shaped cross-warp reduction through
`tmp_shared`. `mul_mat_id` is worse: `ncols_dst > 1` goes to a dedicated MoE
kernel (one warp per token) while `ncols_dst == 1` uses the general one.

**But the warp count is not the mechanism, and that is settled upstream.** On
#25618 thc1006 forced the `MMVQ_PARAMETERS_GENERIC` warp counts across widths
3..8 on sm_86, confirmed with SASS that the edit reached the machine code and
only there, moved kernel runtime by up to **26.68%**, and changed **not one
output byte** in 150 records per build. Their divergence rates do group along the
dispatch boundaries (widths 2-4, 5-8, 9+), so the table describes *where* the
boundaries are — it just is not what puts the widths in different groups. They
also established that divergence starts at **width 2**, the smallest speculative
width there is: 57 of 75 requests differed from baseline at `--spec-draft-n-max 1`.

That intervention never crossed the 1-vs-many boundary, which is the one that
matters for us — `rows_per_block` (1 vs 2) and the `small_k` / `halve_iters`
variant choice both change only there, and neither was touched. So the flag is
still worth running, but the expectation should be that it is not sufficient on
its own, and the leading candidates are the **family switches**: `mul_mat_id`,
flash attention's VEC-vs-MMA split, and MMVF-to-MMF for F16/BF16 weights.

`GGML_CUDA_MMVQ_BATCH_INVARIANT` (build flag, default off) pins the geometry to
the `ncols_dst == 1` arm and forces every width onto the plain variant, so all of
1..8 launch the same shape. It is not free: it gives up `small_k` and
`halve_iters` at `ncols_dst == 1`, which is the GB10 bs=1 tuning.

**It covers the dense path only.** `mul_mat_vec_q_case` returns at
`mmvq.cu:1018` for `has_ids && ncols_dst > 1` — the dedicated MoE kernel — before
the flag is ever consulted, and `mul_mat_vec_q_moe_launch` never calls
`calc_launch_params`. So on a MoE model every expert matmul still takes a
different kernel than the 1-token path. The flag as written cannot make qwen4exp
exact on its own.

### The full audit: what else is keyed on the token count

Ranked by expected contribution. Only the first two are measured; the rest are
read off the dispatch.

| | op | site | 1 token vs 7 |
| --- | --- | --- | --- |
| 1 | `MUL_MAT_ID` (512 experts x 48 layers) | `mmvq.cu:1018` | dedicated MoE kernel, 1 warp/token, **no cross-warp reduction at all** vs the general kernel's `tmp_shared` tree |
| 2 | dense quantized `MUL_MAT` | `mmvq.cu:415,543` | `nwarps` 4-or-8 vs 2, `rows_per_block` 1 vs 2 — **flag handles this** |
| 3 | fused up+gate+SwiGLU | `ggml-cuda.cu:1844` | one fused kernel at 1 token, three ops at >1 (fusion refused when `dst->ne[2] != 1`) |
| 4 | `FLASH_ATTN_EXT` family | `fattn.cu:462-482` | `fattn-vec` (fp32 warp dot) vs `fattn-mma-f16` (tensor-core tiles) — a different algorithm |
| 5 | `FLASH_ATTN_EXT` KV split | `fattn-common.cuh:1142` | stream-k cuts the KV range into a different number of chunks and recombines in a different order |
| 6 | dense F16/BF16/F32 `MUL_MAT` | `mmvf.cu:830,855` | MMVF (warp dot) at `ne11 == 1`, MMF (mma tiles) at >= 2, because `ampere_mma_available` is true on SM121 |
| 7 | `SUM_ROWS` (indexer score) | `sumrows.cu:36` | `(nrows/nsm) < 2` picks a different block width, so a different accumulate grouping — only inside one `n_kv` band |
| 8 | `TOP_K` (indexer) | `argsort.cu:98` | `DeviceRadixSort` vs `DeviceSegmentedRadixSort`; no FP arithmetic, but `relu` and `-INF` masking manufacture exact ties |

**Not pinnable without giving something up.** Once a matmul reaches cuBLAS
(`ggml-cuda.cu:1909`) the tiling and split-K are cuBLAS-internal and there is no
ggml-side knob. And pinning the FA stream-k split means using the 1-token split
for a 7-token verify, i.e. discarding the occupancy stream-k exists to recover —
the same trade Thinking Machines report as the dominant cost of attention
invariance.

### Suspects cleared

- **The gated delta net is innocent on CUDA.** `gated_delta_net.cu:170` launches
  one kernel with no `n_tokens` branch; `n_tokens` is only the trip count of a
  sequential loop. The `build_delta_net_autoregressive` / `_chunking` split at
  `delta-net-base.cpp:435` is real but dead here — with `n_rs_seq > 0`,
  `build_recurrent_attn` calls `ggml_gated_delta_net` directly and never reaches
  it. The `LLM_FUSED_OP_GDN_AR`/`GDN_CH` tags are capability probes read only by
  `llama-context.cpp:524`; they select no kernel. Worth an assert so the split
  cannot open up later.
- **The router argsort is a consequence, not a cause.** At `n_expert` = 512 the
  dispatch takes the bitonic early return (`argsort.cu:283`), one block per row,
  identical at 1 and 7 rows. The `nrows == 1` radix split is unreachable for it.
- **`RMS_NORM`, `SOFT_MAX`, `SSM_CONV`, `topk_moe`, the hyper-connection ops, the
  lightning indexer, and every elementwise op** are one-block-per-row or flat
  grids, with block width derived from the K dimension rather than the token
  count.

### A trap in our own trace tooling

The first version of `[trace]` reported `cache_s_l0` moving by 9.3e-04, far more
than anything else at layer 0, and it was wrong. A per-token activation is
`[n_embd, n_tokens]`, so its element 0 is the chunk's first token in both runs and
is comparable. A recurrent-cache row is the rollback ring, and slot 0 holds the
state after the **last** token of the ubatch — a different token in each run. The
number was one step of the recurrence, not an invariance failure. `cache_*`
tensors are now excluded from the comparison.

**A generated model cannot gate any of this.** On the `test-llama-archs`
qwen4exp model every phase scores an exact zero, including a 6-deep rollback run
with prefill 1, where the deeper snapshot planes were never written at all. Its
recurrent state simply does not reach its logits. This is why
`test-recurrent-state-rollback` is green in CI and fails on the real model.

### The snapshot ring goes stale after a narrow ubatch

Found while chasing the above, and unrelated to it. `[roll-ar]` advances R tokens
one at a time and rolls all R back, which must return to where it started:

| depth | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| max abs logit delta | 2.54 | 8.25 | 10.20 | 8.93 | 12.40 | 13.85 |

`build_recurrent_attn` copies only `min(n_seq_tokens, K)` snapshot planes into
the cache, and `build_conv_state` clamps its window rather than shifting, so
after a narrow ubatch the deeper planes still describe positions from an
earlier, wider one. A rollback reads them as if they were valid. This is what
makes `test-recurrent-state-rollback` fail its dirty-ctx case on the real model:
its reference logits come from a replay that had already read a stale plane.

A full reachability audit says **not reachable in our serving config**, and the
bounds are tighter than "it happens not to fire":

- **The speculative loop is safe by arithmetic.** With D drafted tokens the
  rollback depth is `D + 1 - accepted.size()` and `accepted.size() >= 1`, so the
  depth is at most D, while the same ubatch wrote `min(D+1, K) = D+1` planes.
  The requested plane is always one the immediately preceding ubatch wrote. The
  split guarantee holds too: `split_equal(..., n_keep_tail = n_rs_seq + 1)`
  admits a sequence only when its remaining count is 0 or >= `n_keep_tail`, so a
  generating slot's 7 tokens are never deferred mid-chunk. A corollary is that
  `use_ckpt_tgt` in the accept path can never be true in RS mode — that branch
  is dead code.
- **SPEC=0 is dead** — `n_rs_seq` is 0, `build_recurrent_attn` takes its
  single-plane branch, and there is no ring at all.
- **The prompt cache is protected by exactly one position.** For a hybrid,
  `seq_pos_min` is the max of the two sub-memories' minima and the recurrent
  minimum is its tail cell's `pos`, so `pos_min` is the sequence frontier. That
  makes the checkpoint gate and the partial-rollback condition the *same*
  condition, so whenever the trim could roll back, the checkpoint block already
  ran and overwrote `n_past`. Every ordinary checkpoint keeps the invariant
  `n_tokens == pos_min + 1 == pos_max + 1`, which leaves `p0 = cell.pos + 1`.

**The one breach needs `--slot-save-path`, and the line is ours.**
`create_checkpoint(*slot, 0, 0, pos_max)` at `server-context.cpp:2897` — added by
`09e12513a`, not upstream — synthesizes a checkpoint with
`pos_min = 0` against `pos_max = N-1`, which both bypasses the gate and makes
the restore land on `p0 = cell.pos` instead of `cell.pos + 1` — a depth-1 read
of a plane the restore never wrote, silently, with `seq_rm` returning true.
`POST /slots/0?action=save` then `?action=restore` then any completion reaches
it, and the synthetic checkpoint survives into the RAM and disk prompt-cache
tiers. Containment is one line: pass the real `seq_pos_min`, or use
`it->pos_max + 1` at the restore site. **We do not pass `--slot-save-path`.**

So the ring stays as it is. Both fixes cost more than the exposure: shifting the
planes is 6 x 112.57 MiB read and written per decode step, about 23% of a step at
45 t/s, paid always to protect a case we never take; and making the planes a true
ring — a per-cell base index, so a write moves the index instead of the data — is
free at runtime but reaches into `mem_cell`, `find_slot`, `s_copy`, both state
serialisers and every `build_conv_state` caller, for five architectures whose
only test models cannot detect a mistake in it.

### A different hybrid hazard, found on the way

`llama_memory_recurrent::get_can_shift()` returns **true** ("shifting the pos is
trivial for recurrent models"), but a mid-sequence removal is not representable
in a recurrent state at all. Called with a finite `p1 <= cell.pos` — which is
what context shift (`seq_rm(id, n_keep, n_keep + n_discard)`) and cache reuse
(`seq_rm(id, head_p, head_c)`) both do — `seq_rm` skips the partial branch, skips
the tail invalidation, matches no cell in its loop, and **returns true having
done nothing**. The attention cache is then shifted or reused while the recurrent
state still summarises the discarded tokens. No warning, and it needs no
speculation. Not ours to fix today, but do not enable `--ctx-shift` or
`--cache-reuse` on a hybrid model expecting correct output.

## Draft model

The draft must be built from the 0731 checkpoint's own tail shards via
`convert_hf_to_gguf.py --dspark --target-model-dir`, which emits arch `dflash`.
The standalone `-DSpark` HF repo is a pre-0731 preview and silently costs ~9
points of acceptance. Legacy arch-`dspark` GGUFs no longer load.
