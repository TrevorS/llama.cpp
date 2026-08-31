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
28.8 GB `per_layer_token_embd` out of `CPU_Mapped` and schedules the per-layer PLE
gathers on the CPU backend. GB10 reports
`pageableMemoryAccessUsesHostPageTables=1`, so the GPU can read those mmap'd pages
directly — forcing them to a plain CPU buffer costs **73% of llama-bench prefill**
(812.70 -> 222.70 t/s) with the GPU at 31.5% busy and 22-38 W, clock unthrottled.
The work is not slower, it is on the wrong device. Plausibly correct upstream for
discrete GPUs; wrong for the only part we serve.

**`ec4d1e89f` — decode is exempt from the power duty cycle.** The firmware cap the
throttle exists to dodge is a *prefill* effect; taxing decode too cost ~21% of tg
for no thermal benefit. Spans shorter than `GGML_CUDA_POWER_EXEMPT_MS` (250) accrue
no debt. Prefill still pays in full — verified by the `[cuda-power]` telemetry
reporting 85.0% effective duty and zero exempt spans on a pp4096 run.

**`37ac8c297` — qwen4exp joins the recurrent rollback allowlist.** It shares
qwen35's GatedDeltaNet state and the same generic `build_rs` path but was missed in
`llm_arch_supports_rs_rollback`, so `n_rs_seq` silently clamped to 0 and the server
fell back to a full 112 MiB state snapshot to host on every speculative cycle.

**`86b5fdc36` — the n-gram history scan is windowed.** `get_prev_tokens()` swept
from position 0 to fill a `below` fallback that only an M-RoPE gap can reach; for a
token ubatch it is O(context) per call for a value nothing reads.

**`53eb047b6` — argsort prefers the capture-safe radix sort.** Upstream treats
`DeviceSegmentedRadixSort` as the constrained fallback; on GB10 it is the faster
path (0.729 s across 20376 launches vs 1.344 s across 984 for `DeviceSegmentedSort`
in one capture).

**`3b0aef182` — no more rescanning from zero.** `split_simple()` searched for the
first unused token from index 0 on every call over a batch sized by the whole
prompt; `llm_graph_input_ple::set_input()` heap-allocated a 3-element vector inside
its per-token loop. Both are O(n^2) in prompt length.

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
  +16.6% acceptance at 28k** (`37ac8c297`). Rule 2 in the flesh — shallow would
  have argued for deleting it.
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

Chunk 1 is exact; chunk 2 already reaches 4.29. That hard step matches the
GEMV-to-GEMM switch, not any gradual accumulation. It is also not ours and not
qwen4exp's: dense Ministral-3B on the same test gives 0.22 (Q4_K_M) and 0.38
(Q8_0). Forcing MMQ changes nothing, and the three env-gated changes of our own
that touch these paths (`LLAMA_KV_NGRAM_WINDOW`, `GGML_CUDA_ARGSORT_RADIX`,
`LLAMA_BATCH_USED_LB`) are each bit-identical when disabled.

So exactness is not reachable by fixing bookkeeping, and `7d71c3a90` — which
did fix a real conv-state bug and moved three of the five diverge.py prompts a
long way — was never going to finish the job. The open question is no longer
"why does it differ" but "does it cost anything", which is an eval question.

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

**The speculative path cannot reach it** — a rollback there is always shallower
than the chunk that just wrote the planes — which is why it is recorded rather
than fixed. The two candidate fixes both have a real cost: shifting the planes
is 6 x 112.57 MiB read and written per decode step, about 23% of a step at 45
t/s, and paid on every step to protect a case we never take. Making the planes a
true ring — a per-cell base index, so a write moves the index instead of the
data — is free at runtime but reaches into `mem_cell`, `find_slot`, `s_copy`,
both state serialisers and every `build_conv_state` caller, for five
architectures whose only test models cannot detect a mistake in it.

## Draft model

The draft must be built from the 0731 checkpoint's own tail shards via
`convert_hf_to_gguf.py --dspark --target-model-dir`, which emits arch `dflash`.
The standalone `-DSpark` HF repo is a pre-0731 preview and silently costs ~9
points of acceptance. Legacy arch-`dspark` GGUFs no longer load.
