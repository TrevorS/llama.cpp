# Deviations from upstream llama.cpp

Every difference between this branch and upstream, and why it exists. If a
deviation is not listed here it should not exist — delete it or add a row.

Base: upstream `3581ba0cf`. 15 commits, 53 files, `↑`4761 `↓`331.

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

`LLAMA_DSV4_MOE_GATE_FUSE=0` changes acceptance (0.6545 → 0.5977) and output
length. **It is not a clean A/B control** — do not reach for it to isolate an
unrelated bug.

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

The 5.0177 PPL figure in older notes is **stale** — it predates ~90 upstream
commits. Never report a PPL delta without rebuilding the baseline.

## Draft model

The draft must be built from the 0731 checkpoint's own tail shards via
`convert_hf_to_gguf.py --dspark --target-model-dir`, which emits arch `dflash`.
The standalone `-DSpark` HF repo is a pre-0731 preview and silently costs ~9
points of acceptance. Legacy arch-`dspark` GGUFs no longer load.
