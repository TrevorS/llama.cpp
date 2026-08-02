# DSpark speculative decoding on DeepSeek-V4-Flash-0731

Record of the 2026-07-31 correctness pass. Target: Unsloth `UD-IQ3_XXS`
(4-shard GGUF). Draft: `DeepSeek-V4-Flash-0731-DSpark-mxfp4.gguf` (11.35 GiB,
83 tensors, 3 blocks). Box: DGX Spark GB10, `GGML_CUDA_POWER=85`,
`GGML_CUDA_GRANULARITY=layer`, `-fa on -ngl 999 -ub 2048 -np 1 --kv-unified`,
`--temp 0`, `-c 8192`.

**Headline: 9.36 t/s (−39% vs no-spec) → 27.4 t/s (+51%) at `n_max=2`, P95.**

> **Quant caveat.** Unsloth's UD-IQ3_XXS shipped with a defective IQ2_S grid until
> 2026-07-31 (HF discussion #9); it was re-quantized to IQ2_XS + more IQ3_XXS.
> All tables below marked *(fixed)* use the corrected build. The defective build
> cost **9.3% tg and 11.9% relative acceptance** — see "Quant quality feeds
> acceptance" near the end.

## Result (fixed quant)

| config | tg (t/s) | vs no-spec |
| --- | --- | --- |
| **DSpark n=3, P95** | **27.99 / 28.05** | **+55%** |
| DSpark n=2, P95 | 27.21 / 27.27 | +51% |
| DSpark n=3, P85 | 24.76 / 24.71 | +56% |
| DSpark n=2, P85 | 23.96 / 23.92 | +51% |
| no-spec, P95 | 18.09 / 18.05 | — |
| no-spec, P85 | 15.85 / 15.83 | — |

P85 n-curve (acceptance in parens): n1 19.70 (77.4%) · n2 23.94 (69.2%) ·
**n3 24.74 (58.3%)** · n4 23.56 (49.8%) · n5 21.06 (38.9%). Best `n` moved from
2 to 3 when the target quant was fixed — a better target shifts the optimum
deeper, so re-tune `n` after any target change.

### Confidence head (`--spec-draft-p-min`)

n=5, P85, fixed quant. Counts are cumulative from the per-round trace:

| `p_min` | rounds | drafted | accepted | acceptance | mean len | drafted/round |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 362 | 1801 | 723 | 40.1% | 2.997 | 4.97 |
| 0.5 | 100 | 293 | 199 | 67.9% | 2.990 | 2.93 |
| 0.7 | 87 | 207 | 179 | 86.5% | 3.057 | 2.38 |

**It halves the drafting work at no cost to accepted length** — mean len flat at
~3.0 while drafted-per-round drops 4.97 → 2.38. Block widths go from `{5: 357}`
(always full) to `{1:36, 2:19, 3:10, 4:7, 5:15}`.

Single-stream throughput barely moves, which the profile predicts: the draft is
only 7.8% of MoE time, so halving it has little to give. `p_min` is a
batched-serving lever (published: no benefit at concurrency 1, +13.8% decode at
32). **0.7 is ds4's retuned default** (from 0.9, commit `a51e6ec`); they also use
a 0.5 cold-start threshold that pauses drafting for 7 cycles when nothing has
been accepted yet.

Threshold semantics: cut on the **per-position** sigmoid, first position below —
matching DeepSpec's `_confident_prefix_length` and ds4 `ds4.c:32369`. The head is
an `AcceptRatePredictor` whose BCE label is a single-position accept probability
(SpecForge `dflash_family_model.py:1229`), so per-position is the coherent use of
it. sglang's `cumprod` survival is a different mechanism — a global top-k
scheduler across all (request, position) pairs under a profiled token budget,
irrelevant at `n_seq == 1`.

Evaluate confidence BEFORE the position's Markov GEMV so a cut skips it (ds4 does
the same); this driver already does.

Every flag introduced here is inventoried in `../ds4-tile/FLAGS.md`.

## Per-position survival

Per-position survival at `block_size=5`, showing the two structural fixes:

| state | pos0 | pos1 | pos2 | pos3 | pos4 | mean len |
| --- | --- | --- | --- | --- | --- | --- |
| as found | 26.0% | 1.8% | 0.0% | 0.0% | 0.0% | 1.278 |
| + Markov chaining | 56.7% | 17.3% | 2.4% | 0.0% | 0.0% | 1.765 |
| + RoPE `NORM` | 76.6% | 55.7% | 37.1% | 24.6% | 11.4% | 3.054 |

Ablation at `n_max=5` — each fix is necessary and they compound far beyond
additive:

| config | acceptance |
| --- | --- |
| neither Markov nor RoPE | 5.6% |
| RoPE only | 13.4% |
| Markov only | 15.3% |
| window off (other two on) | 14.6% |
| all three | **41.5%** |

## Defects fixed

1. **`block_size` read the wrong key.** Probed only `dflash.block_size`; DSpark
   exports `dspark.dspark.block_size` (= 5). Silently over-drafted at the 16
   default.
2. **Noise token.** `llama_vocab_mask()` returns −1 (this vocab has no
   `<mask>`); must read `dspark.dspark.noise_token_id` (= 128799). Every decode
   failed with −1 before this.
3. **Confidence head was post-norm.** Reference taps pre-norm `x`; fixed via
   `h_prenorm`.
4. **`cparams.dspark_draft_chain` was never initialized.** `llama-context.cpp`
   set `mtp_draft_chain = false` but had no line for the DSpark twin, so
   `dspark.cpp` read indeterminate memory and armed the in-graph chain at
   random. The chain's output (`t_dspark_meta`) had no consumer — dead code that
   corrupted the live logits buffer when it fired.
5. **Encoder `fc` F16 accumulator overflow.** `fc.weight` ships F16
   `[12288, 4096]`; the reference runs `main_proj` in bf16. DeepSeek-V4's
   massive-activation tokens (BOS / chat-template prefix) reach ±4.6e4 at
   prefill (vs ±20 at decode), so reducing 12288 terms saturates F16's 65504
   ceiling → inf → the following RMS norm maps inf → NaN. Fixed with
   `ggml_mul_mat_set_prec(cur, GGML_PREC_F32)`.
6. **`sliding_window=128` was never loaded.** Needed the interleaved-SWA cache:
   `build_attn_inp_kv()` asserts `swa_type == NONE`, so the draft had to move to
   `build_attn_inp_kv_iswa()` with per-layer sub-cache routing on the
   mode-1 KV-inject path.
7. **The Markov head was never applied.** Zero occurrences of `markov` in
   `common/speculative.cpp`. The reference chains each block position's logits
   on the previously sampled token — including position 0, biased by the anchor.
   It is the only channel telling position *i* what was chosen at *i−1*, since
   the whole block decodes in one bidirectional pass from noise embeddings.
   Both mechanisms to do this already existed in the tree and neither was wired.
8. **Wrong RoPE convention.** `LLM_ARCH_DSPARK` was grouped with
   `LLAMA_ROPE_TYPE_NEOX`. The draft is a 3-layer DeepSeek-V4 stack and must be
   `NORM` (interleaved pairs) like `DEEPSEEK4`/`DEEPSEEK4_MTP`. `DFLASH` stays
   NEOX — that backbone genuinely is NeoX.

## Diagnostic signatures worth keeping

**"Acceptance stalls near 2 tokens, position 0 healthy, deeper positions
collapse" = wrong RoPE convention.** The anchor stays correct because its signal
arrives through the injected target KV and the shared lm_head, not through its
own rope. Documented independently by the upstream DSpark PR author on the
`YanissAmz/DeepSeek-V4-Flash-DSpark-draft-GGUF` card; our measured 1.765 mean
accepted length matched it exactly.

**Cumulative acceptance decaying with generation length is an artifact.**
Windowed per-round acceptance was flat (~36% across all 512 tokens: 34.7 / 33.0
/ 41.1 / 38.0 by quartile). Short runs over-weight a few easy early rounds.
Bucket by token position before claiming drift.

**A NaN logit row does not crash — it argmaxes to the same low token id
forever** (8 `&`, 9 `'`, 4 `"`), so it presents as 0% acceptance. `enc_out`
NaN enters once during *prefill* encode, is injected into the draft KV cache,
and poisons every subsequent round for the whole sequence — hence
prompt-dependence: only prompts whose prefix overflows are affected. Use
`LLAMA_DFLASH_PROBE=1`.

## Output is not invariant to decode batch width

Greedy text differs across draft widths (n=1/2/5 each differ from no-spec).
**This is not a speculative-decoding bug.**

Decisive control: with speculation entirely disabled, two byte-identical
concurrent requests at `--temp 0` (`-np 2`, decode width 2) produced different
text from the solo run *and from each other* — solo `b23b7bce`, par-1
`07470531`, par-2 `8fdab550`. `-ub` 2048 vs 512 was stable, because that only
changes *prefill* width; the sensitive axis is **decode** width, which
speculation changes by construction.

Corroborating: a fully broken draft (NaN logits, 0 of 2540 accepted) still
produced correct coherent output, so verification never accepts a token the
target did not produce.

**Rule: output equality cannot gate spec-decoding correctness on this
architecture.** Gate on per-position acceptance and paired A/B of
`draft_n`/`accepted`. Same class as the expert-order finding (permuting the
top-k expert list, a mathematical no-op, flips top-1 on 5.5% of tokens). Within
a single config and one sequence, output *is* bit-reproducible run to run.

## Profile (nsys, `n_max=2`)

`--cuda-graph-trace=node` is **required** or the kernel table comes back empty —
llama.cpp submits through CUDA graphs and nsys defaults to graph granularity.
Two further traps: `nsys launch` rejects `--cpuctxsw`, and `--duration` SIGTERMs
the target unless `--kill=none`.

Splitting MoE expert matvecs by weight type cleanly separates draft from target
(draft is MXFP4, target is IQ2/IQ3/Q6_K):

| | kernels | GPU ms | share |
| --- | --- | --- | --- |
| target experts | 6,477 | 1421.4 | 92.2% |
| draft experts (MXFP4) | 534 | 119.9 | 7.8% |

7.8% matches the 3/43 layer ratio. The draft is cheap, which is why `n=2` pays
for itself at 60% acceptance. At the API level `cudaStreamSynchronize` is 95.8%
of time (16,568 calls) — ordinary memory-bound decode waiting on the GPU.

## Quant quality feeds acceptance

Unsloth's UD-IQ3_XXS shipped with an IQ2_S grid that Unsloth themselves described as *"much
worse than even IQ1_S for quantization, hence the high max KLD yet lower median KLD"*; it was
swapped for IQ2_XS on 2026-07-31. Census, both 1328 tensors: IQ2_S 84 → 0, IQ2_XS 0 → 50,
IQ3_XXS 41 → 75, everything else identical (+1.15 GiB).

Paired measurement, same binary, DSpark n=2 @ P85:

| target build | tg | drafted | accepted | acceptance |
| --- | --- | --- | --- | --- |
| fixed (IQ2_XS) | **24.40 / 24.43** | 420 | 300 | **71.43%** |
| defective (IQ2_S) | 22.38 / 22.27 | 448 | 286 | 63.84% |

**+9.3% tg and +11.9% relative acceptance from the quant fix alone.** The draft is trained
against the real model, so a corrupted expert grid pushes the target's logits away from what the
draft predicts — acceptance is a downstream victim of target quant quality. Re-check acceptance
after any target-quant change; do not carry an acceptance number across builds.

To diff a remote GGUF recipe without downloading it, range-fetch the first ~3 MB of each shard
and parse the tensor table (it precedes the weight data). That identified the IQ2_S → IQ2_XS
swap from HF's live files before the fix was announced.

## Draft GGUF dtypes

`fc` (main_proj) is written **f32** and `conf_proj` **f32**.

`conf_proj` matches the reference, which upcasts deliberately: *"proj in the
checkpoint is stored in bf16, while the parameter here is stored in fp32 for fp32
confidence score"* (`DSparkConfidenceHead`). It is 4352 values, so f32 costs 8.7 KB.

`fc` is about portability, not accuracy. Measured at identical config (n=3, c4096,
paired):

| fc dtype | runtime `PREC_F32` | enc_out | acceptance |
| --- | --- | --- | --- |
| F16 | ON | clean | 0.4927 |
| F32 | ON | clean | 0.4887 |
| BF16 | OFF | clean | 0.4927 |
| **F16** | **OFF** | **nan=449** | **0.0000** |

In *this* fork the dtype is worth 0.4% relative, because `ggml_mul_mat_set_prec`
forces F32 compute regardless of weight dtype. On a **stock** runtime without that
override an F16 `fc` is catastrophic — the encoder overflows, every draft logit is
NaN, and speculative decoding stops accepting entirely (tg falls below no-spec).
f32 costs +100 MB on an 11.35 GiB draft and is safe everywhere.
`LLAMA_DSPARK_NO_FC_F32=1` disables the runtime override to test a checkpoint's own
dtype against a stock runtime.

BF16 is also safe but has 7 mantissa bits to F16's 10; it was measured equivalent
here only because the compute is F32 either way.

## SETTLED: the draft must be built from the 0731 checkpoint's own shards

A rebuilt draft measured 9 points below the original (0.4887 vs 0.5827) and I
first blamed `token_embd`/`output`, then the `fc` dtype. Both wrong. Evidence:

* `token_embd`/`output` are **byte-identical** across the builds -- the
  `--main-gguf` choice changes nothing there (they are copied raw, and the two
  Unsloth builds quantize them the same).
* `fc` dtype is worth **0.4%** relative (0.4927 f16 vs 0.4887 f32, paired).
* Every *module-derived* tensor differed: fc, markov_w1/w2, attn, experts.

The cause is the **`--module-dir` source**. There are two DSpark module copies on
a typical box and they are NOT the same weights:

| source | `mtp.0.main_proj.weight` sha |
| --- | --- |
| **`deepseek-ai/DeepSeek-V4-Flash-0731` shards 46-48** | **`1b405d74`** |
| `deepseek-ai/DeepSeek-V4-Flash-DSpark` (standalone repo) | `b1e6e67a` |
| a module-only local copy of the same standalone repo | `b1e6e67a` |

The standalone `-DSpark` repo is the **pre-0731 preview module**. The 0731
checkpoint ships its own DSpark module in its tail shards (`mtp.0/1/2`, 4705
tensors across shards 46-48), and that is the only correct source for a 0731
draft. Using the preview module costs **9 points of acceptance** (0.5827 ->
0.4887) and looks exactly like a quality regression.

Build the draft with:

    --module-dir ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/<rev>/

Verification that a draft has the right module: `markov_w1` sha `a5eb4c73` and
`blk.2.ffn_down_exps` sha `172b7596` (preview module gives `d7f8b1be` /
`5c260399`). The shipped draft reproduces 556 drafted / 324 accepted = 0.5827
with the runtime override both ON and OFF.

## KV cache dtype: `-ctk q8_0` (needs `-ctv q8_0` too)

llama.cpp rejects mismatched K/V types even though MLA stores no V (every tier
reports `V: 0.00 MiB`), so both flags are required.

Memory at 1M ctx (measured):

| tier | f16 | q8_0 |
| --- | --- | --- |
| CSA compressed | 5376 | **2856** |
| raw SWA window | 96.75 | 51.40 |
| HCA compressed | 160 | 85 |
| indexer (mxfp4) | 357 | 357 (unaffected) |
| **context total** | **6001** | **3361 MiB** |

Compute is unchanged. Net **−2.6 GiB at 1M**.

Quality, DSpark n=3, P85, 2 runs per arm:

| depth | f16 acceptance | q8_0 acceptance | q8_0 tg | q8_0 pp |
| --- | --- | --- | --- | --- |
| 4k | 0.5827 | 0.5192 | −7.3% | — |
| 34k | 0.4943 / 0.4919 | **0.5385 / 0.5385** | **+4.8%** | −4.5% |

**f16 degrades with depth (0.583 -> 0.494) while q8_0 is depth-stable
(0.519 -> 0.539)**, so the two cross over somewhere between 4k and 34k. q8_0 is
bit-deterministic across runs (315/585 twice); f16 wobbles slightly. Unexplained,
but reproducible — note that quantizing `type_k` flips on `attn_rot_k`, enabling
the Hadamard rotation (confirmed `attn_rot_k = 1` under q8_0, `0` under f16), which
exists to handle outliers in quantized caches.

**Recommendation: f16 for short context, q8_0 for long** — at depth it wins on
memory AND acceptance.

## Memory budget and what actually consumes it

Measured with target + draft resident at c40960 ub2048:

| | model | context | compute | total |
| --- | --- | --- | --- | --- |
| draft | 11295 | 13 | **1074** | 12383 MiB |
| target | 98961 | 342 | 1093 | 100396 MiB |

Weights are 107.7 of the 110.1 GiB. **The draft's compute buffer (1074 MiB) is as
large as the target's** because `common/speculative.cpp:2959` builds the draft
context from `common_context_params_to_llama(params)` — it inherits `n_ctx`,
`n_batch` and `n_ubatch` verbatim, and there is no draft-side `-c`/`-ub` flag. The
draft needs the wide ubatch for feature injection (which chunks the prompt by
`n_ubatch`), but a draft-side override would reclaim most of that.

KV scaling is near-O(1) in context: `context ~= 96.75 + ctx x 0.00562 MiB`
(raw window pinned at 2304 cells; only CSA/HCA/indexer grow). 1M costs 6001 MiB of
KV against ~45 GiB for a dense equivalent. **The blocker is the compute buffer**,
1059 -> 4585 MiB from 8k -> 1M at ub2048, or 1144 MiB at ub512.

`earlyoom -m 3,2` left only ~1.8 GiB of working headroom with 110 GiB resident and
killed ~8 measurements. Relaxed to `-m 1,1` (2026-08-01) -> ~4.3 GiB; every
previously-failing leg then completed.

## Deep fills are duty-limited: P70 goes 2.7x deeper than P85

256k ctx + DSpark draft, target `-ub 2048` / `--spec-draft-ubatch 256`, q8_0 KV,
~100k-token fill, watchdog abort at SoC 94 C:

| duty | start SoC | pp | reached | note |
| --- | --- | --- | --- | --- |
| P85 | ~54 C | 364.40 | 28672 (29%) | ramps monotonically through 94 |
| **P70** | 60.7 C | 263.80 | **77824 (79%)** | **2.7x deeper, ~80 s short of finishing** |
| P65 | 63.9 C | 269.64 | 36864 (37%) | INVALID — soaked chassis |

**Mechanism.** At P85 SoC climbs straight through 94 C. At P70 it *oscillates in an
84-92 C band*, shedding heat in the duty gaps — power swinging 27 W → 83 W → 33 W.
The wall stops being a ramp and becomes a hover. Deep fills are limited by sustained
power, so duty cycling is the lever that moves the ceiling.

**Throughput does not buy depth.** `--spec-draft-ubatch` bought +12.9% pp (322.86 →
364.40) and the fill then hit the *same* token count at the *same* temperature, just
11% sooner. Worth having for memory and speed; useless for depth.

**Thermal soak dominates, and it invalidated the P65 leg.** P65 started 3 C warmer on
a chassis still soaked from P70 and reached half the depth; its power trace gives it
away (77.5 W where P70 drew 51.0 W at the same point — backwards for a lower duty).
**Deep-fill legs need a genuinely cold chassis and cannot be run back-to-back**;
8 minutes of cooldown is not enough. P65 remains untested.

**Cold-start rule: idle time predicts deep-fill capacity; SoC die temp does not.**
Four P70 fills, same config:

| idle before | start SoC | prompt | reached | power @t+152s |
| --- | --- | --- | --- | --- |
| ~25 min | 43.7 C | 98k | **96700 (98%)** | 38.5 W |
| ~10 min | 60.7 C | 98k | 77824 (79%) | 51.0 W |
| 9 min | 49.8 C | 85k (shorter) | 43008 (51%) | 69.5 W |
| 8 min | 63.9 C | 98k | 36864 (37%) | 77.5 W |

The 49.8 C / 9-min run reached half the depth of the 60.7 C / 10-min run *with a
shorter prompt*, drawing ~2x the power at the same elapsed time. The die cools in
minutes; the heatsink does not. A `wait until SoC <= 50 C` gate passed in 540 s and
produced the second-worst result — **gate on ~25 min of idle instead**.

Best result: **96700 of ~98500 tokens (98%) at P70 from a genuinely cold box**,
pp 257.61, only ~8 s spent >= 93 C, SoC max 94.4 — and the oscillation stayed healthy
(dipping to 79.9 C at t+482), so the run was not diverging.

**Recipe:** `GGML_CUDA_POWER=70` + `--spec-draft-ubatch 256` + `-ctk/-ctv q8_0` at
`-c 262144` with target `-ub 2048` handles ~78k-token fills from cold with the draft
attached (pp ~264, tg ~24, acceptance ~0.53).

## Context ladder with the draft attached

| context | status |
| --- | --- |
| 34k | comfortable, P85/ub2048, pp 343 / tg 20.2 / acc 0.5385 |
| 256k | ~4 GiB headroom with `--spec-draft-ubatch 256`; deep fills need P70 |
| 512k | loads at ub1024, tg 24.0 / acc 0.5755; memory low-water 2.6 GiB (thin) |
| 1M | target-ONLY (11 GiB headroom, tg ~16). With draft ~1 GiB over budget — only a
smaller target quant closes it |

## Canonical references

- Reference implementation: `inference/model.py` in the HF
  `deepseek-ai/DeepSeek-V4-Flash-DSpark` repo — `DSparkBlock.forward_head` /
  `forward_embed`.
- sglang `python/sglang/srt/models/deepseek_v4_dspark.py`
  (`is_neox_style=False`, `view_as_complex(unflatten(-1,(-1,2)))`),
  `speculative/dspark_components/*`.
- ds4.c adjacent-pair rotation loop (~line 10192).
- SpecForge `tests/test_utils/test_dflash_mask.py` — element-level mask ground
  truth (`anchor_pos` derives from `q_idx / block_size` only).
- Paper: arXiv 2607.05147. **No DeepSeek per-position profile exists for
  `block_size=5` or for V4-Flash at all** — their Figure 2 is Qwen3-4B at γ=7
  and is a bitmap with six endpoint values in prose. Reported accepted length is
  5.57 / 5.12 / 3.49 (math/code/chat) on Qwen3-4B, ceiling γ+1, including the
  bonus token. Every V4-Flash number is relative to MTP-1, never to
  non-speculative decoding.

## Open

- **Confidence head only truncates host-side.** Wired and working (blocks
  truncate to `draft=4`/`2`/`1` at `n_max=5`), but the `p_min` tuning legs were
  killed by `earlyoom` before producing a throughput number. Published results
  suggest the payoff is at concurrency (+13.8% decode at 32, nothing at 1).
- **Block-anchored window is a negative at `n=2`.** Correct against the
  reference but measurably no better; may matter at larger `n`.
- **`fc` is F16 in our converted GGUF.** The runtime `GGML_PREC_F32` fix covers
  it; writing BF16 in `scripts/convert_ds4_mtp_module.py` would match the
  reference directly.
- **`earlyoom` kills `llama-server` mid-run** (SIGTERM ×2 → "Received second
  interrupt"). Killed several nsys captures and sweep legs. Check
  `journalctl | grep earlyoom` before diagnosing a mystery server death.
