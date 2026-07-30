# Static token-table routing in DeepSeek-V4-Flash: interventional results

All numbers from `kld_leg.sh` on DS4-Flash UD-IQ3_XXS, GB10, wikitext `wiki.test.raw`,
`-c 512 -ub 2048 --chunks 100`, KL-divergence against stored base logits
(`~/models/ds4/kld/base-udiq3-c512-n100.kld`), `GGML_CUDA_POWER=80`. Raw logs in
`~/.cache/hashify-kld/`, collected with `kld_table.py`.

The model: 43 layers, all MoE, 256 routed experts + 1 shared, top-6, `expert_weights_norm`,
`expert_weights_scale` 1.5, and `hash_layer_count = 3` — layers 0–2 route by a trained
token→expert lookup (`ffn_gate_tid2eid`) instead of a learned router. The question the whole
study exists to answer is why that number is 3.

## 0. The instrument, and its floor

| leg | condition | Mean KLD | same top-1 | ln(PPL(Q)/PPL(base)) |
| --- | --- | --- | --- | --- |
| null | unmodified model | 0.000124 | 99.961% | +0.00273 ± 0.00071 |
| self | override fed the model's own `tid2eid` | 0.000124 | 99.961% | +0.00273 ± 0.00071 |
| rank | override fed a trace-distilled table | 0.000321 | 99.878% | +0.00287 ± 0.00074 |

`self` reproduces `null` to every printed digit, so the override path is bit-exact and the
harness is deterministic run to run — the 0.000124 floor is the reduced precision of the
stored logits, not noise. `rank` is the end-to-end self-test of the whole
trace → table → runtime path: a table distilled from route traces reproduces the model to
within ~44 tokens of 51200.

Perplexity's own bias against the stored logits is +0.00273, and **that bias is larger than
several of the interventions below.**

## 1. Expert order is not free, and perplexity cannot see it

Permuting the top-k expert list is a mathematical no-op: weights are gathered by id and
normalised by their sum, so only the accumulation order — the rounding — changes.

| leg | permutation | Mean KLD | same top-1 | ln PPL ratio |
| --- | --- | --- | --- | --- |
| perm | reversed | 0.020974 | 94.475% | +0.00170 |
| sortid | sorted by expert id | 0.020999 | 94.624% | +0.00243 |
| rot1 | rotated by one | 0.020782 | 94.792% | +0.00256 |
| roll012 | runtime `ggml_roll`, same layers | 0.021317 | 94.431% | +0.00338 |

Three unrelated permutations and an independent runtime mechanism all land on the same
number — permutation-generic, as floating-point non-associativity predicts. **5.4% of top-1
predictions change**, and every one of these legs has a perplexity ratio *below* the null bias.

`sortid` is the one that matters in practice: sorting the expert list by id is exactly what
grouped-GEMM and tile-packing dispatch do.

### The same magnitude arrives from three unrelated causes, and PPL sees none of them

The interesting question is not whether a synthetic permutation matters — it is whether things
people actually do land in the same place. Three do, measured on the same base logits:

| perturbation | what it is | Mean KLD | same top-1 | ln PPL ratio |
| --- | --- | --- | --- | --- |
| null | nothing | 0.000124 | 99.961% | +0.00273 |
| `LLAMA_DSV4_UNION_GATHER=0` | shipped kernel, off | 0.000124 | 99.961% | +0.00273 |
| `LLAMA_DSV4_LID_SHORTCUT=0` | shipped kernel, off | 0.000124 | 99.961% | +0.00273 |
| roll expert list @ L3 | mathematical no-op | 0.018963 | 94.761% | +0.00232 |
| **`LLAMA_DSV4_MOE_GATE_FUSE=0`** | **shipped kernel, off** | **0.018895** | **94.718%** | **+0.00110** |
| sort expert list by id @ L0–2 | mathematical no-op | 0.020999 | 94.624% | +0.00243 |
| **`-ub 2048` → `-ub 512`** | **serving parameter** | **0.021657** | **94.510%** | **+0.00246** |

Two of our three shipped kernels are **bit-exact** — they reproduce the stored logits to every
printed digit, and this instrument proves it in about two minutes. The third, the fused
sqrt-softplus MoE gate, is not: it differs from the unfused reference on **5.3% of top-1
predictions**. All three shipped as "no PPL regression", and on the perplexity ratio the
inexact one looks *better* than the null.

Changing the micro-batch from 2048 to 512 — no code change, a scheduling decision a server
makes under load — moves **5.5%** of top-1 predictions. That is the same magnitude as
deliberately scrambling the expert order.

So a mathematical no-op, a kernel fusion, and a batch-size knob all land within 15% of each
other in divergence, and perplexity separates none of them from the null. The practical rule
is the whole point of this section: **validate MoE changes with same-top-1 against stored
logits, not with perplexity.** It costs one 2-minute leg and it distinguishes a genuinely exact
kernel from a merely PPL-neutral one, which perplexity provably cannot.

### The effect is one-shot

| layers permuted | Mean KLD | same top-1 |
| --- | --- | --- |
| L0 | 0.020993 | 94.745% |
| L0–1 | 0.021002 | 94.553% |
| L0–2 | 0.020974 | 94.475% |

One reordered layer does the same damage as three. Rounding decorrelates once and does not
compound — unlike routing content, which does (§3).

### The floor falls steeply with depth

`LLAMA_MOE_ORDER_ROLL` rotates the *live* routing, so the floor can be measured at any layer:

| roll at | Mean KLD | same top-1 |
| --- | --- | --- |
| L0–2 | 0.021317 | 94.431% |
| L3 | 0.018963 | 94.761% |
| L10 | 0.013801 | 95.533% |
| L16 | 0.010165 | 96.263% |
| L19 | 0.008723 | 96.510% |
| L27 | 0.002242 | 98.043% |
| L35 | 0.000950 | 98.941% |
| L39 | 0.000538 | 99.220% |
| L42 | 0.000124 | 99.961% |

Monotone across nine depths and decaying to exactly the null floor at the last layer, where
there is nothing downstream for the nudge to grow through. Per-layer damage is therefore **not
comparable across depths without its matched floor**.

### Not a CUDA artifact either: the effect reproduces on CPU

The obvious objection to everything above is that it measures a quirk of one GPU backend's
kernels — tile packing, MMQ, graph capture. So the same roll was run on gpt-oss-20b with
`-ngl 0`: no CUDA at all, 20 CPU threads, the llamafile/REPACK paths, F32 accumulation,
`-ub 512`, 20 chunks, against a CPU-generated base.

| backend | Mean KLD | RMS Δp | same top-1 |
| --- | --- | --- | --- |
| CUDA (`-ngl 999`, ub 2048) | 0.119476 | 5.647% | 74.910% |
| **CPU (`-ngl 0`, ub 512)** | **0.115854** | **4.120%** | **71.784%** |

28.2% of top-1 predictions move on CPU against 25.1% on GPU. Two implementations sharing no
kernel code produce the same phenomenon at the same scale, so it is a property of the
arithmetic rather than of any backend.

What this does *not* establish: precision attenuation. The CPU path already accumulates in F32
and there is no F64 knob to sweep, so the FP-associativity mechanism remains inferred — from
permutation-genericity, one-shot behaviour, monotone depth decay, and now backend independence
— rather than demonstrated by making the effect shrink.

### It is not a DeepSeek quirk: three architectures, three vendors, same structure

`LLAMA_MOE_ORDER_ROLL` lives in the shared `build_moe_ffn` path, so it runs on any MoE with no
model-specific code.

| model | arch | experts | layers | wiki PPL |
| --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash | `deepseek4` | 256, top-6, IQ3_XXS | 43, all MoE | 5.03 |
| gpt-oss-20b | `gpt-oss` | 32, top-4, MXFP4 | 24, all MoE | 195 |
| Nemotron-3-Nano-30B-A3B | `nemotron_h_moe` | 128, top-6, Q8_0 | 52, 23 MoE (hybrid SSM) | 8.25 |

Rolling one layer's expert list, same top-1 agreement against each model's own stored logits:

| model | null | first MoE layer | mid | late | last MoE layer | all layers |
| --- | --- | --- | --- | --- | --- | --- |
| DS4-Flash | 99.961% | 94.745% (L0) | 96.510% (L19) | 98.941% (L35) | 99.961% (L42) | — |
| gpt-oss-20b | 99.996% | **75.208%** (L0) | 81.992% (L12) | — | 99.996% (L23) | 74.910% |
| Nemotron-30B | 99.992% | 97.137% (L1) | 98.235% (L20) | 98.992% (L38) | 99.992% (L51) | 96.996% |

Every structural feature holds on all three:

- **the effect exists** — 3.0%, 5.3% and 24.8% of top-1 predictions move;
- **it is one-shot** — rolling *every* layer equals rolling just the first (gpt-oss 74.910% vs
  75.208%; Nemotron 96.996% vs 97.137%);
- **it decays monotonically with depth to exactly the null at the last layer** — Nemotron runs
  0.004184 → 0.001387 → 0.000363 → 0.000002 → 0.000000 across L1/20/38/49/51;
- **perplexity is blind** — every ln-ratio sits at or below the model's own null bias, several
  are negative.

Nemotron also gives a free negative control: layers 4 and 26 are *not* MoE blocks in that hybrid
stack, and rolling them returns exactly the null, confirming the hook only perturbs MoE routing.

Effect size does *not* transfer between models — 3% to 25% — and near-tie density explains only
part of that (§2c). Structure transfers; magnitude does not.

### Closing the loop: the benchmark cannot see it either, and the arithmetic says why

Perplexity being blind invites the obvious retort — *use a benchmark instead*. HellaSwag, 2000
tasks, two models, with and without the expert-list roll (still a mathematical no-op).
llama.cpp prints a running accuracy per task, so differencing the correct-count recovers each
individual answer; both reconstructions had zero anomalies (every difference exactly 0 or 1).

| model | top-1 flips | headline accuracy | answers changed | right→wrong | wrong→right | net |
| --- | --- | --- | --- | --- | --- | --- |
| gpt-oss-20b | 24.8% | 58.05% → 57.70% | **139/2000 = 6.95%** | 73 | 66 | −0.35 pp |
| Nemotron-30B | 3.0% | 71.10% → 71.00% | **20/2000 = 1.00%** | 11 | 9 | −0.10 pp |

**The score is stable because the flips are symmetric, not because the model behaves the same.**
Both models change roughly a third of their top-1 flip rate into changed benchmark answers
(6.95/24.8 = 0.28; 1.00/3.0 = 0.33), and in both the right→wrong and wrong→right counts are
balanced, so the net is a random walk: |net| of 7 against √139 ≈ 12 expected, and 2 against
√20 ≈ 4.5. The headline moves by about a seventh of the per-item churn, in the direction chance
happened to pick.

That is the mechanism behind benchmark blindness, and it is unfixable by scaling: detecting a
net shift that is a random walk over `f·n` flipped items needs `n` growing as 1/f², while the
*per-item* disagreement is visible at any `n`. A benchmark can detect this only if it reports
per-item agreement against a reference run — which is the same measurement as same-top-1, on a
smaller and less sensitive sample.

Put end to end on one perturbation that is provably a no-op:

| observable | change |
| --- | --- |
| perplexity ratio | at or below the null's own bias |
| HellaSwag headline accuracy | −0.35 pp / −0.10 pp (both inside CI) |
| HellaSwag individual answers | 6.95% / 1.00% changed |
| top-1 predictions | 24.8% / 3.0% changed |
| greedy generation | diverges mid-derivation |

The standard validation stack — perplexity plus a benchmark — is blind to a change that alters
a quarter of a model's token-level decisions and 7% of its benchmark answers. Same-top-1
against stored logits detects it in one 2-minute leg. That is the entire recommendation.

A caveat worth stating in both directions: flat benchmark accuracy is *reassuring* about
quality — the model is no worse — and *alarming* about validation, because it means the
benchmark cannot distinguish a bit-exact kernel from an inexact one. In agentic or
long-generation settings, where token-level divergence compounds rather than cancelling, the
reassurance is weaker than 0.35 pp suggests.

*(DS4-Flash could not be measured here: `--hellaswag` aborts on our fork with "DSV4 coupled raw
writes require equal sequence lengths" — the fused attention path requires equal sequence
lengths and the multiple-choice harness batches ragged ones. That is a real limitation of our
tree, and it means the standard llama.cpp task harnesses have never been run against it.)*

## 2. Per-layer sensitivity vs. observational predictability

`P` is the precision of a held-out static token→top-6 table (`rtrc_analyze.py --marginal
--split 0.9`), measured *before* the legs ran. The ordering L19 < L3 < L16 < L27 was
pre-registered.

| layer | P | hashify KLD | same top-1 | ln PPL ratio | floor | excess | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L19 | 0.570 | 0.024264 | 94.392% | +0.00272 | 0.008723 | 0.0155 | 2.8× |
| L35 | 0.460 | 0.029576 | 94.875% | +0.01153 | 0.000950 | 0.0286 | 31× |
| L3 | 0.449 | 0.041405 | 92.529% | +0.00129 | 0.018963 | 0.0224 | 2.2× |
| L4 | 0.484 | 0.042603 | 92.639% | +0.00713 | — | — | — |
| L10 | 0.484 | 0.043719 | 92.278% | +0.00804 | 0.013801 | 0.0299 | 3.2× |
| L16 | 0.364 | 0.044486 | 92.251% | +0.01308 | 0.010165 | 0.0343 | 4.4× |
| L27 | 0.251 | 0.149494 | 87.702% | +0.09538 | 0.002242 | 0.1473 | 67× |

The pre-registered ordering held, over a 6.2× spread in raw damage. Three things to state
carefully:

- L3, L4 and L10 are within 5% of each other. The relation is **flat at the high-P end and
  explosive at the low-P end**, not a clean rank correlation (Spearman ρ ≈ 0.71 on raw damage
  across seven layers, ≈ 0.81 on floor-excess).
- L27 has a *smaller* floor than L3 and 3.6× the damage — 67× its own floor, against 2–3× for
  L3 and L19. Routing becomes genuinely context-dependent with depth.
- **The two normalisations disagree, and L35 is where they break.** Its absolute damage is
  near the bottom (0.0296) but its floor is 20× smaller than L3's, so by ratio it looks like
  L27. A multiplicative model — damage = propagation gain × intrinsic routing difference —
  would justify the ratio, but an infinitesimal rounding nudge and an O(1) routing change do
  not have to propagate with the same gain, so that model is an assumption, not a measurement.
  Report raw damage (what deployment feels) alongside the floor (a separate fact about
  perturbation propagation), and do not derive a single score from them.

L19's perplexity ratio (+0.002720) is indistinguishable from the null bias (+0.002733): the
most token-predictable learned layer in the model is free to hashify by that measure, and it
is also the top-ranked layer on an unrelated code corpus.

## 2b. Both ends of the stack are token-driven, and the top is cheapest

The original wiki trace's L37–42 were corrupt (collected before the CUDA-graph fix), so a
fresh 14-chunk leg was collected. Held-out predictability at the deep end:

L34 0.228 · L35 0.462 · L36 0.310 · L37 0.319 · L38 0.308 · **L39 0.485** · L40 0.409 ·
L41 0.400 · **L42 0.497**

L42 and L39 rank behind only L19 (0.570) across the whole model, and the code corpus shows the
same rise at the top. Predictability is **U-shaped in depth** — both ends route by token
identity, the middle does not.

Interventionally (this table is distilled from 28674 rows, ~93% instance coverage, so `d19` is
run with the *same* table to calibrate the table-quality penalty):

| leg | layer | P | table | Mean KLD | same top-1 | ln PPL ratio | floor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| d19 | L19 | 0.570 | deep | 0.034049 | 93.624% | +0.00850 | 0.008723 |
| d42 | L42 | 0.497 | deep | **0.021900** | 95.573% | +0.01886 | 0.000124 |
| d39 | L39 | 0.485 | deep | 0.029992 | 95.090% | +0.01907 | 0.000538 |

The calibration lands at 0.034049 / 0.024264 = **1.40×**, matching the 1.39× penalty implied
independently by the row-matched control in §5. With table quality thus held fixed, **L42 and
L39 are cheaper to hashify than L19**, which was the cheapest mid-stack layer — so the top of
the stack is the cheapest place in the model to replace a router with a token table.

Two observations that follow:

- **At the top, damage shows up in perplexity; mid-stack it hides.** L19's perplexity ratio
  sits exactly on the null bias while L42's is 10σ above it, even though L42's KLD is *lower*.
  A late-layer change lands directly on the logits instead of diffusing through the remaining
  stack.
- **A token-table layer is prefetchable at any depth** — its expert ids depend only on the
  input token, which is known before the forward pass begins. So the lookahead argument for
  hash layers does not explain putting them at the *bottom*, where they are more expensive
  than at the top. If there is a reason for the bottom, it is more likely training dynamics
  than inference.

## 2c. Second corpus: two confounds, and the relation survives both

The first attempt on `rtrc-corpus-code.txt` (own base logits, PPL 2.0105) said the §2 relation
did not replicate — damage was monotone in depth and, if anything, *anti*-correlated with `P`.
That conclusion was wrong, and finding out why produced the two controls this measurement
actually needs.

**Confound A — the layer set confounded predictability with depth.** L19/25/27/42 spans 23
layers, over which the matched floor falls from 0.002625 to exactly zero, while `P` varies only
2×. Propagation swamps predictability, so that comparison could never have shown one.

**Confound B — table coverage dilutes the contrast.** The code table had 91.6% instance
coverage, so ~8% of positions took the generic fallback *at every layer* — a common damage term
that compresses any per-layer difference toward 1. A longer trace (`codefull`, 169 evals /
86018 rows) rebuilt it at **99.88%**, matching the wiki setup.

**Control: adjacent layers, matched floors, matched coverage.** Two predictions were written
down before these legs returned.

Prediction 1 — L20 is half as predictable as L19 at the same depth (`P` 0.220 vs 0.447), so it
should cost *more*. The 91.6%-coverage run had it backwards:

| pair | table coverage | L19 | L20 | ratio |
| --- | --- | --- | --- | --- |
| c19 / c20 | 91.6% | 0.023658 | 0.020829 | 0.88 ✗ |
| cf19 / cf20 | 99.88% | 0.013982 | **0.017095** | **1.22 ✓** |

Fixing coverage flips the sign, and both damages drop — the fallback term was large and shared.

Prediction 2 — the strongest design available. Take the *same two layers* on both corpora. On
wiki L34/L35 sit at `P` 0.276 vs 0.460; on code they are equal (0.318 vs 0.325). So the same
physical pair should differ on wiki and not on code:

| corpus | L34 `P` | L35 `P` | L34 KLD | L35 KLD | ratio | floors |
| --- | --- | --- | --- | --- | --- | --- |
| wiki | 0.276 | 0.460 | 0.062967 | 0.029576 | **2.13×** | 0.001041 / 0.000950 |
| code | 0.318 | 0.325 | 0.013091 | 0.013041 | **1.004×** | 0.000618 / 0.000505 |

Identical to 0.4% on code — inside the error bars — and 2.13× apart on wiki. Same layers, same
weights, matched floors; only the text differs, and the damage follows each corpus's own
observational profile in both directions.

### The code layer set, re-run at 99.88% coverage

The original §2c numbers are superseded — they used the 91.6% table. Predictors are held-out
`P` on the same `codefull` trace the table came from (90/10, 3163 test tokens):

| layer | `P` | KLD | same top-1 | floor | excess |
| --- | --- | --- | --- | --- | --- |
| L42 | 0.399 | 0.008178 | 98.294% | −0.000001 | 0.00818 |
| L27 | 0.341 | 0.010312 | 97.647% | 0.001128 | 0.00918 |
| L19 | 0.447 | 0.013982 | 97.392% | 0.002625 | 0.01136 |
| L34 | 0.318 | 0.013091 | 97.549% | 0.000618 | 0.01247 |
| L35 | 0.325 | 0.013041 | 97.608% | 0.000505 | 0.01254 |
| L25 | 0.132 | 0.015284 | 97.322% | 0.001539 | 0.01375 |
| L20 | 0.220 | 0.017095 | 96.902% | 0.002492 | 0.01460 |

Spearman ρ between floor-excess and (1 − `P`) is **0.82** across these seven layers, against
0.81 for the wiki set on the same statistic. The relation replicates once coverage is fixed;
the earlier null came from the diluted table.

L42 is again the cheapest layer in the model — third independent confirmation, now on both
corpora and under two different tables.

### The magnitude scale is a property of the corpus, not the intervention

Measured directly from the stored base logits with `margin_stats.py`, no GPU required:

| P(top-1 margin < …) | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 | 1.0 | median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| wiki | 0.33% | 1.70% | 3.33% | 8.48% | 16.53% | 30.08% | 2.06 |
| code | 0.10% | 0.68% | 1.45% | 4.03% | 7.05% | 12.55% | 5.95 |

Near-ties are 2.3× rarer on code and the median margin is 2.9× wider. Calibrate a margin
threshold on wiki's `roll@L19` flip rate (3.490% → 0.106 logits) and apply it unchanged to the
code distribution: **predicted 1.575% flip rate, observed 1.145%** — zero free parameters,
within 1.38×, using the pure-rounding perturbation that is identical on both corpora.

So neither same-top-1 nor KLD is a corpus-independent measure of severity, and cross-corpus
*magnitudes* mean nothing. Within-corpus, layer-relative comparisons are the only valid ones —
and those track predictability once depth and coverage are matched.

**Net:** the observational proxy predicts interventional damage, conditional on two controls
that are easy to miss. Reported without them it produces a null (Confound A), a sign flip
(Confound B), or an unreplicable magnitude (near-tie density). The §2 table is a wiki-only
ranking for exactly that reason: its layers span 24 depths.

## 2d. The adjacent-pair design — this supersedes §2 as the primary evidence

§2 ranks layers spanning 24 depths, so it is depth-confounded by construction. The clean design
compares **adjacent** layers, where the propagation floor is matched to within a few percent,
and asks whether the damage ratio tracks the predictability ratio. Ratios are written with the
*less* predictable layer on top, so the prediction is always ≥ 1.

| pair | `P` gap | predicted (1−`P`) ratio | observed |
| --- | --- | --- | --- |
| L18 / L19 | 0.231 | 1.54 | 1.68 |
| L36 / L35 | 0.225 | 1.42 | 1.59 |
| L34 / L35 | 0.184 | 1.34 | **2.13** |
| L11 / L10 | 0.178 | 1.34 | 1.25 |
| L24 / L25 | 0.001 | 1.00 | 1.05 |
| L33 / L34 | 0.002 | 1.00 | 0.99 |
| L07 / L08 | 0.006 | 1.01 | 0.86 |

**All four large-gap pairs exceed all three zero-gap pairs** — complete separation, exact
one-sided Mann–Whitney p = 0.029 at n = 7. Spearman between predicted and observed ratio across
all seven pairs is 0.64, dragged down mainly by L34/L35 overshooting its prediction (2.13 vs
1.34).

The zero-gap pairs are the honest error bar on any per-layer claim, and they are **not**
uniformly 1.0: {0.99, 1.05, 1.16}. Two are tight; L07/L08 is 16% apart at ~6σ despite a 1.6%
predictability difference. A floor mismatch does not explain it — adjacent floors drift ~4% per
layer in the shallow stack, under 1% of that pair's damage. So there is genuine per-layer
structure that predictability does not capture, and it appears larger near the bottom of the
stack, which is exactly where the shipped hash layers are.

Practical reading: an effect below ~1.16 on a single adjacent pair is not evidence. Three of the
four test pairs clear that comfortably; L11/L10 at 1.25 does not, and should be reported as
marginal rather than as a fourth confirmation.

## 3. Cumulative curve

| n | layers | Mean KLD | same top-1 | ln PPL ratio |
| --- | --- | --- | --- | --- |
| 0 | — | 0.000124 | 99.961% | +0.00273 |
| 1 | 3 | 0.041405 | 92.529% | +0.00129 |
| 2 | 3–4 | 0.050483 | 91.953% | +0.00567 |
| 3 | 3–5 | 0.063476 | 91.114% | +0.01008 |
| 4 | 3–6 | 0.072632 | 90.220% | +0.01397 |
| 6 | 3–8 | 0.106640 | 88.533% | +0.02910 |
| 8 | 3–10 | 0.128831 | 87.349% | +0.04553 |
| 16 | 3–18 | 0.278606 | 81.051% | +0.14255 |

The first layer costs 0.041 and each further layer only ~0.010–0.014, so the cumulative curve
saturates and is not a per-layer sensitivity measure. Perplexity only clears its own bias at
n = 4 — which is where the independent c2048 perplexity curve put its knee.

## 4. Which layers to pick

| set | layers | Mean KLD | same top-1 |
| --- | --- | --- | --- |
| sel3 | 4,10,19 (highest P) | 0.064988 | 91.020% |
| cont3 | 3–5 | 0.063476 | 91.114% |
| bad3 | 24,27,32 (low P) | 0.281282 | 82.984% |
| sel6 | 3,4,10,19,20,35 | 0.106228 | 88.376% |
| cont6 | 3–8 | 0.106640 | 88.533% |

Oracle selection buys **nothing** over the naive contiguous choice at either size, but a
low-predictability set is 4.4× worse. The rule is a threshold, not a gradient: avoid the bad
layers; above the threshold the good ones are interchangeable. "Hash the first N layers" works
because the shallow layers happen to sit in the high-P pool.

Three badly-chosen layers (0.281) do about the same damage as sixteen well-chosen ones (0.279).

## 5. What the table needs to know (n = 4, layers 3–6)

| condition | Mean KLD | same top-1 | ln PPL ratio |
| --- | --- | --- | --- |
| learned router (null) | 0.000124 | 99.961% | +0.00273 |
| per-token, wiki, full table | 0.072632 | 90.220% | +0.01397 |
| per-token, wiki, row-matched to the code trace | 0.100582 | 89.024% | +0.03426 |
| constant — every token gets the layer's modal set | 0.296504 | 81.039% | +0.16003 |
| uniform random | 0.353879 | 79.047% | +0.20139 |
| per-token, **code**-distilled, evaluated on wiki | 0.379683 | 78.286% | +0.21191 |

Token identity is worth ~4×. And a table does not transfer: the code-distilled table is
**worse than uniform random routing**. The row-matched wiki control settles that this is
domain and not data volume — same number of traced rows, 3.8× less damage. Systematic error
beats random error, because random routing spreads across all 256 experts and partially
averages out while an out-of-domain table concentrates confidently on a narrow wrong set.

## Caveats

- Two eval corpora, and they disagree about §2 — see §2c. The per-layer predictability
  relation is reported as a wikitext result, not a general one.
- Tables are distilled from traces of the *same* text the eval draws from, so instance
  coverage is 100% and the costs here are **lower bounds** on what an out-of-sample table
  would cost. §5's transfer row is the upper bound.
- Quality is one axis. Hash layers also buy lookahead — their experts are known before the
  token is embedded — so a systems answer to "why 3" (enough layers to hide expert fetch
  latency) is not excluded by any of this. This box keeps all experts resident and cannot
  test it.
- KLD ratios and differences are both unprincipled normalisations for a divergence; raw
  damage and its matched floor are reported side by side rather than a derived score.
