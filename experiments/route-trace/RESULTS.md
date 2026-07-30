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

## 2c. Second corpus: the per-layer predictability relation does NOT replicate

Same design on `rtrc-corpus-code.txt` with its own base logits (PPL 2.0105), in-domain
code-distilled table (91.6% instance coverage, so these carry a table-quality penalty relative
to §2 — but a common multiplicative one, which leaves the *ordering* and the *spread* intact).

**The floor on code is exactly zero**: the null leg gives KLD −0.000001 and same top-1
**100.000%**. Wiki's 0.000124 floor is stored-logit precision flipping near-ties; on code the
model is confident enough that nothing flips at all.

| layer | code P | KLD | same top-1 | floor | excess |
| --- | --- | --- | --- | --- | --- |
| L42 | 0.392 | 0.012156 | 98.027% | −0.000001 | 0.0122 |
| L27 | 0.294 | 0.015126 | 97.169% | 0.001128 | 0.0140 |
| L25 | 0.214 | 0.020141 | 97.020% | 0.001539 | 0.0186 |
| L19 | 0.448 | 0.023658 | 96.898% | 0.002625 | 0.0210 |

Damage and excess are both **monotone decreasing with depth** and show no relation to `P` —
the most predictable layer (L19) is the most expensive here, and the second most predictable
(L42) the cheapest. On this corpus depth explains everything and predictability explains
nothing.

What differs: wiki PPL 5.03 against code PPL 2.01. The damage spread collapses from 6× to 2×
and the `P` spread from 0.235–0.570 to 0.214–0.448. The likely reading is that on wiki the
damage is dominated by how many near-ties a layer's perturbation can flip, and `P` correlates
with that; on code there are few near-ties left to flip, so only raw propagation depth
survives.

**So the §2 relation is a wikitext result, not a law.** It was pre-registered and it held there
over a 6× spread, but one corpus of replication is enough to show it does not generalise. Two
things do replicate across both corpora:

- the floor's monotone decay with depth, reaching exactly the null at the last layer;
- **L42 being the cheapest layer to hashify** (lowest on wiki at 0.021900, lowest on code at
  0.012156).

### Why it does not replicate: near-tie density, measured from the base logits

The proposed reason — a routing perturbation can only flip a prediction where the top two are
close — is a claim about the *base* distribution, so it is testable with no GPU at all.
`margin_stats.py` decodes the stored `--kl-divergence-base` file and reports the top-1 margin.

| P(top-1 margin < …) | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 | 1.0 | median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| wiki | 0.33% | 1.70% | 3.33% | 8.48% | 16.53% | 30.08% | 2.06 |
| code | 0.10% | 0.68% | 1.45% | 4.03% | 7.05% | 12.55% | 5.95 |

Near-ties are ~2.3× rarer on code and its median margin is 2.9× wider. Calibrating a margin
threshold on the wiki `roll@L19` flip rate (3.490% → 0.106 logits) and applying it unchanged to
the code margin distribution **predicts a 1.575% flip rate on code against 1.145% observed** —
a zero-free-parameter cross-corpus prediction landing within 1.38×, using the pure-rounding
perturbation that is identical in nature on both corpora.

So the absolute damage scale is set by the corpus, not by the intervention: KLD scales the same
way (0.008723 wiki vs 0.002625 code at the same roll, 3.3×, against the 3.05× flip-rate ratio).
**"Same top-1 %" is not a corpus-independent measure of perturbation severity**, and neither is
KLD — only comparisons *within* one corpus and one base-logits file are meaningful. That alone
explains the magnitude collapse in §2c; whether it also explains the ordering reversal needs a
depth-controlled pair, which is what the adjacent-layer legs test.

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
