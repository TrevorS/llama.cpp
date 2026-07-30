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
| L19 | 0.008723 | 96.510% |
| L27 | 0.002242 | 98.043% |

Fewer layers left for the nudge to grow through. Per-layer damage is therefore **not
comparable across depths without its matched floor**.

## 2. Per-layer sensitivity vs. observational predictability

`P` is the precision of a held-out static token→top-6 table (`rtrc_analyze.py --marginal
--split 0.9`), measured *before* the legs ran. The ordering L19 < L3 < L16 < L27 was
pre-registered.

| layer | P | hashify KLD | same top-1 | ln PPL ratio | matched floor |
| --- | --- | --- | --- | --- | --- |
| L19 | 0.570 | 0.024264 | 94.392% | +0.00272 | 0.008723 |
| L3 | 0.449 | 0.041405 | 92.529% | +0.00129 | 0.018963 |
| L4 | 0.484 | 0.042603 | 92.639% | +0.00713 | — |
| L16 | 0.364 | 0.044486 | 92.251% | +0.01308 | — |
| L27 | 0.251 | 0.149494 | 87.702% | +0.09538 | 0.002242 |

Prediction confirmed, 6.2× spread. Two things to state carefully:

- L3 and L4 are a near-tie (1.7σ). The relation is **flat at the high-P end and explosive at
  the low-P end**, not a clean rank correlation.
- L27 has the *smallest* floor of the three measured and the *largest* damage — 67× its own
  floor, against 2–3× for L3 and L19. Routing becomes genuinely context-dependent with depth;
  early and mid layers sit close to token-determined.

L19's perplexity ratio (+0.002720) is indistinguishable from the null bias (+0.002733): the
most token-predictable learned layer in the model is free to hashify by that measure, and it
is also the top-ranked layer on an unrelated code corpus.

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

- One eval corpus (wikitext). The code corpus is used for table-building and for the
  observational profile, not as an evaluation set.
- Tables are distilled from traces of the *same* text the eval draws from, so instance
  coverage is 100% and the costs here are **lower bounds** on what an out-of-sample table
  would cost. §5's transfer row is the upper bound.
- Quality is one axis. Hash layers also buy lookahead — their experts are known before the
  token is embedded — so a systems answer to "why 3" (enough layers to hide expert fetch
  latency) is not excluded by any of this. This box keeps all experts resident and cannot
  test it.
- KLD ratios and differences are both unprincipled normalisations for a divergence; raw
  damage and its matched floor are reported side by side rather than a derived score.
