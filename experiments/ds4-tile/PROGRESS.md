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
