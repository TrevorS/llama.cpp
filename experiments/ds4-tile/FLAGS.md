# DSV4 optimization flag inventory

Single source of truth for every DSV4 env gate: default, what ON/OFF does,
where it's gated, and validation status. Update this file in the SAME commit
as any default change or new flag. (Motivation: HC_FUSED sat silently off for
a full bench campaign; a missing FUSED_LID cost two crashed d65536 cycles.)

Last audit: 2026-07-14 (commit 210a22df3 era).

## Serving/bench-relevant flags

| Flag | Default | ON does | OFF does | Notes / validation |
| --- | --- | --- | --- | --- |
| `LLAMA_DSV4_FUSED_LID` | **OFF** (`=1` enables) | fused score+top-k op; O(chunk×nt) working set | unfused 8-op chain; O(n_ctx×nt×n_head) relu intermediate | **MANDATORY at depth** — unfused OOMs ≥ d65536 ub2048 (~8.6 GB relu). Token-identical (index-set + greedy gates). DEFAULT-ON CANDIDATE. |
| `LLAMA_DSV4_FUSED_LID_TG_DEPTH` | 4096 | — | — | decode (nt=1) uses unfused below this n_lid (faster at bs1 shallow); 0 = always fuse, huge = never |
| `LLAMA_DSV4_HC_FUSED` | **ON** (`=0` disables, since 2026-07-14) | fused HC weighted_sum/post/sinkhorn kernels | scalar per-stream/unrolled chains (~16k-node decode graphs) | weighted_sum/post token-identical (gate 2); sinkhorn mode 21183→5439 nodes. Campaign refs always ran ON. |
| `LLAMA_DSV4_CSA_TILE` | **ON, W=16** (`=0` disables) | B2 per-tile union-gather CSA attention (prefill) | dense masked CSA FA | all gates passed (PPL, passkey 5/5, determinism). +12.2% pp@d65k → +61.9% @d262k IQ3. Self-gated: nt_s>W, n_stream==1, nt_s%W==0, n_csa≥TILE_MIN, 256-alignment |
| `LLAMA_DSV4_CSA_TILE_UCAP` | 4096 | — | — | per-tile union cap; keep n_raw+u_cap 256-aligned; tail few % tiles truncate at d65k+ (passkey/PPL clean) |
| `LLAMA_DSV4_CSA_TILE_MIN` | 12288 | — | — | min n_csa to activate (~d49k); below it tiled FA loses (12 vs 41 TFLOPS small-nb latency wall) |
| `LLAMA_DSV4_CSA_GATHER` | **OFF** (`=1` enables) | B1 decode gather: attend raw window + 512 selected rows only | dense masked CSA FA at decode | +8.3% tg@d65k, grows with depth; decode FA share 2.5% @512k with it ON. First ~25 greedy tokens identical then fp-reassociation. DEFAULT-ON CANDIDATE. |
| `LLAMA_DSV4_LID_INT8` | **OFF** (`=1` enables) | int8 dp4a score kernel (prefill/batch) | wmma/fp16 score path | 1.36x kernel, +3.2% pp; 0.5% score error, PPL-neutral. Selection-set class. DEFAULT-ON CANDIDATE. |
| `LLAMA_DSV4_LID_DEC` | **OFF** (`=1` enables) | dedicated nt=1 decode score kernel (warp-per-comp, int8) | 16-token-tile kernel with 15/16 padding waste at decode | 2.0x kernel, +5.2% tg (+10.8% with GATHER). Same int8 numerics class. DEFAULT-ON CANDIDATE. |
| `LLAMA_DSV4_LID_FP4` | OFF | e2m1 block-32 QAT fake-quant of indexer q/k (numerics only, no speedup) | fp16/fp32 indexer numerics | the model's official QAT indexer numeric; 0.93 top-512 overlap, deep PPL statistically identical. Keep opt-in until fp4 STORAGE lands. |
| `LLAMA_DSV4_MOE_TILE` | OFF | MoE expert-tile bridge kernels in the graph | standard MoE path | NEGATIVE RESULT (tile bridge regresses; hc-batch memory). Leave OFF. |

## Debug/instrumentation flags (never in perf runs)

| Flag | Purpose |
| --- | --- |
| `LLAMA_DSV4_UNION_STATS=1` | exact per-tile union sizes (full-bitmap popcount) printed per call; syncs every union op |
| `LLAMA_DSV4_LID_DUMP` / `_NLID` | one-shot score-input dump for the fp4 oracle |
| `LLAMA_DSV4_COMPRESS_DEBUG` | compressor debug prints |
| `GGML_SCHED_DEBUG=2` + `-lv 9` | full graph/split dump (node-count attribution) |

## Retired / superseded

- `LLAMA_DSV4_CSA_UNION`, `LLAMA_DSV4_CSA_UNION_CAP`, `LLAMA_DSV4_CSA_UNION_FULL`
  — whole-batch union path, replaced by `LLAMA_DSV4_CSA_TILE` (iter 18d/19).
- `LLAMA_DSV4_LID_INT4` — negative result (compute-bound on nibble→int8 expansion, 1.6x slower than int8), reverted (iter 15).

## Canonical bench/serving env (until the default-on candidates land)

```
LLAMA_DSV4_FUSED_LID=1 LLAMA_DSV4_CSA_GATHER=1 LLAMA_DSV4_LID_DEC=1 LLAMA_DSV4_LID_INT8=1
```
(tile + HC-fused already default-on; add `LLAMA_DSV4_LID_FP4=1` for QAT-numerics runs)
