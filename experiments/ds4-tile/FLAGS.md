# DS4 flag inventory (everything we introduced on this branch)

Single source of truth for every env gate and CLI flag added since the
upstream base (e3546c794): default, what ON/OFF does, where it's gated, and
validation status. Update this file in the SAME commit as any default change
or new flag. (Motivation: HC_FUSED sat silently off for a full bench
campaign; a missing FUSED_LID cost two crashed d65536 cycles.)

Audit method: `git diff e3546c794..HEAD | grep getenv` + added `add_opt`
CLI args. `LLAMA_DSV4_COMPRESS_DEBUG` is upstream (#24162), not ours.
`GGML_NO_BACKTRACE`/`GGML_BACKTRACE_LLDB` are upstream debug knobs.

Last audit: 2026-07-14 (commit 210a22df3 era).

## Serving/bench-relevant flags

| Flag | Default | ON does | OFF does | Notes / validation |
| --- | --- | --- | --- | --- |
| `LLAMA_DSV4_FUSED_LID` | **ON** (`=0` disables, since 2026-07-14 P4a) | fused score+top-k op; O(chunk×nt) working set | unfused 8-op chain; O(n_ctx×nt×n_head) relu intermediate | **MANDATORY at depth** — unfused OOMs ≥ d65536 ub2048 (~8.6 GB relu). Token-identical (index-set + greedy gates). |
| `LLAMA_DSV4_FUSED_LID_TG_DEPTH` | 4096 | — | — | decode (nt=1) uses unfused below this n_lid (faster at bs1 shallow); 0 = always fuse, huge = never |
| `LLAMA_DSV4_HC_FUSED` | **ON** (`=0` disables, since 2026-07-14) | fused HC weighted_sum/post/sinkhorn kernels | scalar per-stream/unrolled chains (~16k-node decode graphs) | weighted_sum/post token-identical (gate 2); sinkhorn mode 21183→5439 nodes. Campaign refs always ran ON. |
| `LLAMA_DSV4_CSA_TILE` | **ON, W=16** (`=0` disables) | B2 per-tile union-gather CSA attention (prefill) | dense masked CSA FA | all gates passed (PPL, passkey 5/5, determinism). +12.2% pp@d65k → +61.9% @d262k IQ3. Self-gated: nt_s>W, n_stream==1, nt_s%W==0, n_csa≥TILE_MIN, 256-alignment |
| `LLAMA_DSV4_FA_SPLIT` | **ON** (`=0` disables) | tile branch runs split attention: dense raw-window FA (ne3=1, full width) + union-only tiled FA, both `ggml_flash_attn_ext_with_lse`, merged by `GGML_OP_DSV4_FA_MERGE`; no `repeat_4d` raw replication | single concat FA per tile (raw window physically replicated T×) | math-identical softmax over the same KV union (class C fp-reassociation); sink lives in the raw half's LSE. Only active inside the CSA_TILE branch with `-fa on` and T ≤ hd. Gates PASSED 2026-07-14: det, A/B byte-identical @42k, passkey 5/5, PPL within noise; +7.1% pp@d65k / +6.1% pp@d131k (new records), tg neutral. See PROGRESS iter 27. |
| `LLAMA_DSV4_CSA_TILE_UCAP` | 4096 | — | — | per-tile union cap; keep n_raw+u_cap 256-aligned; tail few % tiles truncate at d65k+ (passkey/PPL clean) |
| `LLAMA_DSV4_CSA_TILE_MIN` | 12288 | — | — | min n_csa to activate (~d49k); below it tiled FA loses (12 vs 41 TFLOPS small-nb latency wall) |
| `LLAMA_DSV4_CSA_GATHER` | **ON** (`=0` disables, since 2026-07-14 P4a) | B1 decode gather: attend raw window + 512 selected rows only | dense masked CSA FA at decode | +8.3% tg@d65k, grows with depth; decode FA share 2.5% @512k with it ON. First ~25 greedy tokens identical then fp-reassociation. |
| `LLAMA_DSV4_LID_INT8` | **ON** (`=0` disables, since 2026-07-14 P4a) | int8 dp4a score kernel (prefill/batch) | wmma/fp16 score path | 1.36x kernel, +3.2% pp; 0.5% score error, PPL-neutral. Selection-set class B (pass-1 only when LID_EXACT). |
| `LLAMA_DSV4_LID_RADIX` | **ON** (`=0` reverts to bitonic) | radix top-k on the f16 scores buffer: 2x8-bit MSB histogram threshold, ordered-scan compaction (all >T + lowest-index fill at ==T), one small bitonic on the top_k selected — one kernel/token replaces the chunk+merge tree | bitonic chunk+merge tree (SORT_N 4096) | Selection bit-identical to bitonic (-0 canonicalized so key-eq == f32-value-eq; ordered compaction, not atomic-append). Active only for half scores (d128), nt>=16, n_lid>top_k; d64/f32 + decode stay bitonic. Expected topk 50.4 to ~5ms @prod shape (op -37%). See PROGRESS iter 34/35. |
| `LLAMA_DSV4_LID_DEC` | **ON** (`=0` disables, since 2026-07-14 P4a) | dedicated nt=1 decode score kernel (warp-per-comp, int8) | 16-token-tile kernel with 15/16 padding waste at decode | 2.0x kernel, +5.2% tg (+10.8% with GATHER). Same int8 numerics class. |
| `LLAMA_DSV4_LID_FP4` | OFF | e2m1 block-32 QAT fake-quant of indexer q/k (numerics only, no speedup) | fp16/fp32 indexer numerics | the model's official QAT indexer numeric; 0.93 top-512 overlap, deep PPL statistically identical. Keep opt-in until fp4 STORAGE lands. |
| `LLAMA_DSV4_LID_EXACT` | OFF (`=1` enables) | two-pass selection: fast pass-1 (int8/dec allowed) to top-(512+m), exact QAT-fp32 rescore + bitonic select | single-pass selection in the pass-1 numeric class | **class A** — selection bit-exact vs official QAT graph; 36/36 zero-tolerance backend-ops. Implies QAT q/k numerics. Price with QAT_WRITE: −7.1% pp@d65536, +200 µs/layer decode (512k tg gate holds). P2/P3a, commits 2a947edec + 749dfde85. |
| `LLAMA_DSV4_LID_RESCORE_M` | 64 | — | — | rescore margin m (candidates = top_k + m, cap 1024−512). Oracle p100 displacement = 26 on real dumps → 2.5× headroom. |
| `LLAMA_DSV4_LID_QAT_WRITE` | OFF (`=1` enables) | e2m1 QAT round-trip ONCE at lid-cache write (`GGML_OP_DSV4_FP4_RT`); score kernels skip k-side re-sim | k-side QAT re-simulated per score call (EXACT still correct, −28.3% pp) | f16-of-QAT is bit-exact (2-bit mantissa × pow2 scale), so cache values ARE official post-QAT values. Companion to EXACT; prerequisite for P3b packed storage. |
| `LLAMA_DSV4_LID_CACHE_MXFP4` | OFF (`=1` enables) | lid cache stored as packed MXFP4 (68 B/row vs 256 B f16, 3.76×); writes via `GGML_OP_DSV4_QAT_SET_ROWS` (QAT rounding in the scatter); decode reads 68 B rows directly, prefill stages to f16 (bit-exact) | f16/f32 lid cache per `-ctk` | P3b. Rows are QAT by construction → subsumes QAT_WRITE for the container. Forces fused lid at decode. State ver 2. Price: pp −2~3.6%, tg neutral @d131k. Add to 512k serving profile (memory-bound); stays opt-in as default. Gates: op suite zero-tolerance (incl EXACT), coherence/determinism, prompt-cache round-trip, perf legs iter 26. |
| `LLAMA_DSV4_MOE_TILE` | OFF | MoE expert-tile bridge kernels in the graph | standard MoE path | NEGATIVE RESULT (tile bridge regresses; hc-batch memory). Leave OFF. |

## Speculative decoding / MTP (server + common)

| Flag | Default | Effect | Notes |
| --- | --- | --- | --- |
| `LLAMA_DSV4_SPEC_FRONTIER` | **ON** (`=0` disables) | the "frontier rewind" spec-decode rollback (server; code's own name for it) | part of the mtp-21.6 t/s recipe (d29e3b6); `atoi`, any non-zero = on |
| `LLAMA_MTP_FUSED_DRAFT` | OFF (set = on) | fused MTP draft chain (only when !chain_heads && !is_mem_shared) | opt-in; backend-sampling interaction untested — see speculative.cpp comment |
| `LLAMA_SPEC_TRACE` | OFF (set = on) | per-round SPECTRACE acc/draft log lines (server) | debug only |
| `LLAMA_DEBUG_NEXTN` | OFF (set = on) | MTP next-N debug prints (llama-context) | debug only |

## Refusal ablation / control-vector (llama-adapter + deepseek4)

| Flag | Default | Effect | Notes |
| --- | --- | --- | --- |
| `LLAMA_CVEC_ABLATE` | OFF (`=1` enables) | projection ablation `cur -= scale*<cur,dir>*dir` instead of additive steering | the working refusal recipe (bb2814b16): ablate + ffn-only + scale 2.0 |
| `LLAMA_CVEC_ABLATE_SCALE` | 1.0 | ablation strength | float |
| `LLAMA_CVEC_SCALE_FILE` | unset | live file re-read every graph build — runtime scale sweeps on one model load | added after the kill/relaunch hard-reset (crash memory #6) |
| `LLAMA_CVEC_AT_FFN` | OFF (set = on) | apply cvec at FFN (and attn unless FFN_ONLY) | |
| `LLAMA_CVEC_FFN_ONLY` | OFF (set = on) | restrict cvec to FFN — matches ds4.c exactly | required for the 100%-flip result |
| `CAPTURE_PROMPTS` / `CAPTURE_OUT` / `CAPTURE_OUT_THINK` / `CAPTURE_TENSOR` | unset | refusal-capture example: prompt list, output paths, tensor to capture | examples/refusal-capture only |

## Server CLI flags we added (pi-ds4-flash L2 KV tier, dcd503d94)

| Flag | Default | Effect |
| --- | --- | --- |
| `--cache-disk DIR` | unset (tier off) | enables the disk KV-state tier in DIR |
| `--cache-disk-mb` | 65536 | on-disk budget MiB (0 = unlimited) |
| `--cache-disk-min-tokens` | 2048 | smallest prompt worth persisting |
| `--cache-disk-max-entry-mb` | 4096 | largest single state diverted to disk |

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

## Canonical bench/serving env

As of P4a (2026-07-14) the fast profile IS the default — no env needed:
FUSED_LID, CSA_GATHER, LID_INT8, LID_DEC, CSA_TILE, HC_FUSED all ON.

Official-exact validation/reference profile (P2+P3a):

```
LLAMA_DSV4_LID_EXACT=1 LLAMA_DSV4_LID_QAT_WRITE=1
```

Pre-P4a baseline reproduction (all four legacy flags off):

```
LLAMA_DSV4_FUSED_LID=0 LLAMA_DSV4_CSA_GATHER=0 LLAMA_DSV4_LID_DEC=0 LLAMA_DSV4_LID_INT8=0
```
