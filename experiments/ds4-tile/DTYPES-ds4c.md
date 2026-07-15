# ds4.c / ds4_cuda.cu — DeepSeek-V4-Flash serving DTYPE MAP

Ground-truth serving-numerics reference for the llama.cpp DS4-Flash port.
Files: `/home/trevor/Projects/ds4/ds4.c` (30,209 lines), `/home/trevor/Projects/ds4/ds4_cuda.cu` (14,100 lines).

**Core invariant:** the residual/activation stream is **f32 everywhere**. All low precision (f16 weights,
int8 Q8_0/Q8_K activations, E4M3 KV, E2M1/FP4 indexer) is either a *storage* dtype or a *simulated
quantization round-trip* that immediately widens back to f32. Weights are dequantized **on the fly inside
dot kernels** — no pre-dequantized weight copies on CPU (GPU repacks Q8_0→f16 device buffers as the one
exception, cuda:667,687-691).

---

## 1. Weight storage per role

Enforced by `tensor_expect_layout`, hard-exit on mismatch (ds4.c:3164-3180).

| Role | Storage | file:line |
|---|---|---|
| Token embeddings | **F16** | ds4.c:3594 |
| Output / lm_head ("OutQ8") | **Q8_0** | ds4.c:3605 |
| All RMSNorm weights + attn_sinks | **F32** | ds4.c:3604,3619,3621,3624,3625,3635,3645,3651 |
| Router gate `ffn_gate_inp` | **F16** (MTP F16/F32) | ds4.c:3652 / 3703 |
| Router bias `ffn_exp_probs_b` | **F32** | ds4.c:3653 |
| Dense attn proj q_a/q_b/kv/output_a/output_b ("AProjQ8") | **Q8_0** | ds4.c:3620-3627 |
| Attn + indexer compressors (ape/kv/gate) | **F16**, norms **F32** | ds4.c:3632-3634, 3641-3645 |
| `indexer_attn_q_b` | **F16 or Q8_0** | ds4.c:3640 |
| Routed experts gate/up=IQ2_XXS, down=Q2_K ("IQ2XXS-w2Q2K"; Q4_K high-mem) | **IQ2_XXS / Q2_K / Q4_K** | ds4.c:3654-3656, 3255-3259 |
| Shared experts ("SExpQ8") | **Q8_0** | ds4.c:3661-3663 |
| HC fn / scale / base | **F16 / F32 / F32** | ds4.c:3616-3618, 3648-3650 |
| MTP e_proj / h_proj | **Q8_0**; enorm/hnorm/norm **F32** | ds4.c:3680-3684 |
| Hash-router table | **I32** | ds4.c:3665 |

GGUF type table at ds4.c:1562-1592 (lists bf16=30), but the enforced `DS4_TENSOR_*` enum admits only
F32/F16/Q8_0/Q2_K/Q4_K/IQ2_XXS/I32 — **no role loads as bf16**.

---

## 2. Dequant / activation-cast strategy

- **F16 weights** → f32 on the fly: `dot_f16_row`, `acc += f16_to_f32(row[i]) * x[i]`, f32 accum (ds4.c:4576-4582).
  Embedding lookup `out[i]=f16_to_f32(row[i])` (ds4.c:4521).
- **Q8_0 matmuls**: activation quantized f32→int8 per-32-block, `d=amax/127` (ds4.c:4949-4972); dot
  int8×int8→**int32** (`dot_i8_32`, ds4.c:4698-4709) scaled by `f16(block_scale)×f32(act_scale)`,
  accumulated in **f32** (ds4.c:4770). GPU uses `__dp4a` (cuda:3830-3840).
- `matvec_f32` deliberately accumulates in **double** (ds4.c:5503-5508). Dispatch `matvec_any` handles
  f32/f16/q8_0 (ds4.c:5525-5533).

---

## 3. MLA attention (CPU ref + CUDA numerically identical)

- **Raw SWA KV cache**: f32 buffer, each pushed element **f16 round-tripped** on CPU
  (`dst[i]=f16_to_f32(f32_to_f16(kv[i]))`, ds4.c:8572,8581); GPU keeps raw KV strictly f32 (cuda:3799,8904).
- **Compressed KV cache**: f32 buffer. Non-RoPE dims get an **E4M3 fake-quant** — per-64-block,
  `scale=2^ceil(log2(amax/448))`, clamp ±448, round-tripped through `dsv4_e4m3fn_dequant` (ds4.c:2517-2532;
  GPU `fp8_kv_quantize_kernel` cuda:4642-4663, bit-identical). Comment ds4.c:2514-2516 ("E4M3-style round
  trip … comparable to the Metal graph's compressed-cache behavior"). On CPU the whole row is *additionally*
  f16 round-tripped (ds4.c:8587) — nope dims see E4M3→f16 in sequence, RoPE tail f16 only. GPU compressed
  cache is **f32 or f16 selectable** (`DS4_GPU_ATTN_COMP_CACHE_F16`, cuda:3795-3815), widened on read via
  `comp_kv_load_elem`.
- **QK^T / softmax / AV**: all **f32**. `kq_scale=1/sqrt(head_dim)`, f32 dot, softmax max seeded from
  per-head **attention-sink** logit, `expf`, denom includes `expf(sink−max)`, AV f32 `axpy`, final `/denom`
  (CPU ds4.c:9028-9079 / 7081-7107; CUDA cuda:4801-4851).
- **RoPE**: **f32** trig (`cosf`/`sinf`/`powf`), in-place on f32, only the `n_rot` tail (ds4.c:6870-6918;
  cuda:4496-4545). Dense vs compressed layers use **different bases** (`DS4_ROPE_FREQ_BASE` vs
  `DS4_COMPRESS_ROPE_FREQ_BASE`) and scale (compressed=1/scale_factor, YaRN ext only when compressed &
  scale>1) (ds4.c:6920-6957).

---

## 4. Lightning indexer (lid) — end-to-end

- **Indexer QAT** — official-graph comment ds4.c:2594-2597: *"the official DeepSeek V4 graph rotates indexer
  activations with a 128-wide Hadamard … and immediately runs the FP4 activation-simulation round trip …
  without it, the top-k compressed-row selection is not the model's graph"*. 128-wide **Hadamard**
  (×0.0883883 = 1/√128) then **E2M1/FP4 fake-quant** — per-32-block, `scale=2^ceil(log2(amax/6))`, clamp ±6,
  e2m1 codebook `{0,.5,1,1.5,2,3,4,6}`. CPU `dsv4_indexer_qat_row_inplace_cpu` /
  `dsv4_fp4_act_quantize_row_inplace_cpu` (ds4.c:2559-2608); GPU `indexer_hadamard_fp4_kernel`
  (cuda:4666-4706), bit-identical. Applied to indexer Q rows (ds4.c:9148,9214) and indexer compressor KV
  (ds4.c:8764). All f32 in/out.
- **lid K cache** (`index_comp_kv`, head_dim=128): **f32** (ds4.c:8341,8543).
- **Scoring kernels**: inputs/weights/K/scores all **f32** device buffers (validated cuda:7827-7830);
  **no int8 path**. Fallback `indexer_scores_kernel` pure f32 (cuda:6706-6745). The **WMMA family**
  (`indexer_scores_wmma{,32,64,128}_kernel`) converts f32→**__half** into shared tiles (`__float2half`,
  cuda:6842,6855) and runs `matrix_a=__half, matrix_b=__half, accumulator=float` (cuda:6859-6866);
  ReLU + per-head weight + scale applied afterward in **f32**; scores output **f32**. `g_quality_mode`
  forces the pure-f32 fallback (cuda:7812-7885).
- **Top-k**: scores compared f32, indices output **uint32**; fast paths gated on **top_k==512**
  (cuda:7299-8085); CUB radix packs f32-orderable key + inverted index into uint64.

---

## 5. MoE

- **Router**: F16 gate. GPU is a **dedicated F16 cuBLAS path** (f32→f16 activation `f32_to_f16_kernel`
  cuda:8521; `cublasGemmEx` `CUDA_R_16F` in / **`CUDA_R_32F`** compute+out cuda:8533-8542; shape-special-cased
  in=4096,out=256 cuda:8488). CPU is `matvec_any`→f32. Score is **not softmax** — `probs=sqrt(softplus(logit))`
  in f32 (ds4.c:7365; cuda:6446). Biased top-6-of-256 selection (bias `ffn_exp_probs_b` added), weights =
  unbiased probs normalized ×`expert_weight_scale=1.5` (ds4.c:7443-7458, ds4.c:204; cuda:6455-6475).
  Hash-routing alternative via `ffn_gate_tid2eid` (ds4.c:7349).
- **Gate_up expert**: IQ2_XXS (or Q4_K) weight · **Q8_K** int8-quantized activation → **int32** dp4a accum
  (`dev_dot_iq2_xxs_q8_K_block` cuda:10440-10462: `0.125·d·bsum`, `d=f16(x→d)·f32(y→d)`) → f32. gate/up accum
  f32; SwiGLU `silu(gate)·up·expert_weight` fused inline in f32 with pre-clamp (cuda:11145-11159; CPU
  ds4.c:7489-7539).
- **Down-proj**: **Q2_K** weight (two f16 scales `d`+`dmin`, min-correction `dall·isum − dmin·summs`,
  cuda:10787-10803; Q4_K variant cuda:12139). The SwiGLU `mid` is **re-quantized f32→Q8_K** before down
  (a 2nd quant point, cuda:13465 / ds4.c:7508). int32 dp4a → f32. Router weight already folded into `mid`,
  so down carries none.
- **Shared expert**: Q8_0 SwiGLU MLP, always active, activation quantized to **Q8_0 (32-blocks)** vs routed
  **Q8_K (256-blocks)** — int8→int32→f32 (ds4.c:7211-7237).
- **No WMMA/tensor-core in any expert matmul** — all `__dp4a` int8. WMMA (`__half`/`__half`/`float`) only in
  F16 router/verify and indexer GEMMs (cuda:3641-3643, 6859-6862).

---

## 6. HC / head-compression / Sinkhorn

Projection matrices (`hc_*_fn`) **F16**, scale/base **F32**. `hc_split_sinkhorn_one` **entirely f32**:
f32 sigmoid gates, f32 row-softmax, **20** f32 normalization iterations, eps=1e-6 (ds4.c:6360-6437; GPU
`hc4_split_one` cuda:6051-6097 also f32). HC-state RMSNorm f64 accum (CPU); weighted-sum reduction f32
(ds4.c:6441-6454).

---

## 7. MTP head

Identical f32-activation / quantized-weight machinery. `e_proj`/`h_proj` **Q8_0**, enorm/hnorm/norm **F32**,
HC head fn F16/F32; routed experts IQ2/Q2K/Q4K, shared Q8_0 (ds4.c:3670-3714). No distinct MTP compute dtype.

---

## 8. Norms & deliberate reproducibility choices

- **RMSNorm sum-of-squares**: CPU **f64** (`double ss`, ds4.c:4527-4528,4536-4537,4548); CUDA **f32**
  (cuda:4317-4321,4340-4344). ← a real CPU↔GPU divergence.
- CPU `matvec_f32` accumulates in **double** for order-independence (ds4.c:5503-5508).
- MoE down-proj GPU: `DS4_CUDA_MOE_NO_ATOMIC_DOWN` forced on → **order-deterministic** accumulation
  ("its atomic accumulation is itself nondeterministic", ds4.c:25197-25198, 25342-25343).
- Compressor/indexer rows finalized in cold-prompt order under streaming (ds4.c:28428-28435); decode scratch
  preallocated for VM-deterministic generation (ds4.c:8384-8385). Bit-exact verify harness refs
  ds4.c:17947,21731,25521.

---

## Summary table

| Stage | Storage | Movement | Compute / accum | Conversions |
|---|---|---|---|---|
| Embeddings | F16 table | → f32 | — | f16→f32 (4521) |
| RMSNorm | F32 weights | f32 act | **CPU f64** / GPU f32 sum-sq | none |
| Dense attn proj | Q8_0 | act f32→int8 | int8→int32→**f32** | quantize_q8_0 (4949) |
| Raw KV cache | f32 (CPU f16-rt) | f16 round-trip | copy | f32↔f16 (8572) |
| Compressed KV | f32 or f16 | **E4M3** ±448/64 + f16 | pool f32 | e4m3 (2527-32), cuda:4657 |
| QK·softmax·AV | — | f32 (comp widened) | **f32**, sink-seeded | comp f16→f32 |
| RoPE | f32 in-place | none | **f32** trig; dense/comp bases | none |
| Indexer QAT | f32 | Hadamard + **FP4** ±6/32 | f32 | e2m1 (2584), cuda:4704 |
| Indexer scoring | f32 | WMMA: f32→**__half** | half×half→**f32** | __float2half (6842,6855) |
| Indexer K-cache / top-k | f32 / uint32 | — | f32 compare, top_k=512 | none |
| Router | F16 gate | GPU f32→f16 (cuBLAS 16F/32F) | f32 softplus+sqrt, ×1.5 | f32_to_f16 (8521) |
| Routed gate_up | IQ2_XXS/Q4_K | act f32→**Q8_K** | int32 dp4a→**f32**; SwiGLU f32 | Q8_K quant (13164) |
| Routed down | Q2_K/Q4_K | mid f32→**Q8_K** (re-quant) | int32 dp4a→**f32** (d+dmin) | Q8_K quant (13465) |
| Shared expert | Q8_0 | act f32→Q8_0(32) | int8→int32→f32 | quantize_q8_0 |
| HC / Sinkhorn | F16 fn / F32 | f32 | **f32** (20 iters) | fn f16→f32 |
| MTP | Q8_0 / F32 | act f32→int8 | f32 (same as main) | — |
| Output head | Q8_0 | act f32→int8 | int8→int32→f32 | — |

---

## Bottom line for the port

Ground-truth serving keeps every activation/residual in f32; the only quantizers that change *values*
(not just storage) are:

1. **E4M3 compressed-KV round-trip** (nope dims, block-64, ±448)
2. **Indexer Hadamard+FP4 round-trip** (block-32, ±6 — mandatory for correct top-k per the official-graph comment)
3. **f16 round-trip on raw-KV push** (CPU only)
4. **int8 Q8_0/Q8_K activation quant** feeding every quantized matmul

Two knobs for bit-parity:

- **RMSNorm accumulation** — CPU f64 vs GPU f32
- **Ordered-vs-atomic MoE down-projection sum**

Evidence corroborated across direct reads + four parallel research passes (indexer, MLA, HC/MTP/weights, MoE).
