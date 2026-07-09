#!/usr/bin/env python3
"""Convert DeepSeek-V4-Flash speculative modules (MTP head / DSpark) to standalone draft GGUFs.

Two modes, one script:

  --mode mtp     official plain MTP head   -> arch "deepseek4mtp"  (1 block)
  --mode dspark  official DSpark module    -> arch "dspark"        (3 blocks)

Source tensors come from the module-only safetensors shards downloaded from HF
(deepseek-ai/DeepSeek-V4-Flash and .../DeepSeek-V4-Flash-DSpark). Hyper-parameter
KVs and the tokenizer block are copied from an existing MAIN-model GGUF (deepseek4
arch), renamed to the target arch prefix. token_embd/output are copied RAW (quantized
bytes, same ggml type) from the main GGUF.

Source dtypes (verified from the safetensors headers + config.json expert_dtype=fp4):
  * MoE routed experts  : FP4 (E2M1), 4-bit packed 2/byte along in-dim,
                          E8M0 scale per 1x32 microblock            -> dequant F32 -> Q8_0
  * dense/attn/shexp/*_proj : FP8 E4M3 + E8M0 128x128 block scale   -> dequant F32 -> F16
  * norms               : BF16                                       -> F32
  * gate.weight / heads : BF16                                       -> F16
  * attn_sink / gate.bias / all hc_*  : F32                          -> F32

Python-only (gguf-py + numpy). No torch, no C++, no inference. Run to produce the GGUF
and it self-validates by reloading with gguf-py.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys

import numpy as np

# in-repo gguf-py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gguf-py"))
import gguf  # noqa: E402
import gguf.quants as gq  # noqa: E402
from gguf import GGUFValueType, GGMLQuantizationType  # noqa: E402

# ----------------------------------------------------------------------------
# dtype decoders (pure numpy, all -> float32)
# ----------------------------------------------------------------------------

FP4_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def _build_e4m3fn_lut() -> np.ndarray:
    """OCP FP8 E4M3FN (bias 7, finite; only S.1111.111 is NaN)."""
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if (b >> 7) & 1 else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0:
            lut[b] = s * (m / 8.0) * (2.0 ** (1 - 7))       # subnormal
        elif e == 0xF and m == 0x7:
            lut[b] = 0.0                                     # NaN -> 0
        else:
            lut[b] = s * (1.0 + m / 8.0) * (2.0 ** (e - 7))  # normal
    return lut


E4M3_LUT = _build_e4m3fn_lut()


def e8m0_to_f32(e: np.ndarray) -> np.ndarray:
    """E8M0 exponent-only scale byte -> float32 (mirrors the reference converter)."""
    e = e.astype(np.uint32)
    bits = np.where(e == 0, np.uint32(0x00400000), e << 23)
    return bits.view(np.float32)


def dec_bf16(raw: bytes) -> np.ndarray:
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (u << 16).view(np.float32)


def dec_f32(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float32).astype(np.float32)


def dec_e4m3(raw: bytes) -> np.ndarray:
    return E4M3_LUT[np.frombuffer(raw, dtype=np.uint8)]


def dequant_fp8(wraw, wshape, sraw, sshape, block=(128, 128)) -> np.ndarray:
    out_dim, in_dim = wshape
    bo, bi = block
    assert out_dim % bo == 0 and in_dim % bi == 0, f"fp8 shape {wshape} not divisible by {block}"
    exp = (out_dim // bo, in_dim // bi)
    assert tuple(sshape) == exp, f"fp8 scale {sshape} != expected {exp}"
    w = dec_e4m3(wraw).reshape(out_dim, in_dim)
    s = e8m0_to_f32(np.frombuffer(sraw, dtype=np.uint8)).reshape(sshape)
    w = w.reshape(out_dim // bo, bo, in_dim // bi, bi) * s[:, None, :, None]
    return w.reshape(out_dim, in_dim)


def dequant_fp4(wraw, wshape, sraw, sshape) -> np.ndarray:
    out_dim, packed = wshape
    in_dim = packed * 2
    assert in_dim % 32 == 0, f"fp4 packed shape {wshape} has no 32-value blocks"
    nb = in_dim // 32
    assert tuple(sshape) == (out_dim, nb), f"fp4 scale {sshape} != expected {(out_dim, nb)}"
    w = np.frombuffer(wraw, dtype=np.uint8).reshape(out_dim, nb, 16)
    low = w & 0x0F
    high = (w >> 4) & 0x0F
    vals = np.stack((low, high), axis=-1).reshape(out_dim, nb, 32)
    s = e8m0_to_f32(np.frombuffer(sraw, dtype=np.uint8)).reshape(out_dim, nb)
    d = FP4_TABLE[vals] * s[:, :, None]
    return d.reshape(out_dim, in_dim).astype(np.float32)


# ----------------------------------------------------------------------------
# safetensors multi-shard reader (no torch/safetensors dependency)
# ----------------------------------------------------------------------------


class SafeTensors:
    def __init__(self, paths):
        self.entries = {}   # name -> (path, dtype, shape, abs_off0, abs_off1)
        self._fh = {}
        for p in paths:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            base = 8 + n
            for name, e in hdr.items():
                if name == "__metadata__":
                    continue
                a, b = e["data_offsets"]
                self.entries[name] = (p, e["dtype"], e["shape"], base + a, base + b)

    def _read(self, name):
        p, dt, sh, a, b = self.entries[name]
        fh = self._fh.get(p)
        if fh is None:
            fh = self._fh[p] = open(p, "rb")
        fh.seek(a)
        return dt, sh, fh.read(b - a)

    def has(self, name):
        return name in self.entries

    def decode(self, base):
        """Return an (out,in)-shaped float32 array for a logical tensor `base`.

        Auto-detects the source encoding from the .weight dtype and the presence
        of a sibling .scale (FP8 E4M3 -> block dequant; I8 -> FP4 dequant; else BF16/F32).
        Handles tensors stored with or without a trailing `.weight`.
        """
        wname = base + ".weight" if self.has(base + ".weight") else base
        if wname not in self.entries:
            raise KeyError(f"missing source tensor: {base}")
        sname = base + ".scale"
        dt, sh, wraw = self._read(wname)
        if self.has(sname):
            _, ssh, sraw = self._read(sname)
            if dt == "F8_E4M3":
                return dequant_fp8(wraw, sh, sraw, ssh)
            if dt == "I8":
                return dequant_fp4(wraw, sh, sraw, ssh)
            raise ValueError(f"{base}: unexpected scaled dtype {dt}")
        if dt == "BF16":
            return dec_bf16(wraw).reshape(sh)
        if dt == "F32":
            return dec_f32(wraw).reshape(sh)
        raise ValueError(f"{base}: unexpected unscaled dtype {dt}")


# ----------------------------------------------------------------------------
# main-GGUF helpers: KV copy + raw tensor copy
# ----------------------------------------------------------------------------

SKIP_KV_SUBSTR = ("indexer", "compress_ratios", "compress_rope", "hash_layer")


def _field_value_and_type(field):
    ts = field.types
    if ts and ts[0] == GGUFValueType.ARRAY:
        return field.contents(), GGUFValueType.ARRAY, ts[1]
    return field.contents(), ts[-1], None


def find_main_gguf_shards(path_or_dir):
    if os.path.isdir(path_or_dir):
        shards = sorted(glob.glob(os.path.join(path_or_dir, "*.gguf")))
    elif "of" in os.path.basename(path_or_dir):
        # a split member -> expand to the whole set
        base = path_or_dir.rsplit("-00", 1)[0]
        shards = sorted(glob.glob(base + "-*.gguf")) or [path_or_dir]
    else:
        shards = [path_or_dir]
    if not shards:
        raise FileNotFoundError(f"no main GGUF shards found for {path_or_dir}")
    return shards


def load_main_kv_and_raw(shards):
    """Return (kv_field_dict, {name: ReaderTensor}) across all shards."""
    kv = {}
    raw = {}
    for s in shards:
        r = gguf.GGUFReader(s)
        for k, f in r.fields.items():
            kv.setdefault(k, f)
        for t in r.tensors:
            if t.name in ("token_embd.weight", "output.weight"):
                raw[t.name] = t
    return kv, raw


def copy_kvs(writer, kv, src_prefix, dst_arch, overrides):
    """Copy tokenizer.* verbatim and <src_prefix>.* renamed to <dst_arch>.*.

    `overrides` maps a FINAL key -> value (applied in place of the copied value,
    or added if absent). Returns the set of emitted keys.
    """
    emitted = set()

    def put(key, val, vtype, sub):
        if key in emitted:
            return
        writer.add_key_value(key, val, vtype, sub)
        emitted.add(key)

    for k, field in kv.items():
        val, vtype, sub = _field_value_and_type(field)
        if k.startswith("tokenizer."):
            put(k, val, vtype, sub)
        elif k.startswith(src_prefix + "."):
            if any(s in k for s in SKIP_KV_SUBSTR):
                continue
            newk = dst_arch + "." + k[len(src_prefix) + 1:]
            if newk in overrides:
                continue  # handled below with correct type
            # per-layer arrays (e.g. swiglu_clamp_exp[43]) must match the draft's
            # block_count; values are uniform in the source, so slice
            n_blk = overrides.get(dst_arch + ".block_count", (None,) * 3)[0]
            if n_blk and isinstance(val, list) and len(val) > n_blk and "clamp" in k:
                val = val[:n_blk]
            put(newk, val, vtype, sub)
        elif k == "general.name":
            put(k, val, vtype, sub)

    # explicit / override keys
    for key, (val, vtype, sub) in overrides.items():
        put(key, val, vtype, sub)
    return emitted


def add_raw_tensor(writer, t):
    """Re-add a quantized tensor byte-for-byte, preserving ggml type and shape."""
    ggml_shape = [int(x) for x in t.shape]           # ne order, e.g. [4096, 129280]
    n_rows = int(np.prod(ggml_shape[1:])) if len(ggml_shape) > 1 else 1
    data = np.ascontiguousarray(t.data).reshape(n_rows, -1)
    writer.add_tensor(t.name, data, raw_dtype=t.tensor_type)


# ----------------------------------------------------------------------------
# tensor mapping
# ----------------------------------------------------------------------------

def core_block_map(b):
    """(src_suffix, dst_name, policy) for one transformer block, prefix mtp.{b}."""
    P = f"blk.{b}."
    return [
        ("attn.wq_a",             P + "attn_q_a",       "f16"),
        ("attn.wq_b",             P + "attn_q_b",       "f16"),
        ("attn.q_norm",           P + "attn_q_a_norm",  "f32"),
        ("attn.wkv",              P + "attn_kv",        "f16"),
        ("attn.kv_norm",          P + "attn_kv_a_norm", "f32"),
        ("attn.wo_a",             P + "attn_output_a",  "f16"),
        ("attn.wo_b",             P + "attn_output_b",  "f16"),
        ("attn.attn_sink",        P + "attn_sinks",     "f32"),
        ("attn_norm",             P + "attn_norm",      "f32"),
        ("ffn_norm",              P + "ffn_norm",       "f32"),
        ("ffn.gate.weight",       P + "ffn_gate_inp",   "f16"),
        ("ffn.gate.bias",         P + "exp_probs_b",    "f32"),
        ("ffn.shared_experts.w1", P + "ffn_gate_shexp", "f16"),
        ("ffn.shared_experts.w3", P + "ffn_up_shexp",   "f16"),
        ("ffn.shared_experts.w2", P + "ffn_down_shexp", "f16"),
        ("hc_attn_fn",            P + "hc_attn_fn",      "f32"),
        ("hc_attn_base",          P + "hc_attn_base",    "f32"),
        ("hc_attn_scale",         P + "hc_attn_scale",   "f32"),
        ("hc_ffn_fn",             P + "hc_ffn_fn",       "f32"),
        ("hc_ffn_base",           P + "hc_ffn_base",     "f32"),
        ("hc_ffn_scale",          P + "hc_ffn_scale",    "f32"),
    ]


# expert stacks: (w_suffix, dst_name).  w1=gate, w2=down, w3=up  (DeepSeek convention)
EXPERT_STACKS = [("w1", "ffn_gate_exps"), ("w2", "ffn_down_exps"), ("w3", "ffn_up_exps")]
N_EXPERTS = 256

# The C++ loader resolves tensor names via tn(id[, suffix, layer]); a suffix is
# appended with a '.'. We store the registered base name (no suffix) in the maps
# and apply the exact suffix the create_tensor call uses here:
#   * FFN_EXP_PROBS_B is created with "bias"  (deepseek4.cpp:172 / dspark.cpp:132)
#   * DSPARK_MARKOV_W1/W2 and CONF_PROJ use tn(id) with NO suffix (dspark.cpp:146-148)
#   * every other module tensor uses "weight"
_NO_SUFFIX = {"dspark.markov_w1", "dspark.markov_w2", "dspark.confidence_proj"}


def final_name(base):
    if base.endswith(".exp_probs_b"):
        return base + ".bias"
    if base in _NO_SUFFIX:
        return base
    return base + ".weight"


def emit_scalar(writer, dst, arr, policy):
    arr = np.ascontiguousarray(arr)
    if policy == "f32":
        writer.add_tensor(final_name(dst), arr.astype(np.float32))
    elif policy == "f16":
        writer.add_tensor(final_name(dst), arr.astype(np.float16))
    else:
        raise ValueError(policy)


def emit_experts(writer, st, src_block_prefix, dst_name):
    # Quantize each expert to Q8_0 individually and stack the (small) quantized
    # bytes, so we never hold a multi-GB float32 stack of all 256 experts at once.
    for w_suffix, dname in EXPERT_STACKS:
        q_rows = []
        for i in range(N_EXPERTS):
            arr = st.decode(f"{src_block_prefix}.ffn.experts.{i}.{w_suffix}")  # (out, in) f32
            q_rows.append(gq.quantize(np.ascontiguousarray(arr), GGMLQuantizationType.Q8_0))
        q = np.stack(q_rows, axis=0)   # (256, out, qbytes) uint8
        writer.add_tensor(final_name(f"{dst_name}.{dname}"), q, raw_dtype=GGMLQuantizationType.Q8_0)


# ----------------------------------------------------------------------------
# conversion driver
# ----------------------------------------------------------------------------

def convert(mode, module_dir, main_gguf, out_path):
    if mode == "mtp":
        arch = "deepseek4mtp"
        n_blocks = 1
    elif mode == "dspark":
        arch = "dspark"
        n_blocks = 3
    else:
        raise ValueError(mode)

    st = SafeTensors(sorted(glob.glob(os.path.join(module_dir, "*.safetensors"))))
    main_shards = find_main_gguf_shards(main_gguf)
    kv, raw = load_main_kv_and_raw(main_shards)
    for need in ("token_embd.weight", "output.weight"):
        if need not in raw:
            raise RuntimeError(f"{need} not found in main GGUF shards {main_shards}")

    # use_temp_file spools tensor data to a temp file instead of buffering the
    # whole (up to ~22 GB) output in RAM -- required so the conversion survives
    # when the main model is resident (runtime testing) and free RAM is tight.
    writer = gguf.GGUFWriter(out_path, arch, use_temp_file=True)

    # -------- KVs --------
    U32 = GGUFValueType.UINT32
    overrides: "dict[str, tuple]" = {
        f"{arch}.block_count": (n_blocks, U32, None),
        f"{arch}.nextn_predict_layers": (1, U32, None),
    }
    if mode == "dspark":
        overrides["dspark.dspark.block_size"] = (5, U32, None)
        overrides["dspark.dspark.noise_token_id"] = (128799, U32, None)
        overrides["dspark.dspark.markov_rank"] = (256, U32, None)
        # HF target_layer_ids [40,41,42] are hidden_states[id+1] = layer OUTPUTS;
        # llama.cpp taps layer INPUTS, so store id+1 (43 = final backbone output)
        overrides["dspark.target_layers"] = ([41, 42, 43], GGUFValueType.ARRAY, GGUFValueType.INT32)
    # general.architecture is already set by the GGUFWriter(out, arch) constructor.
    copy_kvs(writer, kv, "deepseek4", arch, overrides)

    # -------- tensors --------
    def blk(b):
        return f"mtp.{b}"

    for b in range(n_blocks):
        src = blk(b)
        for suffix, dst, policy in core_block_map(b):
            emit_scalar(writer, dst, st.decode(f"{src}.{suffix}"), policy)
        emit_experts(writer, st, src, f"blk.{b}")

    if mode == "mtp":
        emit_scalar(writer, "output_norm",     st.decode("mtp.0.norm"),          "f32")
        emit_scalar(writer, "output_hc_fn",    st.decode("mtp.0.hc_head_fn"),    "f32")
        emit_scalar(writer, "output_hc_base",  st.decode("mtp.0.hc_head_base"),  "f32")
        emit_scalar(writer, "output_hc_scale", st.decode("mtp.0.hc_head_scale"), "f32")
        emit_scalar(writer, "blk.0.nextn.e_proj", st.decode("mtp.0.e_proj"), "f16")
        emit_scalar(writer, "blk.0.nextn.h_proj", st.decode("mtp.0.h_proj"), "f16")
        emit_scalar(writer, "blk.0.nextn.enorm",  st.decode("mtp.0.enorm"),  "f32")
        emit_scalar(writer, "blk.0.nextn.hnorm",  st.decode("mtp.0.hnorm"),  "f32")
    else:  # dspark
        # only block 0 taps the target-layer features
        emit_scalar(writer, "fc",              st.decode("mtp.0.main_proj"), "f16")
        emit_scalar(writer, "enc.output_norm", st.decode("mtp.0.main_norm"), "f32")
        emit_scalar(writer, "output_norm",     st.decode("mtp.2.norm"),          "f32")
        emit_scalar(writer, "output_hc_fn",    st.decode("mtp.2.hc_head_fn"),    "f32")
        emit_scalar(writer, "output_hc_base",  st.decode("mtp.2.hc_head_base"),  "f32")
        emit_scalar(writer, "output_hc_scale", st.decode("mtp.2.hc_head_scale"), "f32")
        emit_scalar(writer, "dspark.markov_w1",      st.decode("mtp.2.markov_head.markov_w1"), "f16")
        emit_scalar(writer, "dspark.markov_w2",      st.decode("mtp.2.markov_head.markov_w2"), "f16")
        emit_scalar(writer, "dspark.confidence_proj", st.decode("mtp.2.confidence_head.proj"), "f16")

    # raw quantized token_embd / output copied verbatim from the main GGUF
    add_raw_tensor(writer, raw["token_embd.weight"])
    add_raw_tensor(writer, raw["output.weight"])

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[write] {out_path}")


# ----------------------------------------------------------------------------
# validation (reload, no inference)
# ----------------------------------------------------------------------------

def validate(mode, out_path):
    r = gguf.GGUFReader(out_path)
    arch = r.fields["general.architecture"].contents()
    tensors = {t.name: t for t in r.tensors}
    print(f"\n=== validate {out_path} ===")
    print(f"architecture           : {arch}")
    print(f"tensor count           : {len(tensors)}")
    print(f"KV count               : {len(r.fields)}")
    size = os.path.getsize(out_path)
    print(f"file size              : {size:,} B ({size/1e9:.3f} GB)")

    # tokenizer present
    tok_keys = [k for k in r.fields if k.startswith("tokenizer.")]
    assert any(k == "tokenizer.ggml.tokens" for k in tok_keys), "tokenizer.ggml.tokens missing"
    assert any(k == "tokenizer.ggml.model" for k in tok_keys), "tokenizer.ggml.model missing"
    print(f"tokenizer KVs          : {len(tok_keys)} (tokens+model present)")

    # hparams present
    hp = [k for k in r.fields if k.startswith(arch + ".")]
    print(f"{arch}.* hparams        : {len(hp)}")
    bc = r.fields[f"{arch}.block_count"].contents()
    print(f"{arch}.block_count      : {bc}")
    n_blocks = 1 if mode == "mtp" else 3
    assert bc == n_blocks, f"block_count {bc} != {n_blocks}"

    # suffix contract: every tensor must be .weight/.bias, except the 3 dspark heads
    bad_suffix = [n for n in tensors
                  if not (n.endswith(".weight") or n.endswith(".bias") or n in _NO_SUFFIX)]
    assert not bad_suffix, f"tensors with wrong/missing suffix: {bad_suffix}"
    # exp_probs_b must carry .bias (not .weight)
    for b in range(n_blocks):
        assert f"blk.{b}.exp_probs_b.bias" in tensors, f"blk.{b}.exp_probs_b.bias missing"
        assert f"blk.{b}.exp_probs_b.weight" not in tensors, f"blk.{b}.exp_probs_b wrongly .weight"
    print("suffixes               : .weight/.bias contract OK (exp_probs_b=.bias, dspark heads suffix-less)")

    # expert stacks
    exp_bad = []
    for b in range(n_blocks):
        for dname in ("ffn_gate_exps", "ffn_down_exps", "ffn_up_exps"):
            t = tensors[final_name(f"blk.{b}.{dname}")]
            sh = [int(x) for x in t.shape]
            if sh[-1] != N_EXPERTS:
                exp_bad.append((t.name, sh))
            assert t.tensor_type == GGMLQuantizationType.Q8_0, f"{t.name} not Q8_0"
    assert not exp_bad, f"expert stacks not [*,*,256]: {exp_bad}"
    print(f"expert stacks          : all [*, *, {N_EXPERTS}] Q8_0  (OK)")

    # dspark heads
    if mode == "dspark":
        for hn in ("dspark.markov_w1", "dspark.markov_w2", "dspark.confidence_proj"):
            t = tensors[hn]   # suffix-less by contract
            sh = [int(x) for x in t.shape]
            print(f"  {hn:24s}: {sh} {t.tensor_type.name}")
        m1 = [int(x) for x in tensors["dspark.markov_w1"].shape]
        assert 129280 in m1 and 256 in m1, f"markov_w1 shape family {m1} not (129280,256)"
        m2 = [int(x) for x in tensors["dspark.markov_w2"].shape]
        assert 129280 in m2 and 256 in m2, f"markov_w2 shape family {m2} not (129280,256)"
        assert "fc.weight" in tensors and "enc.output_norm.weight" in tensors, "fc.weight / enc.output_norm.weight missing"

    # raw copies
    for rn in ("token_embd.weight", "output.weight"):
        t = tensors[rn]
        print(f"  {rn:24s}: {[int(x) for x in t.shape]} {t.tensor_type.name}")

    # full tensor listing
    print("--- all tensors ---")
    for n in sorted(tensors):
        t = tensors[n]
        print(f"  {n:34s} {t.tensor_type.name:8s} {[int(x) for x in t.shape]}")
    print("=== validation OK ===")


def main():
    ap = argparse.ArgumentParser(description="Convert DS4-Flash MTP/DSpark module -> draft GGUF")
    ap.add_argument("--mode", required=True, choices=["mtp", "dspark"])
    ap.add_argument("--module-dir", required=True, help="dir with the module safetensors shards")
    ap.add_argument("--main-gguf", required=True, help="main deepseek4 GGUF (dir or any split member) for KVs + token_embd/output")
    ap.add_argument("--out", required=True, help="output GGUF path")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    if not args.validate_only:
        convert(args.mode, args.module_dir, args.main_gguf, args.out)
    validate(args.mode, args.out)


if __name__ == "__main__":
    main()
