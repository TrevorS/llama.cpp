#!/usr/bin/env python3
"""Build the teacher-forcing corpus + alignment metadata from fetched API JSON.

For each api/NN.json:
  1. render the chat template (from the GGUF) over the request messages with
     add_generation_prompt=True, append the API's content -> unit text
  2. tokenize the unit with llama-tokenize (--no-bos, specials parsed) and
     align our token byte-spans against the API's per-token `bytes` arrays
  3. pad the unit with filler so each unit occupies EXACTLY one n_ctx=2048
     chunk of the concatenated corpus (accounting for the global BOS that
     llama-perplexity prepends: file-token index = dump position - 1)

Outputs: corpus.txt (feed to llama-perplexity --save-all-logits) and
meta.json (per unit: chunk index, comparable positions, API logprobs/top20
mapped to our vocab ids).

Usage: build_corpus.py [--model GGUF] [--tokenize BIN] [--template-kwargs JSON]
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE   = os.path.dirname(os.path.abspath(__file__))
REPO   = os.path.abspath(os.path.join(HERE, "..", ".."))
N_CTX  = 1024
FIRST  = N_CTX // 2          # perplexity emits rows for positions >= FIRST
FILLER = " the"              # 1 token each in the DS4 BPE (validated at run)

sys.path.insert(0, os.path.join(REPO, "gguf-py"))
from gguf import GGUFReader  # noqa: E402


def bytes_to_unicode():
    # standard GPT-2 byte<->unicode table (matches llama.cpp BPE)
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("\xa1"), ord("\xac") + 1)) + \
         list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip([chr(c) for c in cs], bs))


class Vocab:
    def __init__(self, gguf_path):
        r = GGUFReader(gguf_path)
        kv = {f.name: f for f in r.fields.values()}
        toks = kv["tokenizer.ggml.tokens"]
        self.pieces = [bytes(toks.parts[i]).decode("utf-8", errors="replace")
                       for i in toks.data]
        tt = kv.get("tokenizer.ggml.token_type")
        self.types = list(tt.parts[i][0] for i in tt.data) if tt else [1] * len(self.pieces)
        ct = kv.get("tokenizer.chat_template")
        self.chat_template = bytes(ct.parts[ct.data[0]]).decode() if ct else None
        u2b = bytes_to_unicode()
        # two candidate byte forms per token: byte-BPE decoded (the standard
        # storage for type-1 tokens) and raw utf-8 (added tokens like math
        # glyphs are stored raw — observed on x, /, approx signs). The span
        # builder matches candidates against the actual text.
        self.tok_bytes = []      # primary (byte-BPE) form
        self.tok_bytes_alt = []  # raw utf-8 form (None if identical)
        for piece, typ in zip(self.pieces, self.types):
            raw = piece.encode("utf-8")
            if typ == 6 and re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", piece):
                # byte-fallback token: literal single byte
                self.tok_bytes.append(bytes([int(piece[3:5], 16)]))
                self.tok_bytes_alt.append(None)
                continue
            if typ != 1:  # control/special/user-defined: literal utf-8
                self.tok_bytes.append(raw)
                self.tok_bytes_alt.append(None)
                continue
            try:
                dec = bytes(u2b[ch] for ch in piece)
            except KeyError:
                dec = raw
            self.tok_bytes.append(dec)
            self.tok_bytes_alt.append(raw if raw != dec else None)
        self.bytes2id = {}
        for i, (b, a) in enumerate(zip(self.tok_bytes, self.tok_bytes_alt)):
            self.bytes2id.setdefault(b, i)
            if a is not None:
                self.bytes2id.setdefault(a, i)


def tokenize(binpath, model, text, tmp):
    with open(tmp, "w") as f:
        f.write(text)
    # --no-escape: the tool otherwise rewrites literal \t, \n etc. in the
    # INPUT (LaTeX like \times becomes TAB+imes) — must match perplexity,
    # which run_arms.sh also invokes with --no-escape
    out = subprocess.run([binpath, "-m", model, "-f", tmp, "--ids", "--no-bos", "--no-escape"],
                         capture_output=True, check=True)
    stdout = out.stdout.decode("utf-8", errors="replace")
    ids = [int(x) for x in re.findall(r"-?\d+", stdout.rsplit("[", 1)[-1])]
    return ids


def render_template(vocab, messages, kwargs):
    import jinja2
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    tmpl = env.from_string(vocab.chat_template)
    return tmpl.render(messages=messages, add_generation_prompt=True,
                       bos_token="", eos_token="", **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser(
        "~/models/ds4/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf"))
    ap.add_argument("--tokenize", default=os.path.join(REPO, "build/bin/llama-tokenize"))
    ap.add_argument("--template-kwargs", default="{}")
    ap.add_argument("--api-dir", default="api")
    ap.add_argument("--out-corpus", default="corpus.txt")
    ap.add_argument("--out-meta", default="meta.json")
    args = ap.parse_args()
    tkw = json.loads(args.template_kwargs)

    vocab = Vocab(args.model)
    if not vocab.chat_template:
        sys.exit("no chat template in GGUF")
    tmp = os.path.join(HERE, ".tok.tmp")

    api_dir = os.path.join(HERE, args.api_dir)
    api_files = sorted(f for f in os.listdir(api_dir) if f.endswith(".json"))
    if not api_files:
        sys.exit("no api/*.json — run fetch_api.py first")

    corpus = []          # unit texts, each exactly N_CTX file tokens
    meta = {"n_ctx": N_CTX, "first": FIRST, "units": []}
    n_file_tokens = 0    # running count; unit k must END at (k+1)*N_CTX - 1
                         # (global BOS shifts dump positions by +1)
    dropped = 0

    for k, fn in enumerate(api_files):
        with open(os.path.join(api_dir, fn)) as f:
            j = json.load(f)
        messages = j["request"]["messages"]
        choice   = j["response"]["choices"][0]
        content  = choice["message"]["content"] or ""
        lp       = (choice.get("logprobs") or {}).get("content") or []

        prompt_text = render_template(vocab, messages, tkw)
        unit_text   = prompt_text + content
        ids = tokenize(args.tokenize, args.model, unit_text, tmp)

        # byte spans of our tokens, matched against the actual text so that
        # raw-stored vocab pieces (math glyphs etc.) resolve correctly
        ubytes = unit_text.encode("utf-8")
        spans, off, bad = [], 0, False
        for tid in ids:
            b = vocab.tok_bytes[tid]
            if ubytes[off:off + len(b)] != b:
                a = vocab.tok_bytes_alt[tid]
                if a is not None and ubytes[off:off + len(a)] == a:
                    b = a
                else:
                    print(f"[{fn}] WARN: token {tid} bytes mismatch at offset {off} — dropping")
                    bad = True
                    break
            spans.append((off, off + len(b), tid))
            off += len(b)
        if bad or off != len(ubytes):
            if not bad:
                print(f"[{fn}] WARN: byte reconstruction {off} != {len(ubytes)} — dropping")
            dropped += 1
            continue

        # API token spans inside the content region
        content_start = len(prompt_text.encode("utf-8"))
        api_spans, aoff = [], content_start
        for t in lp:
            blen = len(bytes(t["bytes"])) if t.get("bytes") else len(t["token"].encode())
            api_spans.append((aoff, aoff + blen, t))
            aoff += blen

        # prompt length constraints for the chunk layout
        n_prompt = sum(1 for s in spans if s[0] < content_start)
        if n_prompt < FIRST + 4:
            print(f"[{fn}] prompt only {n_prompt} tokens (< {FIRST+4}) — dropping")
            dropped += 1
            continue
        if len(ids) > N_CTX - 8:
            print(f"[{fn}] unit {len(ids)} tokens too long — dropping")
            dropped += 1
            continue

        # align: our token span == api token span (byte-exact)
        ours_by_start = {s[0]: s for s in spans}
        comparable = []
        for (a0, a1, t) in api_spans:
            s = ours_by_start.get(a0)
            if not s or s[1] != a1:
                continue
            top = []
            for tl in t.get("top_logprobs", []):
                tb = bytes(tl["bytes"]) if tl.get("bytes") else tl["token"].encode()
                oid = vocab.bytes2id.get(tb, -1)
                top.append({"id": oid, "logprob": tl["logprob"]})
            comparable.append({
                "unit_tok_idx": spans.index(s),
                "id": s[2],
                "api_logprob": t["logprob"],
                "top": top,
            })

        # pad each unit to EXACTLY N_CTX tokens — one standalone perplexity
        # chunk. analyze.py self-locates each unit by token-subsequence match
        # in the dump, so no assumption about perplexity's BOS layout (global
        # add_bos prepend + per-chunk token[0] overwrite) is baked in here.
        target = N_CTX
        text = unit_text
        n = len(ids)
        for _ in range(8):
            if n == target:
                break
            if n > target:
                sys.exit(f"[{fn}] unit overshot chunk ({n} > {target})")
            text = text + FILLER * (target - n)
            ids = tokenize(args.tokenize, args.model, text, tmp)
            n = len(ids)
        if n != target:
            sys.exit(f"[{fn}] padding failed to converge ({n} vs {target})")

        corpus.append(text)
        meta["units"].append({
            "file": fn,
            "unit_idx": k - dropped,
            "ids": ids,                       # full N_CTX token ids (locator key)
            "n_comparable": len(comparable),
            "comparable": comparable,
        })
        n_file_tokens += target
        print(f"[{fn}] unit {k-dropped}: prompt {n_prompt} tok, "
              f"{len(comparable)}/{len(lp)} comparable ({100.0*len(comparable)/max(1,len(lp)):.0f}%)")

    with open(os.path.join(HERE, args.out_corpus), "w") as f:
        f.write("".join(corpus))
    with open(os.path.join(HERE, args.out_meta), "w") as f:
        json.dump(meta, f)
    os.unlink(tmp)
    print(f"\ncorpus: {len(corpus)} units ({dropped} dropped), "
          f"{n_file_tokens} file tokens -> run:\n"
          f"  llama-perplexity -m <model> -f {HERE}/corpus.txt -c {N_CTX} "
          f"--chunks {len(corpus)} --parse-special --save-all-logits <arm>.kld "
          f"-fa on -ub 2048 -b 2048 --no-mmap")


if __name__ == "__main__":
    main()
