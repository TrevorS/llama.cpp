#!/usr/bin/env python3
"""Fetch DeepSeek-API reference completions with top-20 logprobs.

Calls api.deepseek.com chat completions (model deepseek-v4-flash) at
temperature 0 with logprobs enabled and saves one JSON per prompt under
experiments/parity/api/. The per-token `bytes` arrays are the alignment
key for build_corpus.py.

Requires DEEPSEEK_API_KEY in the environment. stdlib only.

Prompt design constraints (see build_corpus.py):
- perplexity only emits logit rows for the second half of each n_ctx=2048
  chunk, so every prompt must tokenize to >= ~1060 tokens -> each prompt
  carries a long fixed preamble (~4.3k chars).
- prompt + 256 completion tokens must stay under ~1780 tokens.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

API_URL = "https://api.deepseek.com/chat/completions"
MODEL   = os.environ.get("PARITY_MODEL", "deepseek-v4-flash")
OUTDIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api")
MAX_TOKENS = 256

# ~4.3k chars of neutral, information-dense context. Long on purpose: it
# pushes every prompt past the 1024-token chunk midpoint. Generic content,
# nothing private.
PREAMBLE = (
    "You are assisting with a series of independent tasks. Before the task, "
    "here is background reading that may or may not be relevant; read it "
    "carefully either way.\n\n"
    "Background: The history of mechanical computation stretches from the "
    "Antikythera mechanism, an ancient Greek geared device for predicting "
    "astronomical positions, through the calculating machines of Pascal and "
    "Leibniz in the seventeenth century, to Babbage's Difference Engine and "
    "the never-completed Analytical Engine in the nineteenth. Ada Lovelace's "
    "notes on the Analytical Engine contain what is often described as the "
    "first published algorithm intended for execution by a machine, a "
    "procedure for computing Bernoulli numbers. The twentieth century brought "
    "electromechanical relays, then vacuum tubes, then transistors, each "
    "generation shrinking the switching element by orders of magnitude while "
    "multiplying its speed. The integrated circuit, demonstrated "
    "independently by Kilby and Noyce, put many transistors on one die; "
    "Moore's 1965 observation that transistor counts doubled roughly every "
    "year (later revised to every two years) held as a self-fulfilling "
    "industry roadmap for five decades. Dennard scaling, the companion "
    "observation that power density stays constant as transistors shrink, "
    "broke down around 2006, ending the era of free single-thread speedups "
    "and pushing architects toward multicore designs, wide vector units, and "
    "eventually domain-specific accelerators. Graphics processors, built for "
    "embarrassingly parallel rasterization workloads, turned out to be well "
    "suited to the dense linear algebra at the heart of neural networks; a "
    "single modern accelerator performs more multiply-accumulate operations "
    "per second than every computer on Earth combined could manage in 1990. "
    "Memory technology followed its own slower curve: core memory gave way "
    "to DRAM, whose capacity grew faster than its bandwidth, and whose "
    "bandwidth grew faster than its latency improved. This growing imbalance "
    "between arithmetic throughput and memory access, often called the "
    "memory wall, now dominates the design of high-performance software. "
    "Caches, prefetchers, and out-of-order execution hide latency for "
    "irregular workloads; blocking, tiling, and fusion restructure regular "
    "computations to reuse data already resident in fast storage. In "
    "large-scale machine learning inference the arithmetic is so regular and "
    "the models so large that performance is frequently bounded by how fast "
    "parameters stream from memory, making bytes-per-parameter, cache "
    "residency, and quantization the levers that matter most. Quantization "
    "trades numeric precision for footprint: eight-bit integers, four-bit "
    "floating point microformats with shared exponents, and even two-bit "
    "codebook schemes each buy bandwidth at some cost in fidelity, and the "
    "engineering question is always whether the fidelity loss is measurable "
    "in the metrics the application cares about. Speculative execution "
    "appears here too, in a different costume: a small cheap model proposes "
    "several tokens which the large model verifies in one pass, converting "
    "serial decoding into parallel verification exactly as branch predictors "
    "convert control dependences into speculative straight-line work.\n\n"
    "That concludes the background reading. Now the task:\n\n"
)

TASKS = [
    # code
    "Write a Python function that merges two sorted lists into one sorted list without using sort(). Include a short docstring.",
    "Explain what this C fragment does and identify the bug: `for (int i = 0; i <= n; i++) { total += arr[i]; }`",
    "Write a SQL query that returns the three most recent orders per customer from tables orders(id, customer_id, created_at) using a window function.",
    "Convert this JSON `{\"name\":\"Ada\",\"langs\":[\"py\",\"c\"]}` into an equivalent YAML document and explain one difference between the formats.",
    "Write a bash one-liner that finds the five largest files under /var/log and prints their sizes human-readably.",
    # math
    "A train leaves at 09:10 travelling 84 km/h. A second train leaves the same station at 09:40 at 112 km/h on the same track. At what time does the second train catch the first? Show your steps.",
    "Compute the derivative of f(x) = x^3 * ln(x) and evaluate it at x = e. Show your work.",
    "A bag has 5 red, 4 blue, 3 green marbles. Two are drawn without replacement. What is the probability both are the same color? Give an exact fraction.",
    "Solve the system: 2x + 3y = 12 and x - y = 1. Show each step.",
    # prose / summary
    "Continue this opening line of a short story for one paragraph, keeping the same tone: 'The lighthouse keeper had not spoken to another person in forty days, and the sea had begun to answer back.'",
    "Summarize the background reading above in exactly three sentences.",
    "Rewrite the following sentence in plain language: 'The aforementioned stipulations notwithstanding, the party of the first part shall retain unencumbered access to the easement.'",
    # factual QA
    "What is the difference between DRAM latency and DRAM bandwidth, and why has their ratio worsened over time? Answer in one paragraph.",
    "Name the two people credited with independently demonstrating the integrated circuit, and state in one sentence what each contributed.",
    "According to the background reading, what ended the era of free single-thread speedups, and roughly when?",
    # reasoning
    "Alice, Bob and Carol each own one pet: a cat, a dog, a parrot. Alice is allergic to fur. Bob's pet can talk. Who owns the dog? Explain briefly.",
    "If all bloops are razzies and some razzies are lazzies, can we conclude some bloops are lazzies? Explain in two sentences.",
    "You have a 3-litre jug and a 5-litre jug. Describe the shortest sequence of steps to measure exactly 4 litres.",
    # translation / multilingual
    "Translate into French, then German: 'The library opens at nine and closes at midnight on weekdays.'",
    "Translate this sentence to Spanish and explain one grammatical choice you made: 'She would have finished the report if the data had arrived on time.'",
]


def call_api(key: str, task: str, thinking_off: bool):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PREAMBLE + task}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "logprobs": True,
        "top_logprobs": 20,
        "stream": False,
    }
    if thinking_off:
        body["thinking"] = {"type": "disabled"}
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read()), body


def main():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("DEEPSEEK_API_KEY not set")
    os.makedirs(OUTDIR, exist_ok=True)
    thinking_off = True
    for i, task in enumerate(TASKS):
        out = os.path.join(OUTDIR, f"{i:02d}.json")
        if os.path.exists(out):
            print(f"{out} exists, skipping")
            continue
        for attempt in range(3):
            try:
                resp, body = call_api(key, task, thinking_off)
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                # unknown-parameter rejection -> drop the thinking field once
                if thinking_off and e.code == 400 and "thinking" in detail:
                    print(f"[{i:02d}] server rejected thinking param, retrying without it")
                    thinking_off = False
                    continue
                print(f"[{i:02d}] HTTP {e.code} ({detail}), attempt {attempt+1}")
                time.sleep(5 * (attempt + 1))
            except Exception as e:  # noqa: BLE001
                print(f"[{i:02d}] {e}, attempt {attempt+1}")
                time.sleep(5 * (attempt + 1))
        else:
            sys.exit(f"[{i:02d}] failed after retries")

        choice = resp["choices"][0]
        msg = choice["message"]
        if msg.get("reasoning_content"):
            print(f"[{i:02d}] WARNING: reasoning_content present "
                  f"({len(msg['reasoning_content'])} chars) — thinking not disabled; "
                  "alignment will fail. Investigate the thinking parameter.")
        lp = choice.get("logprobs")
        n_tok = len(lp["content"]) if lp and lp.get("content") else 0
        with open(out, "w") as f:
            json.dump({"request": body, "response": resp}, f)
        print(f"[{i:02d}] ok: {n_tok} tokens, finish={choice.get('finish_reason')}, "
              f"content={len(msg.get('content') or '')} chars")
        time.sleep(1)
    print("done")


if __name__ == "__main__":
    main()
