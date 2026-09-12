"""Fixed greedy probe set for A/B'ing one server instance against itself.

The earlier MTP A/B compared this checkout against a separate worktree at a
different context length, so it could not separate speculation from the
environment. Run this against the same instance with --experimental-mtp off,
then again with it on: identical hashes prove equivalence, and the timing delta
is the real speedup.

Two measured constraints on what this can prove:

* Compare cold against cold only. A radix prefix hit changes greedy output
  versus a cold prefill (measured: same prompt, temperature 0, cold run differs
  from two separate warm runs, which match each other bit for bit). Every probe
  prompt here is unique, so each one is cold in both runs.
* Warm-vs-warm can never expose a poisoned donated snapshot, because a poisoned
  state reproduces itself. Only MTP off versus MTP on, both cold, is a valid
  equivalence test.

These checkpoints are reasoners, so `content` stays empty until the thinking
budget is spent; the harness hashes `reasoning_content` as well.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request

PARAGRAPH = (
    "Border Gateway Protocol selects the best path for each prefix by walking an ordered "
    "list of attributes. Weight is Cisco-local and comes first, then local preference for "
    "traffic engineering, then locally originated paths, then the shortest AS path, then "
    "origin type, then multi exit discriminator, then eBGP over iBGP, then the IGP cost to "
    "the BGP next hop, then the lowest router id, then the shortest cluster list, then the "
    "lowest originator id, and finally the lowest neighbor address. "
)

PROBES = [
    {
        "name": "arithmetic",
        "messages": [{"role": "user", "content":
            "List every integer from 1 to 40 that is divisible by 3 or by 4, one per line, "
            "and after each number write its remainder when divided by 7 in parentheses. "
            "No commentary."}],
        "max_tokens": 1200,
    },
    {
        "name": "python",
        "messages": [{"role": "user", "content":
            "Write a Python function `merge_intervals(intervals)` that merges overlapping "
            "closed intervals and returns them sorted, then write a second function "
            "`insert_interval(merged, new)` that inserts into an already-merged list in "
            "O(n). Add a short __main__ block with five assertions. No prose."}],
        "max_tokens": 2000,
    },
    {
        "name": "prose",
        "messages": [{"role": "user", "content":
            "Explain in three paragraphs why a serving engine that streams expert weights "
            "over PCIe from host memory has a very different fixed-versus-marginal cost "
            "model per decode step than one whose weights are all resident in HBM, and "
            "what that implies for verifying several draft tokens in a single pass."}],
        "max_tokens": 1600,
    },
    {
        "name": "repetition",
        "messages": [{"role": "user", "content":
            "Output exactly the following pattern and nothing else: sixteen lines, each "
            "line containing the digit 0 repeated twice, then a space, then the line "
            "number. Start at 1."}],
        "max_tokens": 1000,
    },
    {
        "name": "stop_string",
        "messages": [{"role": "user", "content":
            "Count down from 30 to 1, one number per line. Stop immediately at 20."}],
        "max_tokens": 1000,
        "stop": ["20"],
    },
]


def long_context_probe(tokens: int) -> dict:
    reps = max(1, tokens // 90)
    context = " ".join([PARAGRAPH] * reps)
    return {
        "name": f"long_{reps * 90}t",
        "messages": [{"role": "user", "content":
            context + "\n\nQuestion: list, in order, the first four tiebreakers that come "
            "after multi exit discriminator, and say what each one is for. Quote the text."}],
        "max_tokens": 1400,
    }


def ask(base_url: str, model: str, probe: dict, timeout: int) -> dict:
    body = {
        "model": model,
        "messages": probe["messages"],
        "temperature": 0,
        "max_tokens": probe["max_tokens"],
        "stream": False,
    }
    if probe.get("stop"):
        body["stop"] = probe["stop"]
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    elapsed = time.time() - start
    message = data["choices"][0]["message"]
    # These checkpoints are reasoners: `content` is empty until the thinking budget is
    # spent and the answer lives in `reasoning_content`. Hashing only `content` compares
    # two empty strings and reports a spurious match.
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    text = f"<reasoning>\n{reasoning}\n<answer>\n{content}"
    usage = data.get("usage", {})
    completion = int(usage.get("completion_tokens") or 0)
    prompt = int(usage.get("prompt_tokens") or 0)
    return {
        "name": probe["name"],
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "finish_reason": data["choices"][0].get("finish_reason"),
        "reasoning_chars": len(reasoning),
        "answer_chars": len(content),
        "seconds": round(elapsed, 2),
        "tok_per_s": round(completion / elapsed, 2) if elapsed > 0 else None,
        "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
        "text": text,
    }


def server_state(base_url: str) -> dict:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
            return json.load(resp)
    except (urllib.error.URLError, TimeoutError):
        return {}


def log_stats(log_path: str | None) -> dict:
    """Last MTP line and the peak expert-unique ratio, straight from the server log."""
    if not log_path:
        return {}
    try:
        out = subprocess.run(
            ["tail", "-c", "4000000", log_path], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    rounds = re.findall(
        r"MTP k=(\d+) rounds=(\d+) accepted=(\d+) acceptance=([\d.]+) "
        r"mean_drafts=([\d.]+) unique_experts=([\d.]+)/(\d+)",
        out,
    )
    if not rounds:
        return {"mtp": "no MTP lines"}
    k, n, acc, rate, drafts, uniq, worst = rounds[-1]
    return {
        "k": int(k),
        "rounds": int(n),
        "acceptance": float(rate),
        "mean_drafts": float(drafts),
        "tokens_per_round": round(float(drafts) + 1.0, 2),
        "unique_experts": float(uniq),
        "unique_worst": int(worst),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18090")
    ap.add_argument("--model", default="qwen3.8-flash-next-mtp")
    ap.add_argument("--label", required=True, help="run tag, e.g. mtp-off-kv160k")
    ap.add_argument("--out", default="results/mtp/probes")
    ap.add_argument("--log", default=None, help="server log, for acceptance stats")
    ap.add_argument("--long-context-tokens", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    probes = list(PROBES)
    if args.long_context_tokens:
        probes.append(long_context_probe(args.long_context_tokens))

    health = server_state(args.base_url)
    if not health.get("status"):
        raise SystemExit(f"no server at {args.base_url}")

    results = []
    for probe in probes:
        try:
            row = ask(args.base_url, args.model, probe, args.timeout)
        except (urllib.error.URLError, TimeoutError) as exc:
            row = {"name": probe["name"], "error": str(exc)}
        results.append(row)
        print(f"{row['name']:<16} {row.get('tok_per_s', '?'):>7} tok/s  "
              f"{row.get('completion_tokens', '?'):>5} tok  sha={row.get('sha256', '-')}")

    payload = {
        "label": args.label,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "health": health,
        "probes": results,
        "mtp": log_stats(args.log),
    }
    path = f"{args.out}/{args.label}.json"
    os.makedirs(args.out, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {path}")
    if payload["mtp"]:
        print("server:", json.dumps(payload["mtp"]))


if __name__ == "__main__":
    main()
