"""Save deterministic streamed completions and timings for an MTP A/B run."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


CASES = [
    ("arithmetic", "Question: What is 17 times 23? Explain your calculation.\nAnswer:"),
    ("python", "Write a Python function that returns the first n Fibonacci numbers.\n```python\n"),
    ("prose", "Explain why the sky looks blue in three sentences.\nExplanation:"),
]


def generate(base, prompt, max_tokens, **options):
    payload = {"model": "mtp-test", "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0, "top_p": 1, "top_k": -1, "stream": True,
               "stream_options": {"include_usage": True}}
    payload.update(options)
    request = urllib.request.Request(base + "/v1/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    chunks, times, usage = [], [], {}
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                text = choice.get("text", "")
                if text:
                    chunks.append(text)
                    times.append(time.perf_counter() - started)
    elapsed = time.perf_counter() - started
    tokens = usage.get("completion_tokens")
    return {"prompt": prompt, "text": "".join(chunks), "usage": usage,
            "elapsed_seconds": elapsed, "ttft_seconds": times[0] if times else None,
            "decode_tokens_per_second": ((tokens - 1) / (times[-1] - times[0])
                                          if tokens and len(times) > 1 else None)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18090")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--suite", choices=["basic", "edges", "long"], default="basic")
    parser.add_argument("--wait-ready", action="store_true")
    args = parser.parse_args()
    if args.wait_ready:
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(args.base_url + "/health", timeout=10) as response:
                    health = json.load(response)
                if health["status"] == "ok":
                    break
                if health["status"] == "error":
                    raise RuntimeError(health)
            except urllib.error.URLError:
                pass
            time.sleep(3)
        else:
            raise TimeoutError("Server did not finish loading")
    results = []
    cases = [(name, prompt, {}) for name, prompt in CASES]
    if args.suite == "edges":
        prompt = CASES[0][1]
        cases = [
            ("one_token", prompt, {"max_tokens": 1}),
            ("two_tokens", prompt, {"max_tokens": 2}),
            ("stop_string", prompt, {"max_tokens": 32, "stop": "."}),
            ("sampled_fallback", prompt, {"max_tokens": 16, "temperature": 0.8,
                                           "top_k": 20, "top_p": 0.95}),
            ("greedy_after_sampled", prompt, {"max_tokens": 32}),
        ]
    elif args.suite == "long":
        notes = "".join(f"Note {i}: blue light scatters more strongly than red light.\n"
                        for i in range(32))
        cases = [("long_prompt", notes + "\nQuestion: Which color scatters more strongly, "
                  "blue or red? Answer in one sentence.\nAnswer:", {"max_tokens": 32})]
    for name, prompt, options in cases:
        for repeat in range(args.repeats):
            settings = {"max_tokens": args.max_tokens, **options}
            result = generate(args.base_url, prompt, **settings)
            result.update(case=name, repeat=repeat)
            results.append(result)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(name, repeat, result["usage"], result["decode_tokens_per_second"], flush=True)
    if args.baseline:
        base = json.loads(args.baseline.read_text())
        if len(base) != len(results):
            raise ValueError("Baseline and candidate have different case counts")
        mismatches = [r["case"] for b, r in zip(base, results)
                      if b["case"] != r["case"] or b["prompt"] != r["prompt"]
                      or (r["case"] != "sampled_fallback" and b["text"] != r["text"])]
        print("Text mismatches:", mismatches, flush=True)
        if mismatches:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
