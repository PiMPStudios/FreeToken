# Local MTP experiment

This checkout adds an opt-in `--experimental-mtp` mode for the local
`RadixArk/Qwen3.8-Flash-Next-NVFP4` and
`RedHatAI/GLM-5.3-Flash-NVFP4` checkpoints. It is an experimental implementation,
not a recommendation to migrate a working server yet.

## Isolation

`scripts/mtp-env.sh` runs the checkout's `.venv` and puts compilation caches,
downloads and temporary build files under this repository. Serving tests bind
to `127.0.0.1:18090`; they read the existing checkpoints without modifying them.
`~/ai` is read-only throughout this experiment. No commits or upstream submissions
have been made.

The environment was copied from the working FreeToken environment, its launchers
were relocated, and this checkout was installed with:

```bash
UV_CACHE_DIR="$PWD/.cache/uv" TMPDIR="$PWD/.cache/tmp" \
  XDG_CACHE_HOME="$PWD/.cache" CUDA_HOME=/usr/local/cuda \
  PATH="/usr/local/cuda/bin:$PATH" \
  uv pip install --python .venv/bin/python --no-build-isolation --no-deps -e .
bash scripts/mtp-env.sh uv pip install --python .venv/bin/python pytest
```

Hardware: RTX PRO 4500 Blackwell, 32 GB VRAM; Threadripper PRO 5975WX;
499 GiB system RAM reported by Linux; NVIDIA driver 595.84; PyTorch 2.11.0+cu130.
Only the 4500 is used. The RTX 3080 remains available to the desktop.
The starting commit is `af71ba4`.

## What the implementation does

The target's layer count and offloaded expert banks stay the same. One additional
sparse-attention cache slot belongs to the draft head. Draft experts are resident
on the GPU, independently of the NVFP4 target cache:

| Checkpoint | Draft experts | Hidden state fed to the head |
| --- | --- | --- |
| Qwen3.8-Flash-Next | BF16, stacked tensors | All four residual streams before the final mixer |
| GLM-5.3-Flash | Block-FP8, with block scales | The target's normalized output |

Qwen normalizes the complete multi-stream input, applies a shared projection to
each stream, and adds the projected next-token embedding to every stream. Its
draft decoder retains hyper-connections and QSA, with no PLE layer. GLM uses a
plain residual decoder with DSA, a concatenated embedding/hidden projection,
and a separate output norm. Its draft experts retain clamped SwiGLU, and the
embedding input is zeroed at position zero as in the reference implementation.

Prompt-time target hidden states populate the draft KV cache. The verifier runs
the current token and one draft token as a causal two-token extension. Target
MoE uses on-demand expert fetch for this short extension. On acceptance, the
engine emits the draft and saves the target's bonus token for the next scheduler
step. On rejection, it restores recurrent state, PLE state and sparse-index tail
buffers, then replays the committed target token. Ordinary scheduler processing
still handles EOS, stop strings and output accounting one token at a time.
Short verification extensions use the multi-token recurrent GDN/KDA kernels.
Qwen materializes the varlen convolution's transposed output into contiguous
channels; GLM materializes a per-token state-slot map for the KDA kernel.

Verification stays within an already allocated page. At a page boundary or the
last output token, it runs ordinary decoding. Sampled requests also use ordinary
decoding. A new request or an idle cache rebuild discards pending draft state.
For Qwen's disk PLE backend, verification stages the proposed token in a private
host history so both verification positions fetch their correct embedding rows.

## Current restrictions

- One request at a time, TP=1, text only.
- Greedy decoding (`temperature=0`, or `top_k=1`, with `top_p=1`). The normal temperature=1
  presets will not use MTP.
- `--cache-type naive --cuda-graph-max-bs 0` and
  `FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1`.
- `--moe-strategy offload`, without CPU expert layers.
- One draft token per verification; only the checkpoint layouts above have been
  audited. FTW and dummy-weight loading are not supported by this experiment.

The extra draft weights reduce space for the target expert cache. Batched
two-token verification also takes a different floating-point path from two
single-token target calls. The mode is deterministic, but a near-tied greedy
decision can therefore differ from the unchanged eager server.

## Commands

GLM candidate:

```bash
FREETOKEN_GLM5_ATTN_FP8=1 FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1 \
bash scripts/mtp-env.sh .venv/bin/ft serve \
  --model /home/tweaver/models/llm/GLM-5.3-Flash-NVFP4 \
  --host 127.0.0.1 --port 18090 --served-model-name mtp-test \
  --moe-strategy offload --nvfp4-backend auto --memory-ratio 0.90 \
  --kv-reserve-tokens 8192 --max-seq-len-override 8192 \
  --max-prefill-length 64 \
  --max-running-requests 1 --cuda-graph-max-bs 0 --cache-type naive \
  --max-output-tokens 128 --experimental-mtp
```

Qwen candidate:

```bash
FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1 \
bash scripts/mtp-env.sh .venv/bin/ft serve \
  --model /home/tweaver/models/llm/Qwen3.8-Flash-Next-NVFP4 \
  --host 127.0.0.1 --port 18090 --served-model-name mtp-test \
  --moe-strategy offload --nvfp4-backend auto --ple-backend disk \
  --memory-ratio 0.90 --kv-reserve-tokens 8192 \
  --max-seq-len-override 8192 --max-prefill-length 64 \
  --max-running-requests 1 --cuda-graph-max-bs 0 --cache-type naive \
  --max-output-tokens 128 --experimental-mtp
```

The unchanged baseline is in `.worktrees/mtp-main`, at `af71ba4`. Set
`FREETOKEN_SOURCE_ROOT="$PWD/.worktrees/mtp-main"` and omit `--experimental-mtp`
to run it with the same environment and settings. The unchanged C++ extensions
built for this environment were copied into that worktree.

Collect output and streamed timing:

```bash
bash scripts/mtp-env.sh .venv/bin/python benchmarks/mtp/compare.py \
  --output results/mtp/candidate.json --baseline results/mtp/baseline.json
```

`compare.py` saves every result before comparing full generated text. Its decode
rate uses the API's completion-token count and the interval from first to last
nonempty streamed chunk; TTFT is recorded separately. It is a small correctness
and timing probe, not a model-quality evaluation.
Use `--suite edges --repeats 1` for short output budgets, stop strings, sampled
fallback and the next greedy request. Sampled output is recorded but excluded
from exact text comparison.

For numerical diagnostics, replace `.venv/bin/ft` in the candidate command with
`.venv/bin/python benchmarks/mtp/audit_server.py`, and set
`MTP_AUDIT_OUTPUT="$PWD/results/mtp/verifier-audit.jsonl"`. The first 64 verification
rounds additionally run two sequential target steps from the same saved state.
The JSONL records logit errors, actual argmax choices, top-two margins for both
paths and recurrent-state errors. These extra forwards invalidate timing for
audited requests. Run an untimed pass before collecting speed measurements.

## Validation record

- Checkpoint header reconciliation passed: Qwen 31 tensors, GLM 1,753 tensors,
  with no unconsumed draft tensors. This checks names, shapes and dtypes without
  reading weight values. Reports: `results/mtp/{qwen,glm}-schema.json`.
- The first targeted run passed 30 tests, covering config/cache wiring,
  speculative commit/rejection, boundaries, and FP8 expert outputs against a
  dequantized PyTorch reference: `results/mtp/unit-tests.log`.
- Existing model/cache regressions passed 39 tests:
  `results/mtp/regression-tests.log`.
- The combined run passed 85 tests, including real GLM rollback across partial
  index groups and Qwen's disk PLE proposal staging:
  `results/mtp/combined-tests.log`.
- GLM's first MTP startup at `--memory-ratio 0.85` rejected the minimum cache plan
  before serving: it required 8,270,118,912 bytes against an 8,057,844,531-byte
  budget. The revised candidate uses `0.90`. Logs:
  `results/mtp/glm-mtp.log` and `results/mtp/glm-mtp-90.log`.
- The ordinary GLM server with CUDA graphs passed the six short completion
  probes. The clean measurement was approximately 7.3-8.7 decode tokens/s:
  `results/mtp/glm-baseline.json`. This is a practical graph-enabled baseline;
  the same-settings eager comparison is recorded below.
- Qwen attention/GDN/PLE regressions passed 31 tests with five skips:
  `results/mtp/qwen-attention-regressions.log`. Its separate multi-stream MTP
  conditioning test passed: `results/mtp/qwen-conditioning-test.log`.
- The revised recurrent paths passed 14 GLM/GDN checks:
  `results/mtp/recurrent-verification-tests-final.log`.
- The final targeted run passed 134 tests with five skips:
  `results/mtp/final-targeted-tests-2.log`. This includes a real-kernel test
  confirming that the numerical audit preserves the verifier's result and a
  regression for the speculative QSA layer's global cache-id map. The server
  parser regression suite separately passed 28 tests.
- The first GLM MTP implementation accepted 169 of the first 200 proposals. Its
  warm repeat measured 6.27 / 6.90 / 6.47 tokens/s (arithmetic / Python / prose).
  Unchanged `af71ba4`, with the same eager settings and memory ratio, measured
  7.75 / 7.32 / 8.59 tokens/s. Python output matched exactly; later arithmetic and
  prose continuations differed. All deterministic edge probes matched. Files:
  `results/mtp/glm-mtp.json`, `glm-eager-baseline.json`, `glm-mtp-edges.json`,
  `glm-eager-edges.json`. The initial graph baseline reported 63 tokens for a
  64-token limit; both eager runs reported 64, so use the eager control for output
  and timing comparisons.

### End-to-end result

The table uses each basic probe's warm repeat. Decode rates come from the same
streamed API client and the same eager runtime settings on unchanged `af71ba4`.

| Checkpoint | Probe | Eager tok/s | MTP tok/s | MTP / eager | Exact text |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen3.8-Flash-Next | arithmetic | 19.16 | 27.42 | 1.43x | yes |
| Qwen3.8-Flash-Next | Python | 19.00 | 28.22 | 1.49x | no, token 27 |
| Qwen3.8-Flash-Next | prose | 19.34 | 20.95 | 1.08x | yes |
| GLM-5.3-Flash | arithmetic | 7.75 | 5.93 | 0.77x | no, later branch |
| GLM-5.3-Flash | Python | 7.32 | 6.63 | 0.91x | yes |
| GLM-5.3-Flash | prose | 8.59 | 6.40 | 0.75x | no, later branch |

Qwen accepted 279 of 325 logged proposals (85.8%). Its target expert cache fell
from 7,001 eager slots to 5,112 MTP slots. All deterministic edge probes matched,
including one- and two-token budgets, stop strings and the greedy request after
sampled fallback. A 523-token prompt processed in 64-token chunks matched and
decoded at 20.92 tokens/s, versus 19.27 eager. Warm TTFT stayed near 2.43 seconds.
See `results/mtp/qwen-{eager-baseline,mtp-final}.json`, the corresponding edge and
long files, and `qwen-python-audit.jsonl`.

The corrected Qwen Python audit compared 31 two-token verification calls with
two sequential calls from the same starting state. Every local argmax matched.
At the output branch, both verification paths selected the same token with only
a 0.25 sequential and 0.125 batched logit margin. Recurrent-state relative RMS
error averaged 0.00653 and logit relative RMS error averaged 0.04549. Small state
differences accumulated across accepted batched rounds and moved this near-tied
decision relative to the full eager history. Sequentially replaying every
accepted target token would preserve the eager state but remove the speedup.

GLM accepted 265 of 325 logged proposals (81.5%). Its target expert cache fell
from 1,163 to 634 slots because its resident draft experts consume 6.75 GiB.
All edge probes matched and warm TTFT remained near 6.10 seconds. Its audit found
no reported proposal-check differences. Three bonus rows were ambiguous, all at
an exact sequential top-two tie; that older audit used `topk` rather than actual
`argmax` for the sequential label, so it cannot establish which tied token
sequential argmax would select. Logit relative RMS error averaged 0.03894 and
recurrent-state relative RMS error averaged 0.02232.

Qwen MTP is useful as an opt-in throughput experiment when deterministic but
not bit-identical near-tie output is acceptable. GLM MTP should remain disabled
on this machine: the draft consumes enough GPU cache that all three measured
probes are slower despite good acceptance. Neither result is ready to migrate
into `~/ai` without choosing that output-equivalence tradeoff explicitly.

## References

- [Upstream speculative-decoding roadmap](https://github.com/FlashML-org/FreeToken/issues/79).
- [DeepSeek DSpark PR](https://github.com/FlashML-org/FreeToken/pull/69) and
  [DFlash PR](https://github.com/FlashML-org/FreeToken/pull/258): different draft architectures.
- [GLM support PR](https://github.com/FlashML-org/FreeToken/pull/270) and its
  [experimental tuning overlay](https://github.com/Cerynitius/freetoken-ox-boost).
  The overlay's MTP path is a design reference, not applied wholesale.
- vLLM's checkpoint-specific [Qwen MTP](https://github.com/vllm-project/vllm/blob/main/vllm/models/qwen4_exp/nvidia/mtp.py)
  and [GLM MTP](https://github.com/vllm-project/vllm/blob/main/vllm/models/glm5next/nvidia/mtp.py)
  implementations were used to check conditioning and normalization conventions.
