# MTP experiment handoff

Two passes are recorded here.

The **2026-09-07 pass** built and measured the local implementation (sections
below through "Current live progress", plus the commands at the end).

The **2026-09-10 pass** is a read-only survey of upstream and of the engine's own
constraints: "Upstream state", "Analysis", "Ranked ideas", "The measurement to
run", "Other engine constraints found", "Suggested PR sequence". It changed no
code and ran no test, server or benchmark. Every throughput and acceptance figure
in those sections is *quoted* from the 2026-09-07 pass or from GitHub comments,
not re-measured; the implied verify costs are arithmetic on those quoted numbers.

Things the 2026-09-10 pass could not check:

- Whether the 1.43x / 0.85x A/B numbers reproduce. It only confirmed the
  `MTP rounds=325 accepted=265` log lines exist and that the verify path was
  enabled (no `FREETOKEN_MTP_VERIFY=0` is set anywhere, despite a comment in
  `speculative/mtp.py:31-34` saying the launcher sets it).
- PR #69 / #258 line numbers, which are computed from diff hunk headers rather
  than checked-out files.
- Prior art. There is no web tooling in this environment, so there is **no
  verified vLLM/SGLang comparison in this document**. The open questions worth
  looking up by hand: whether a hybrid GDN/KDA target stores recurrent state per
  draft position or re-runs prefill on rejection; how verify batches are padded
  into CUDA graphs; and whether published MTP acceptance lengths are greedy-only,
  since engines commonly gate or tune speculation on temperature.

## VERDICT (measured 2026-09-10, third pass - this supersedes the 1.43x claim)

A third pass added a per-phase CUDA-event profiler to the verify round
(`FREETOKEN_MTP_PROFILE=1`, in `python/freetoken/speculative/mtp.py`) and ran the
fixed probe set on one instance, same flags, cold-vs-cold. Data in
`results/mtp/probes/ab-2026-09-10.json` and `phase-breakdown-2026-09-10.json`.

**MTP k=2 is a net loss on every probe, and it loses even with a perfect drafter.**

| probe | MTP off | MTP on k=2 | ratio |
|---|---|---|---|
| arithmetic | 31.09 | 25.86 | 0.83x |
| python | 35.66 | 25.33 | 0.71x |
| prose | 37.73 | 20.29 | 0.54x |
| repetition | 24.89 | 23.32 | 0.94x |
| stop_string | 8.08 | 7.80 | 0.97x |

Greedy output also diverges on 4 of 5 probes. Enabling MTP also costs **41% of the
expert cache** (`moe_cache_size` 4987 -> 2954; KV grew only 0.39 GiB, so ~4.98 GiB
is resident BF16 draft experts).

Phase breakdown, ms per round (acceptance 0.802, 2.43 tokens/round, round = 119.8 ms
= 4.52 single steps):

| phase | ms/round | share |
|---|---|---|
| verify | 72.4 | 60% |
| reverify (partial accept) | 10.5 | 9% |
| draft | 7.4 | 6% |
| replay (reject) | 4.1 | 3% |
| snapshot | 2.8 | 2% |
| restore | 1.0 | 1% |
| accept | 0.1 | 0% |
| *outside the forward* | 21.5 | 18% |

The arithmetic that settles it: break-even is 2.43 x 26.5 = 64.4 ms/round; the
non-verify work is already 47.4 ms, leaving a **17.0 ms budget for a 3-row verify
when a single 1-row forward costs 20.3 ms**. Assume a drafter that is never wrong
(3 tokens/round, no reverify/replay): round = 104.2 ms against a 79.5 ms budget,
**0.76x**. k=1 lands at 0.72-0.94x. Neither wins.

The cost shape says the problem is not per-row scaling: 1 row 20.3 ms, 2 rows
62.0 ms, 3 rows 72.4 ms. **Entering the verify path costs ~60 ms fixed, ~10 ms per
extra row.** A verify forward is ~3x a decode step whatever k is.

Ruled out by measurement, so nobody re-investigates them: the serial draft loop
(7.4 ms, 6%), the `int()` sync in `_accept_prefix` (0.1 ms), and whole-pool
snapshot/restore, idea P3 (3.8 ms combined). Ideas A-H in the section below were
ranked against a model this pass falsified.

Still open, in priority order: (1) why a verify forward costs ~60 ms fixed against
20.3 ms for a decode step - `speculative_verify` is already special-cased in
`layers/moe.py`, `layers/embedding.py`, `kernel/triton/nvfp4_linear.py`,
`attention/qsa_sparse.py` and `models/qwen4_exp/ple_disk.py`, so it does not bypass
the expert cache; this needs a kernel-level profile. (2) The 21.5 ms/round spent
outside the model forward, unattributed. (3) Whether `MTPDecoder._batch`'s
`torch.cat` of the whole host token list per verify (disk PLE staging) makes the
72.4 ms grow with context.

**Practical consequence: run with MTP off.** The value of this work is now the
measurement, not the feature.

## Objective and boundaries

Add opt-in multi-token prediction (MTP) to this FreeToken checkout for these
existing local checkpoints:

- `/home/tweaver/models/llm/Qwen3.8-Flash-Next-NVFP4`
- `/home/tweaver/models/llm/GLM-5.3-Flash-NVFP4`

All code, build products, caches, tests, and results stay in
`/home/tweaver/Developer/GitRepos/FreeToken`. The working configuration under
`/home/tweaver/ai` is read-only and must not be changed. Do not migrate anything
to the working setup until the experiment is correct and the user asks for that
separate step. Do not commit, push, create a PR, or post to GitHub unless the user
explicitly asks; repository policy also forbids agents from doing the remote
operations.

The initial supported scope is deliberately narrow: TP=1, one running request,
greedy text generation, naive cache, CUDA graphs off, overlap scheduling off,
GPU expert offload, and one draft token per verification round.

## Current implementation

The checkout has an `--experimental-mtp` flag and checkpoint-specific draft
heads for Qwen3.8-Flash-Next and GLM-5.3-Flash. The target model performs a
two-token causal verification extension, accepts a matching draft, queues the
target bonus token, or restores recurrent/index state and replays the committed
token after rejection.

The implementation includes:

- checkpoint parsing and cache-group extension for one draft attention layer;
- resident BF16 Qwen draft experts and resident block-FP8 GLM draft experts;
- Qwen four-stream conditioning before its final hyper-connection mixer;
- GLM normalized target output conditioning, position-zero embedding masking,
  and clamped SwiGLU for its FP8 experts;
- rollback of recurrent, convolution, PLE, and partial sparse-index state;
- short multi-token recurrent GDN/KDA verification paths;
- Qwen disk-PLE proposal staging using a private host-token history;
- ordinary FreeToken fallback for sampled, multimodal, page-boundary, and
  final-token cases;
- local schema-audit, comparison, edge-case, long-prompt, and numerical-audit
  tools under `benchmarks/mtp/`.

The main design and exact commands are documented in
`docs/mtp-experiment.md`. The isolated runner is `scripts/mtp-env.sh`. A detached
unchanged checkout at `.worktrees/mtp-main` provides the `af71ba4` baseline.

## Evidence collected

- Both local checkpoint schemas reconcile with the draft heads: Qwen has 31 MTP
  tensors and GLM has 1,753, with no unused draft tensors. See
  `results/mtp/qwen-schema.json` and `results/mtp/glm-schema.json`.
- The latest targeted run passed 134 tests with five skips. It covers cache
  wiring, draft conditioning, FP8 expert math, recurrent verification, real GLM
  rollback, Qwen PLE, attention backends, output boundaries, and the numerical
  audit harness. The parser regression suite separately passed 28 tests. See
  `results/mtp/final-targeted-tests-2.log`.
- The first GLM MTP run loaded and served successfully, accepting 169 of its
  first 200 proposals (84.5%). Its warm repeat measured 6.27, 6.90, and 6.47
  tokens/s on the arithmetic, Python, and prose probes.
- The unchanged eager GLM baseline measured 7.75, 7.32, and 8.59 tokens/s on the
  same probes. Python output matched exactly. Later arithmetic and prose text
  diverged, so the first implementation was functional but not yet accepted as
  numerically equivalent. All deterministic edge probes matched, including
  one/two-token limits, stop strings, and greedy generation after sampled
  fallback.
- `--memory-ratio 0.85` is too small for GLM's draft head plus the minimum cache
  plan on the 32 GB RTX PRO 4500. The test command now uses `0.90`.
- No new model download has been needed. Both existing checkpoints contain the
  required MTP weights.

## Current live progress

The experimental implementation and its local validation are complete. No
experiment server is active. `git diff --check` and `compileall` pass; Ruff is
not installed in the isolated environment. The latest targeted suite passed 134
tests with five skips, and the server parser suite passed another 28 tests.

Qwen is the successful performance result. Warm MTP throughput was 27.42, 28.22,
and 20.95 tokens/s for arithmetic, Python, and prose, versus eager 19.16, 19.00,
and 19.34 tokens/s (1.43x, 1.49x, and 1.08x). It accepted 279 / 325 proposals
(85.8%). Its target expert cache fell from 7,001 to 5,112 slots. Arithmetic,
prose, all edge cases, and the 523-token chunked-prefill probe matched exactly;
Python was deterministic but branched at generated token 27 / character 98.
See `results/mtp/qwen-mtp-final.json`, its edge/long companions, and
`qwen-python-audit.jsonl`.

The corrected Qwen Python audit covered 31 rounds and found zero local
sequential-vs-batched argmax mismatches. At the divergent branch, both local
paths selected the same token with margins of 0.25 sequential and 0.125 batched.
Accepted batched rounds accumulate small floating-point state differences and
can therefore move a near-tied decision relative to the full eager history.
Bit-identical eager state would require sequentially replaying every accepted
target token, which removes the target-batching speed benefit.

GLM is functional but slower. Warm MTP throughput was 5.93, 6.63, and 6.40
tokens/s versus eager 7.75, 7.32, and 8.59. It accepted 265 / 325 proposals
(81.5%), but its 6.75 GiB resident draft experts reduced the target expert cache
from 1,163 to 634 slots. Python and all edge cases matched; arithmetic and prose
branched later. Its older audit reported no proposal-check differences and three
bonus differences at exact sequential top-two ties. Because that version used
`topk` for the sequential label, those tied rows are ambiguous. See
`results/mtp/glm-mtp-recurrent.json` and `glm-verifier-audit.jsonl`.

The tested recommendation is to use Qwen MTP only when deterministic output
with possible near-tie differences is acceptable, and to leave GLM MTP disabled
on this hardware. `docs/mtp-experiment.md` contains the exact commands, A/B
table, implementation detail, and limitations. Nothing was changed under
`~/ai`, no model download was needed, and no commit was made.

## Upstream state (surveyed read-only 2026-09-10, nothing posted)

Speculative decoding is on the Roadmap (#79: "Speculative decoding, including
MTP / DFlash / Dspark"). CONTRIBUTING requires discussing Roadmap features with
maintainers in the Developer Slack *before* implementing.

| Item | State | Note |
| --- | --- | --- |
| PR #69 DSpark (DSV4, +5360/53 files) | open, `dirty`, PR 2 of 3 | the only greedy+sampled-exact design in flight |
| PR #258 DFlash (+3147/21 files) | open, `dirty`, 1 commit | separate diffusion drafter, `--speculative-algorithm dflash` |
| Issue #421 "MTP for qwen4_exp" | open, **0 replies** | asks exactly whether #69 generalises; unanswered |
| Issue #173 | open | verify graph captured for bs [1,2,4]; bs=3 falls to eager, 22 -> 11 tok/s |

- **Zero maintainer reviews or comments on either PR.** No labels, no reviewers.
  There is no upstream abstraction to conform to yet, and two incompatible ones
  are competing: #69 uses a module-global `set_dspark_enabled()` + 7 new `Batch`
  fields + `Req.complete_n`; #258 uses a real `speculative/BaseSpecWorker` ABC +
  factory + `--speculative-algorithm`. #258's shape is the pluggable host; #69's
  acceptance/rollback semantics are the correct payload. They conflict today.
- `main` itself has **no** spec-decode runtime: no `speculative/` package, no
  `--speculative-*` flags. The local tree is the only MTP on qwen4_exp/glm5_next.
- **This checkout is behind `origin/main`.** PR #427 (`checkpoint_quant_config()`
  / `QuantConfig` handed to weight readers) is merged upstream and absent here
  (`git grep QuantConfig HEAD` = 0). Rebase before touching the loader, and home
  any `draft_expert_quant` in that new `QuantConfig` rather than adding a third
  quant string.
- Do not imitate #258: it disables `cache_manager.check_integrity()` to mask a
  known 2-page/request leak, and gates speculation on `batch.size == 1`.
- Portable from #69 regardless of which interface wins: `accepted_prefix`,
  `sampling_probs`, `rejection_accept` (min(1,p/q) + residual resample), the
  `(1+k)*len(batch.reqs)` logits-row guard, the `max_device_len - start - 1`
  budget clamp, `release_tail` before `device_len` drops, and a **per-token
  EOS/stop scan inside the block**. Key invariant, verbatim: the draft must
  sample from q, not argmax, or the output is no longer the target distribution.

## Analysis: why MTP wins or loses here

Mean tokens per round = `1 + acceptance`; throughput ratio = that divided by the
verify cost expressed in eager steps. Both local models and upstream's failing
PR are the same mechanism:

| Case | acceptance | tok/round | measured ratio | implied verify cost |
| --- | --- | --- | --- | --- |
| Qwen3.8-Flash-Next (local) | 0.858 | 1.858 | 1.43x | ~1.30 eager steps -> win |
| GLM-5.3-Flash (local) | 0.815 | 1.815 | 0.85x | ~2.14 eager steps -> strict loss |
| PR #69 on 2x RTX 6000 Ada, TP2 (community tester gdevenyi, 39.25 vs 35.91 tok/s) | 0.42 | 1.42 | 0.91x | ~1.56, acceptance is what dies |

The whole feature reduces to one target: **make a k-token verify cost under
~1.8 eager steps.** GLM needs a 20% cut, and it is available from slot headroom.

Why slots dominate: the expert cache is one unified pool keyed by flat
`layer_id*num_experts + expert` (`moe/offload_cache.py:160-171`), so
`slots/layer = cache_size / num_moe_layers`. Resident draft experts are charged
1:1 in bytes against `net_cache_budget_bytes` (`engine/cache_budget.py:31-38`).
Qwen: 4.88 GiB -> 7001->5112 slots -> 106/layer vs top_k=10, misses barely move.
GLM: 6.98 GiB -> 1163->634 -> 15.1/layer vs top_k=8, misses saturate; each fp8
draft expert evicts 1.8 NVFP4 target experts. gdevenyi hit the identical
mechanism on #69 (host pool 143->153 GiB shrank `moe_cache_size` 5622->5551) and
called it "structural rather than a tuning problem"; the #69 author concedes
"exactness does not imply a workload-independent speedup".

Ceiling at k=1 with 86% acceptance is 1.86x; Qwen measured 1.43x, so ~25% is
still on the table from cheaper verifies and larger k.

## Ranked ideas

A. **Draft experts as banks inside the offload cache, not resident tensors.**
   Removes the tax that flips GLM's sign and sank #69's numbers. Three verified
   blockers: `_bank_layer` range guard `models/nvfp4_banks.py:50-60` rejects
   `layer >= num_moe_layers`; `assert len(layers) == num_moe_layers`
   (`engine/engine.py:643`) means the draft `OffloadMoELayer` must be built after
   the target stack; and `moe/offload_cache.py:305-307` allows one `quant_format`
   / one `_BANK_SCHEMAS` per cache while drafts are bf16 (Qwen) / block-FP8 (GLM)
   against NVFP4 targets. Either a mixed-format schema or requantize the 1-layer
   draft into the target layout (a rounding error on proposals, and it makes the
   draft a free bank). Cheap first variant: keep draft dense/attention resident,
   stream only draft experts.

B. **Verify through the prefill/chunk MoE path, not the decode path.**
   `layers/moe.py:228,248` forces verify onto `_decode_routed`'s on-demand fetch,
   opting out of `_prefill_routed`'s whole-layer double-buffered stream, which is
   the regime that actually amortises PCIe. Missing experts are already deduped
   per layer (`offload_kernels.py:331-343` folds `tokens*top_k` into one bitmask),
   so Qwen k=2 is K=20 against a 128-entry admission cap; spill only past k>12
   Qwen / k>16 GLM. #69 deliberately gave this up by choosing short-decode hybrid.

C. **Stop snapshotting the whole pool.** `StateSnapshot` clones
   `conv_states + recurrent_states + *all* slot_states + index rings every round
   (`4*mr + ceil(2*mr) + 1` slots, ~2.7 GiB/step at mr=4). Rollback itself was
   never the problem: one save+restore is ~0.11 ms (Qwen) / ~0.14 ms (GLM). Use
   per-slot ping-pong via the existing `pool.copy_from` COW primitive, or skip
   rollback entirely with `fused_recurrent_kda`'s `ssm_state_indices [N,T]`
   per-position state store (state at every accept length in one kernel; accept
   becomes an `index_copy_`). `SlotStateSpec`/`slot_states` is the declared seam.

D. **Free GLM's prefill double buffer.** 2E=576 of its 1163 slots
   (`cache_budget.py:71-73`) are the prefill double buffer, ~49% of the pool not
   decode-resident at all. A decode-heavy spec mode that shrinks it buys back
   ~15 slots/layer, roughly the whole 15.1 -> 19-20 gap.

E. **Re-enable hybrid / CPU-layer MoE under speculation.**
   `validate_mtp_config` forbids `moe_cpu_layers` and forces `--moe-backend
   offload`, forfeiting `moe_hybrid_max_fetch`, the one mechanism that makes a
   verify cheaper than k steps when the cache misses - exactly the GLM case.

F. **Sampled acceptance** (port #69's `rejection_accept` + residual). Greedy-only
   will not survive review.

G. **Verify CUDA graphs keyed `(bs, k)` including odd sizes.** Graphs bake
   `T == bs` (`max_q_len=1`, `cu_seqlens=arange(bs+1)`), so verify has no graph
   today; the local verify graph is hard-disabled (`use_verify_graph = False and
   (...)`) because it cloned last-token logits onto both rows. Issue #173 is the
   cliff when a size is missing from the grid; #258 shows the other failure, where
   per-len GDN snapshot buffers OOM at capture for block 16.

H. **k>1** is where the remaining ~25% lives, but only after the measurement below.

## The measurement to run before any of A-H

Nothing in-tree measures **distinct-activated-expert growth across k adjacent
tokens**; `benchmarks/bench_offload_cache_copy.py:162` assumes
`active_unique = min(E, bs*topk)`, i.e. zero overlap. That number *is* the entire
benefit of verification. Small GPU job: log the per-layer active-expert bitmask
union at k=1,2,3,4,8 on real prompt streams for both checkpoints. If adjacent
tokens share 60%+ of experts, verify is cheap and B/D/E are worth doing; if they
share almost nothing, MTP on this offload path is a dead end at any k for every
model. Either result is worth having before writing engine code.

## Other engine constraints found (main, `git show HEAD:...` refs)

- **Overlap correctness rests on `complete_one()` being an unconditional +1**
  (`engine/engine.py:934-935`, `core.py:88-90`). It runs inside the launch, so
  batch N+1's metadata is final before N's tokens reach the host, and values flow
  device-side `token_pool -> token_pool` (`scheduler/scheduler.py:868,872`). A
  verify makes the advance conditional on a host comparison; deferring commit
  means either a `copy_done.synchronize()` ahead of the next launch (kills the
  overlap) or speculative page allocation plus rollback.
- Host staging is sized in *requests*, not tokens: CPU-MoE
  `max_tokens = max(max_running_req, cuda_graph_max_bs)` (`engine/engine.py:698-700`),
  disk-PLE pinned rows and its n-gram context (`models/qwen4_exp/ple_disk.py:144-159,
  204-224`). That is why the local code hand-appends host tokens.
- Sparse index tiers are **self-healing**: QSA/DSA compressed rows are a pure
  function of position, so a rejected write is stale-but-unreachable (scoring
  clamps to `kvlen // ratio`). The invariant that must survive any rewind is
  `page_size % ratio == 0` (page_size 64). `ring_capacity_for(index_ratio,
  num_speculative_tokens)` already exists in `kvcache/qsa_pool.py:47-49`.
- Paged KV + radix are near-free: at page_size 64 a k-token draft usually
  allocates nothing, and radix insertion reads only `req.cached_len`, so keeping
  `cached_len` on the accepted frontier hides rejected tokens from the tree.
- Two silent-corruption traps: GDN snapshots fire mid-forward at x64 boundaries
  and are *donated to the radix tree* (`scheduler/cache.py:340-420`), so a verify
  crossing 64 can freeze rejected tokens into a slot another request COW-restores;
  and Qwen disk-PLE commits an n-gram window containing rejected ids, which
  produces a *valid* hash and wrong rows with no crash.
- Weight surface: qwen4_exp drops `mtp.*` explicitly at
  `models/qwen4_exp/weight.py:103`; glm5_next never requests it (pull-by-name
  over `range(config.num_layers)`, `models/glm5_next/weight.py:216`).
  `weight_map` is not the obstacle - `mtp.*` keys are in the index and
  `_ShardReader` is a pure name lookup. Draft KV slot: `attention_group_for_layer`
  needs exactly one owner, so the draft layer must be appended to a group; that
  works for DSA (`kvcache/dsa_pool.py:48-60` sizes from `layer_ids`, no bound
  check, which is why GLM appears to work) but raises for QSA/MHA
  (`kvcache/mha_pool.py:44-47` sizes `layer_map` from `num_layers`). The
  upstream-shaped fix is to size MHA/QSA from `layer_ids`, not to hack
  `kvcache/__init__.py` as this tree currently does. MoE still needs #69's
  `extra_moe_layers` id continuation because the cache sizes from `num_moe_layers`.
- **FTW is a hard wall**: converted checkpoints contain only what the readers
  yielded, so `mtp.*` is physically absent from every existing FTW, `.ftw` replay
  is model-agnostic, and adding a draft layer trips the bank-count cross-check
  (`checkpoint/ftw.py:473-480`). Dummy weights need a `dummy_moe_expert_sources`
  override. Any upstream MTP must state this and probably add an FTW re-convert path.

## Suggested PR sequence (if this goes upstream)

1. `fix(kvcache)`: size the MHA/QSA `_layer_map` from `layer_ids`, as `dsa_pool`
   already does. Small, obviously correct, needed by MTP, reviewable alone.
2. Cache PR: draft-bank addressing (range guard, build order, mixed-format schema).
3. `speculative/` worker + MTP head, `--speculative-algorithm mtp` plus a
   `num_speculative_tokens` knob. gdevenyi's literal complaint on #69 was "if there
   is a knob for block size... I did not see one exposed".
4. Sampled acceptance.
5. Verify graph capture over the 2-D `(bs, k)` grid.

Every PR should report acceptance, slot tax and verify cost together, because the
reviewers' reference point for this feature is now "speculation made it slower".
Per repo policy an agent must not post; issue #421 needs a human reply, and the
abstraction question in it is one only the maintainers can answer.

## Plan from here

0. Run the distinct-expert-overlap measurement above; it decides whether A-H are
   worth anything. Then rebase onto `origin/main` for PR #427's `QuantConfig`.
1. Have the user review the Qwen output-equivalence caveat and the deliberately
   narrow runtime restrictions before any migration decision.
2. If bit-identical eager output is mandatory, prototype an exact sequential
   state-commit mode and benchmark it; expect the current Qwen speedup to vanish.
3. If GLM acceleration remains a goal, investigate a smaller draft-weight format
   or another cache strategy. Re-run the unchanged eager A/B because target
   expert-cache pressure is the measured bottleneck.
4. Migrate into `~/ai` only as a separate user-requested task. Do not commit or
   publish these changes unless the user explicitly asks.

## Useful commands

Run the targeted tests:

```bash
bash scripts/mtp-env.sh .venv/bin/python -m pytest \
  tests/engine/test_mtp.py tests/kvcache/test_qsa_pool.py \
  tests/models/test_glm5_next_config.py \
  tests/models/test_glm5_next_model.py tests/models/qwen4_exp/test_config.py \
  tests/models/qwen4_exp/test_skeleton.py tests/models/qwen4_exp/test_ple_disk.py \
  tests/models/qwen4_exp/test_qsa_backend.py tests/models/qwen4_exp/test_qsa_hf.py \
  tests/models/qwen4_exp/test_gdn.py tests/models/qwen4_exp/test_ple.py \
  tests/kernels/test_fp8_blockscale_moe.py tests/engine/test_cache_budget.py \
  tests/models/test_glm5_next_kda_snapshot.py -m 'not slow' -q
```

Never kill unrelated Python processes. Resolve the experiment frontend PID from
its exact `ft serve` or `audit_server.py serve` command and send `SIGINT` so the
pinned expert banks are released cleanly.
