"""Experimental MTP verification for single-request greedy serving."""

from __future__ import annotations

import copy
import json
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import torch
from freetoken.core import Batch
from freetoken.models.config import FullAttentionGroupConfig
from freetoken.utils import init_logger, torch_dtype

logger = init_logger(__name__)


def _accept_prefix(proposal: torch.Tensor, tokens: torch.Tensor) -> int:
    n = 0
    limit = min(int(proposal.numel()), max(0, int(tokens.numel()) - 1))
    while n < limit and int(tokens[n]) == int(proposal[n]):
        n += 1
    return n


def with_mtp_cache(config):
    if config.model_type not in ("qwen4_exp", "glm5_next"):
        raise ValueError("Experimental MTP supports qwen4_exp and glm5_next only")
    groups = tuple(
        replace(g, layer_ids=(*g.layer_ids, config.num_layers),
                num_index_layers=g.num_index_layers + 1)
        if isinstance(g, FullAttentionGroupConfig) else g
        for g in config.attention_groups
    )
    return replace(config, attention_groups=groups)


def mtp_verify_enabled() -> bool:
    """Multi-token verify. Default on. Set FREETOKEN_MTP_VERIFY=0 to draft without verifying."""
    return os.environ.get("FREETOKEN_MTP_VERIFY", "1") != "0"


def mtp_profile_enabled() -> bool:
    """FREETOKEN_MTP_PROFILE=1 times each phase of a verify round.

    A round is verify + accept + draft + maybe restore/reverify/replay, and the
    throughput A/B alone cannot say which of those costs the tokens2x. Timings are
    CUDA-event gaps flushed on a later round, so profiling adds no sync and the
    tok/s measured with it on is still the real number.
    """
    return os.environ.get("FREETOKEN_MTP_PROFILE", "0") != "0"


MTP_PHASES = ("snapshot", "verify", "accept", "restore", "reverify", "replay", "draft")


def validate_mtp_config(config):
    from freetoken.env import ENV

    if config.max_running_req != 1 or config.tp_info.size != 1:
        raise ValueError("Experimental MTP requires --max-running-requests 1 and TP=1")
    if config.use_dummy_weight:
        raise ValueError("Experimental MTP requires real checkpoint weights")
    model_type = getattr(getattr(config, "model_config", None), "model_type", None)
    if model_type == "qwen4_exp":
        if config.moe_cpu_layers:
            raise ValueError("Experimental Qwen MTP does not support CPU expert layers")
        if config.moe_strategy not in ("offload", "hybrid", "auto"):
            raise ValueError("Experimental Qwen MTP requires --moe-strategy offload|hybrid|auto")
        k = int(getattr(config, "experimental_mtp_tokens", 1) or 1)
        if k < 1 or k > 8:
            raise ValueError("Experimental Qwen MTP requires --experimental-mtp-tokens in 1..8")
        return
    if (getattr(config, "cache_type", None) != "naive"
            or config.cuda_graph_max_bs != 0 or config.cuda_graph_bs
            or not ENV.DISABLE_OVERLAP_SCHEDULING):
        raise ValueError(
            "Experimental GLM MTP requires --cache-type naive --cuda-graph-max-bs 0 "
            "and FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1"
        )
    if config.moe_strategy != "offload" or config.moe_cpu_layers:
        raise ValueError("Experimental GLM MTP currently requires GPU expert offload without CPU layers")
    if int(getattr(config, "experimental_mtp_tokens", 1) or 1) != 1:
        raise ValueError("Experimental GLM MTP only supports one draft token")


def load_mtp(config, device):
    from freetoken.models.glm_moe_dsa.weight import _ShardReader
    from freetoken.models.register import get_model_spec

    path = Path(config.model_path)
    index = path / "model.safetensors.index.json"
    if not index.is_file():
        raise ValueError("Experimental MTP requires a local safetensors checkpoint with its index")
    hf = json.loads((path / "config.json").read_text())
    text = hf.get("text_config", hf)
    model_config = config.model_config
    if model_config.model_type == "qwen4_exp":
        from freetoken.models.qwen4_exp.mtp import Qwen4ExpMTP

        if text.get("mtp_num_hidden_layers") != 1 or text.get("mtp_use_dedicated_embeddings", False):
            raise ValueError("Qwen MTP requires one draft layer with shared embeddings")
        cls = Qwen4ExpMTP
    else:
        from freetoken.models.glm5_next.mtp import Glm5NextMTP

        if text.get("num_nextn_predict_layers") != 1 or model_config.num_layers != text["num_hidden_layers"]:
            raise ValueError("GLM MTP requires one draft layer and the complete target stack")
        cls = Glm5NextMTP
    # the draft head reuses the target layer class, so it merges the same packed parts the target's dense reader does
    packed = get_model_spec(hf["architectures"][0]).packed_modules_mapping
    with torch.device("meta"), torch_dtype(config.dtype):
        head = cls(model_config)
    reader = _ShardReader(str(path), json.loads(index.read_text())["weight_map"], torch.device("cpu"))
    try:
        head.load(reader, device, packed)
    finally:
        reader.close()
    logger.info_rank0("Loaded checkpoint MTP head with independent resident draft experts")
    return head


class StateSnapshot:
    """Save recurrent state and the partial sparse-index groups before verification."""

    def __init__(self, engine):
        pool = engine.linear_state_pool
        tensors = [pool.conv_states, pool.recurrent_states, *pool.slot_states.values()]
        kv = engine.kv_cache
        for name in ("_pending_ring", "_tail_k", "_tail_gate"):
            value = getattr(kv, name, None)
            if value is not None:
                tensors.append(value)
        self._copies = [(t, t.clone()) for t in tensors]

    def restore(self):
        for tensor, saved in self._copies:
            tensor.copy_(saved)


class MTPDecoder:
    def __init__(self, engine, head):
        self.engine = engine
        self.head = head
        self.uid = None
        self.expected_len = -1
        self.draft = None
        self.last_hidden = None
        self.rounds = 0
        self.accepted = 0
        self.accepted_tokens = 0
        self._qwen = engine.config.model_config.model_type == "qwen4_exp"
        cfg_k = int(getattr(engine.config, "experimental_mtp_tokens", 1) or 1)
        env_k = os.environ.get("FREETOKEN_MTP_K")
        k = int(env_k) if env_k else cfg_k
        self.k = 1 if not self._qwen else max(1, min(k, 8))
        self._expert_uniques: list[list[int]] = []
        self._prof_on = mtp_profile_enabled()
        self._pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._ms: dict[str, float] = {}
        self._cnt: dict[str, int] = {}
        self._outcomes = {"full": 0, "partial": 0, "reject": 0, "plain": 0}
        self._hidden_decode = self._hidden_verify = None
        if self._qwen:
            model = engine.model.model
            width = model.hc_count * engine.config.model_config.hidden_size
            dt, dev = engine.config.dtype, engine.device
            self._hidden_decode = torch.empty(1, width, device=dev, dtype=dt)
            self._hidden_verify = torch.empty(self.k + 1, width, device=dev, dtype=dt)
            model._mtp_hidden_buf = self._hidden_decode
            model._mtp_hidden_decode = self._hidden_decode
            model._mtp_hidden_verify = self._hidden_verify
            model._capture_mtp_hidden = True

    def reset(self):
        self.uid = None
        self.expected_len = -1
        self.draft = self.last_hidden = None

    def _gdn_slot(self, req):
        return req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx

    @contextmanager
    def _phase(self, name: str):
        """Time one phase on the compute stream. The gap between two events spans
        host-induced stalls too, which is the point: launch overhead is a suspect."""
        if not self._prof_on:
            yield
            return
        try:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record(self.engine.stream)
        except Exception as exc:
            self._disable_profiling(exc)
            yield
            return
        try:
            yield
        finally:
            try:
                stop.record(self.engine.stream)
                self._pending.append((name, start, stop))
            except Exception as exc:
                self._disable_profiling(exc)

    def _disable_profiling(self, exc: Exception):
        """A diagnostic must never take the serving path down with it."""
        if not self._prof_on:
            return
        self._prof_on = False
        self._pending = []
        logger.warning_rank0(f"MTP profiling disabled: {exc!r}")

    def _flush_phases(self):
        if not self._prof_on or not self._pending:
            return
        keep = []
        try:
            for name, start, stop in self._pending:
                if stop.query():
                    self._ms[name] = self._ms.get(name, 0.0) + start.elapsed_time(stop)
                    self._cnt[name] = self._cnt.get(name, 0) + 1
                else:
                    keep.append((name, start, stop))
        except Exception as exc:
            self._disable_profiling(exc)
            return
        self._pending = keep

    def _profile_suffix(self, rounds: int):
        if not self._prof_on:
            return ""
        self._flush_phases()
        parts = []
        for name in MTP_PHASES:
            if self._cnt.get(name):
                per_round = self._cnt[name] / max(1, rounds)
                mean = self._ms[name] / self._cnt[name]
                parts.append(f"{name}={per_round:.2f}x{mean:.1f}ms")
        total = sum(self._ms.get(n, 0.0) for n in MTP_PHASES) / max(1, rounds)
        outcomes = " ".join(f"{k}:{v}" for k, v in self._outcomes.items())
        return f" phase[{' '.join(parts)} sum={total:.1f}ms/round] outcomes[{outcomes}]"

    def _batch(self, req, start, tokens, *, verify=False):
        r = copy.copy(req)
        r.cached_len, r.device_len = start, start + len(tokens)
        if verify and self._qwen:
            # Disk PLE stages rows from the request's host tokens, including the proposal.
            r.input_ids = torch.cat([req.input_ids[:start], tokens.to("cpu")])
        b = Batch([r], "prefill")
        b.padded_reqs = b.reqs
        b.input_ids = tokens.to(torch.int32)
        b.positions = torch.arange(start, r.device_len, dtype=torch.int32, device=tokens.device)
        b.out_loc = self.engine.page_table[req.table_idx, start:r.device_len].clone()
        b.speculative_verify = verify
        slot = self._gdn_slot(req)
        b.linear_table_idx = torch.tensor([slot], dtype=torch.int32, device=tokens.device)
        b.active_table_idx = torch.tensor([req.table_idx], dtype=torch.int64, device=tokens.device)
        self.engine.attn_backend.prepare_metadata(b)
        return b

    def _target(self, batch, *, all_logits=False):
        model, eng = self.engine.model, self.engine
        runner = getattr(eng, "graph_runner", None)
        verify_t = int(getattr(runner, "verify_tokens", 0) or 0) if runner is not None else 0
        use_verify_graph = (
            getattr(batch, "speculative_verify", False)
            and getattr(runner, "mtp_verify_graph", None) is not None
            and int(batch.input_ids.numel()) == verify_t
        )
        use_decode_graph = (
            runner is not None and batch.is_decode and runner.can_use_cuda_graph(batch)
        )
        if use_verify_graph:
            with eng.ctx.forward_batch(batch), model.forward_host_ctx(batch, True):
                logits = runner.replay_mtp_verify(batch)
            return logits, self._hidden_verify
        if use_decode_graph:
            with eng.ctx.forward_batch(batch), model.forward_host_ctx(batch, True):
                logits = runner.replay(batch)
            hidden = self._hidden_decode if self._hidden_decode is not None else None
            return logits, hidden
        with eng.ctx.forward_batch(batch), model.forward_host_ctx(batch, False):
            if self._qwen:
                model.model._capture_mtp_hidden = True
                try:
                    output = model.model.forward(batch.input_ids, batch)
                    hidden = model.model._mtp_hidden
                finally:
                    model.model._mtp_hidden = None
            else:
                output = model.model.forward(batch.input_ids)
                hidden = output
            phase = batch.phase
            try:
                if all_logits:
                    batch.phase = "decode"
                logits = model.lm_head.forward(output)
            finally:
                batch.phase = phase
        return logits, hidden

    def _verify(self, req, start, inputs):
        verify = self._batch(req, start, inputs, verify=True)
        uniques: list[int] = []
        verify.mtp_expert_uniques = uniques
        logits, hidden = self._target(verify, all_logits=True)
        if uniques:
            self._expert_uniques.append(uniques)
        return logits, hidden

    def _draft_run(self, req, start, hidden, tokens):
        """Fill draft KV over ``tokens``. Returns last-step mixed hidden and its argmax id."""
        eng = self.engine
        mixed = token = None
        for offset in range(0, len(tokens), 64):
            t = tokens[offset:offset + 64]
            b = self._batch(req, start + offset, t)
            with eng.ctx.forward_batch(b):
                embedding = eng.model.model.embed_tokens.forward(t.long())
                mixed = self.head.forward(hidden[offset:offset + 64], embedding, b)
                token = eng.model.lm_head.forward(mixed).argmax(-1)[-1:].to(torch.int32)
        return mixed, token

    def _draft(self, req, start, hidden, tokens):
        mixed, first = self._draft_run(req, start, hidden, tokens)
        if first is None:
            return None
        if self.k == 1 or not self._qwen:
            return first
        drafts = [first]
        multi = getattr(self.head, "last_multi", None)
        if multi is None:
            return first
        multi = multi[-1:]
        t = first
        pos = start + int(tokens.shape[0])
        for _ in range(1, self.k):
            mixed, nxt = self._draft_run(req, pos, multi, t)
            drafts.append(nxt)
            multi = self.head.last_multi[-1:]
            t = nxt
            pos += 1
        return torch.cat(drafts, dim=0)

    def _result(self, req, token):
        from freetoken.engine.engine import ForwardOutput
        from freetoken.scheduler.prefill import ChunkedReq

        token = token.reshape(-1).to(torch.int32)
        req.complete_n(int(token.numel()))
        self.expected_len = req.device_len
        cpu = token.to("cpu")
        # Overlap launches the next decode before drain. Disk PLE hashes host
        # input_ids at device_len-1; a 2-token commit leaves the host two behind
        # unless we append here. Skip count rides on ForwardOutput so the next
        # step cannot overwrite it before this batch drains.
        host_appended = 0
        if not isinstance(req, ChunkedReq):
            req.append_host(cpu)
            host_appended = int(cpu.numel())
        event = torch.cuda.Event()
        event.record(self.engine.stream)
        return ForwardOutput(token, cpu, event, host_appended)

    def try_forward(self, batch):
        from freetoken.scheduler.prefill import ChunkedReq

        self._flush_phases()
        req = batch.reqs[0]
        sp = req.sampling_params
        if req.aborted or not sp.is_greedy or req.mm_items:
            self.reset()
            return None
        if req.uid != self.uid or (batch.is_decode and req.device_len != self.expected_len):
            self.reset()
        self.uid = req.uid
        if batch.is_prefill:
            logits, hidden = self._target(batch)
            token = logits[-1:].argmax(-1).to(torch.int32)
            start = req.cached_len
            embeddings = batch.input_ids[1:]
            states = hidden[:-1]
            if start > 0 and self.last_hidden is not None:
                start -= 1
                states = torch.cat([self.last_hidden, states])
                embeddings = batch.input_ids
            elif start > 0:
                # Radix prefix hit: draft KV for the cached prefix already lives in shared pages.
                embeddings = batch.input_ids[1:]
                states = hidden[:-1]
            if not isinstance(req, ChunkedReq):
                states = torch.cat([states, hidden[-1:]])
                embeddings = torch.cat([embeddings, token])
            self.draft = self._draft(req, start, states, embeddings)
            self.last_hidden = hidden[-1:].clone()
            return self._result(req, token)

        p = req.device_len - 1
        proposal = self.draft
        k = 0 if proposal is None else min(
            self.k, int(proposal.numel()), req.remain_len - 1
        )
        can_verify = (k >= 1 and req.device_len + k <= self.engine.max_seq_len
                      and mtp_verify_enabled())
        if not can_verify:
            self._outcomes["plain"] += 1
            logits, hidden = self._target(batch)
            token = logits[-1:].argmax(-1).to(torch.int32)
            with self._phase("draft"):
                self.draft = self._draft(req, p, hidden[-1:], token)
            return self._result(req, token)

        proposal = proposal[:k].to(torch.int32)
        with self._phase("snapshot"):
            snapshot = StateSnapshot(self.engine)
        with self._phase("verify"):
            logits, hidden = self._verify(req, p, torch.cat([batch.input_ids[:1], proposal]))
        with self._phase("accept"):
            tokens = logits.argmax(-1).to(torch.int32)
            n = _accept_prefix(proposal, tokens)
        self.rounds += 1
        if n == 0:
            self._outcomes["reject"] += 1
            with self._phase("restore"):
                snapshot.restore()
            with self._phase("replay"):
                logits, hidden = self._target(batch)
                token = logits[-1:].argmax(-1).to(torch.int32)
            with self._phase("draft"):
                self.draft = self._draft(req, p, hidden[-1:], token)
        else:
            self.accepted += 1
            self.accepted_tokens += n
            if n < k:
                self._outcomes["partial"] += 1
                with self._phase("restore"):
                    snapshot.restore()
                    kept = proposal[:n]
                with self._phase("reverify"):
                    logits, hidden = self._verify(
                        req, p, torch.cat([batch.input_ids[:1], kept])
                    )
                    tokens = logits.argmax(-1).to(torch.int32)
            else:
                self._outcomes["full"] += 1
            hidden = hidden.clone()
            token = tokens[: n + 1]
            with self._phase("draft"):
                self.draft = self._draft(req, p, hidden[: token.numel()], token)
        if self.rounds == 1 or self.rounds % 25 == 0:
            acc = self.accepted / self.rounds
            mean_len = self.accepted_tokens / self.rounds
            extra = ""
            if self._expert_uniques:
                last = self._expert_uniques[-1]
                topk = int(getattr(self.engine.config.model_config, "num_experts_per_tok", 0) or 0)
                extra = f" unique_experts={sum(last) / len(last):.1f}/{topk * (k + 1)}"
            if self._prof_on:
                extra += self._profile_suffix(self.rounds)
            logger.info_rank0(
                f"MTP k={self.k} rounds={self.rounds} accepted={self.accepted} "
                f"acceptance={acc:.3f} mean_drafts={mean_len:.2f}{extra}"
            )
        return self._result(req, token)
