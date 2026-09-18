"""Exercise speculative commit/rejection against a sequential recurrent toy model."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.speculative.mtp import MTPDecoder, StateSnapshot, _accept_prefix, validate_mtp_config


def _advance(state, token):
    state = (state * 3 + token + 1) % 13
    return state, (state + token) % 17


def _engine():
    pool = SimpleNamespace(conv_states=torch.zeros(1), recurrent_states=torch.tensor([2]),
                           slot_states={"ple": torch.zeros(1)})
    kv = SimpleNamespace(_pending_ring=torch.zeros(1), _tail_k=torch.zeros(1), _tail_gate=torch.zeros(1))
    return SimpleNamespace(
        linear_state_pool=pool, kv_cache=kv,
        page_table=torch.arange(128).reshape(1, -1), max_seq_len=128,
        config=SimpleNamespace(page_size=64, model_config=SimpleNamespace(model_type="glm5_next")),
        attn_backend=SimpleNamespace(prepare_metadata=lambda b: None),
        graph_runner=SimpleNamespace(mtp_verify_graph=None, can_use_cuda_graph=lambda b: False),
    )


def _fixture(monkeypatch, *, token=3, position=4, budget=8):
    eng = _engine()
    decoder = MTPDecoder(eng, None)
    req = Req(torch.tensor([1] * position + [token]), 0, position, budget, 7,
              SamplingParams(), None)
    batch = Batch([req], "decode")
    batch.padded_reqs = batch.reqs
    batch.input_ids = torch.tensor([token], dtype=torch.int32)
    decoder.uid, decoder.expected_len = req.uid, req.device_len
    calls = []

    def target(b, *, all_logits=False):
        states, logits = [], []
        for tok in b.input_ids.tolist():
            state, out = _advance(int(eng.linear_state_pool.recurrent_states[0]), tok)
            eng.linear_state_pool.recurrent_states.fill_(state)
            eng.linear_state_pool.conv_states.add_(tok)
            eng.linear_state_pool.slot_states["ple"].add_(1)
            eng.kv_cache._pending_ring.add_(tok + 1)
            eng.kv_cache._tail_k.add_(tok + 2)
            eng.kv_cache._tail_gate.add_(tok + 3)
            states.append([state])
            logits.append(torch.nn.functional.one_hot(torch.tensor(out), 17).float())
        calls.append(b.input_ids.tolist())
        return torch.stack(logits), torch.tensor(states)

    def result(r, token):
        r.complete_one()
        decoder.expected_len = r.device_len
        return token

    monkeypatch.setattr(decoder, "_target", target)
    monkeypatch.setattr(decoder, "_result", result)
    monkeypatch.setattr(decoder, "_draft", lambda *args: torch.tensor([0], dtype=torch.int32))
    return eng, decoder, req, batch, calls


@pytest.mark.parametrize("accepted", [False, True])
def test_commit_matches_sequential_state(monkeypatch, accepted):
    eng, decoder, req, batch, calls = _fixture(monkeypatch)
    state1, token1 = _advance(2, 3)
    state2, token2 = _advance(state1, token1)
    decoder.draft = torch.tensor([token1 if accepted else (token1 + 1) % 17])
    first = decoder.try_forward(batch)
    if accepted:
        assert first.tolist() == [token1, token2]
        assert len(calls) == 1
        assert int(eng.linear_state_pool.recurrent_states[0]) == state2
        consumed = [3, token1]
    else:
        assert first.tolist() == [token1]
        assert calls[-1] == [3]
        assert int(eng.linear_state_pool.recurrent_states[0]) == state1
        consumed = [3]
    assert int(eng.linear_state_pool.conv_states[0]) == sum(consumed)
    assert int(eng.linear_state_pool.slot_states["ple"][0]) == len(consumed)
    assert int(eng.kv_cache._pending_ring[0]) == sum(t + 1 for t in consumed)
    assert int(eng.kv_cache._tail_k[0]) == sum(t + 2 for t in consumed)
    assert int(eng.kv_cache._tail_gate[0]) == sum(t + 3 for t in consumed)


def test_token_pool_writes_k2_bonus_slots():
    from freetoken.scheduler.scheduler import _write_sampled_tokens

    pool = torch.zeros(1, 16, dtype=torch.int32)
    mapping = (torch.tensor([0]), torch.tensor([5]))
    _write_sampled_tokens(pool, mapping, torch.tensor([10, 11, 12], dtype=torch.int32), 1)
    assert pool[0, 5:8].tolist() == [10, 11, 12]
    assert pool[0, :5].tolist() == [0, 0, 0, 0, 0]


def test_accept_prefix_stops_at_first_mismatch():
    proposal = torch.tensor([1, 2, 3])
    assert _accept_prefix(proposal, torch.tensor([1, 2, 9, 4])) == 2
    assert _accept_prefix(proposal, torch.tensor([9, 1, 2, 3])) == 0
    assert _accept_prefix(proposal, torch.tensor([1, 2, 3, 4])) == 3
    assert _accept_prefix(proposal, torch.tensor([1])) == 0


def test_k2_full_accept_emits_drafts_and_bonus(monkeypatch):
    eng, decoder, req, batch, calls = _fixture(monkeypatch)
    decoder.k = 2
    state1, token1 = _advance(2, 3)
    state2, token2 = _advance(state1, token1)
    state3, token3 = _advance(state2, token2)
    decoder.draft = torch.tensor([token1, token2])
    first = decoder.try_forward(batch)
    assert first.tolist() == [token1, token2, token3]
    assert calls == [[3, token1, token2]]
    assert int(eng.linear_state_pool.recurrent_states[0]) == state3


def test_k2_reject_first_replays_one(monkeypatch):
    eng, decoder, req, batch, calls = _fixture(monkeypatch)
    decoder.k = 2
    _, token1 = _advance(2, 3)
    decoder.draft = torch.tensor([(token1 + 1) % 17, 0])
    first = decoder.try_forward(batch)
    assert first.tolist() == [token1]
    assert calls[-1] == [3]
    state1, _ = _advance(2, 3)
    assert int(eng.linear_state_pool.recurrent_states[0]) == state1


def test_k2_partial_accept_reverifies_prefix(monkeypatch):
    eng, decoder, req, batch, calls = _fixture(monkeypatch)
    decoder.k = 2
    state1, token1 = _advance(2, 3)
    state2, token2 = _advance(state1, token1)
    decoder.draft = torch.tensor([token1, (token2 + 1) % 17])
    first = decoder.try_forward(batch)
    assert first.tolist() == [token1, token2]
    assert calls[0] == [3, token1, (token2 + 1) % 17]
    assert calls[1] == [3, token1]
    assert int(eng.linear_state_pool.recurrent_states[0]) == state2


def test_output_boundary_uses_one_target_token(monkeypatch):
    eng, decoder, req, batch, calls = _fixture(monkeypatch, position=4, budget=1)
    _, token = _advance(2, 3)
    decoder.draft = torch.tensor([token])
    assert decoder.try_forward(batch).tolist() == [token]
    assert calls == [[3]]


def test_sampled_request_drops_speculation(monkeypatch):
    _, decoder, req, batch, calls = _fixture(monkeypatch)
    decoder.draft = torch.tensor([10])
    req.sampling_params.temperature = 0.8
    assert decoder.try_forward(batch) is None
    assert decoder.draft is None
    assert not calls


def test_snapshot_restores_in_place():
    eng = _engine()
    state = eng.linear_state_pool.slot_states["ple"]
    snapshot = StateSnapshot(eng)
    state.fill_(18)
    snapshot.restore()
    assert eng.linear_state_pool.slot_states["ple"] is state
    assert state.item() == 0


def test_qwen_verification_stages_proposed_token_for_disk_ple(monkeypatch):
    from freetoken.models.qwen4_exp.ple_disk import DiskRowTable

    _, decoder, req, _, _ = _fixture(monkeypatch)
    decoder._qwen = True
    original = req.input_ids.clone()
    batch = decoder._batch(req, 4, torch.tensor([3, 11]), verify=True)
    disk = object.__new__(DiskRowTable)
    disk.eos_token_id = 99
    staged = []
    disk.fill = lambda runs, graph: staged.extend(run.tolist() for run in runs)
    disk.host_fill_batch(batch, False)
    assert staged == [[1, 1, 3, 11]]
    torch.testing.assert_close(req.input_ids, original)


def test_top_p_request_uses_normal_sampler(monkeypatch):
    _, decoder, req, batch, calls = _fixture(monkeypatch)
    req.sampling_params.top_p = 0.9
    assert decoder.try_forward(batch) is None
    assert not calls


def test_chunked_prompt_pairs_previous_hidden_with_actual_next_token(monkeypatch):
    from freetoken.scheduler.prefill import ChunkedReq

    _, decoder, _, _, _ = _fixture(monkeypatch)
    decoder.reset()
    drafts = []

    def draft(req, start, hidden, tokens):
        drafts.append((start, hidden.flatten().tolist(), tokens.tolist()))
        return torch.tensor([0])

    monkeypatch.setattr(decoder, "_draft", draft)
    first = ChunkedReq(torch.tensor([3, 4]), 0, 0, 8, 7, SamplingParams(), None)
    b = Batch([first], "prefill")
    b.input_ids = first.input_ids
    decoder.try_forward(b)
    second = Req(torch.tensor([3, 4, 5, 6]), 0, 2, 8, 7, SamplingParams(), None)
    b = Batch([second], "prefill")
    b.input_ids = torch.tensor([5, 6])
    token = decoder.try_forward(b)
    state, hidden = 2, []
    for t in [3, 4, 5, 6]:
        state, _ = _advance(state, t)
        hidden.append(state)
    assert drafts == [(0, hidden[:1], [4]), (1, hidden[1:], [5, 6, int(token[0])])]


@pytest.mark.parametrize("field,value", [
    ("max_running_req", 2), ("cache_type", "radix"),
    ("cuda_graph_max_bs", 1), ("cuda_graph_bs", [1]),
    ("moe_strategy", "hybrid"), ("moe_cpu_layers", "1"),
])
def test_unsupported_runtime_modes_fail_before_loading(monkeypatch, field, value):
    import freetoken.env

    monkeypatch.setattr(freetoken.env, "ENV", SimpleNamespace(DISABLE_OVERLAP_SCHEDULING=True))
    config = SimpleNamespace(max_running_req=1, tp_info=SimpleNamespace(size=1),
                             cache_type="naive", cuda_graph_max_bs=0, cuda_graph_bs=None,
                             moe_strategy="offload", moe_cpu_layers=None, use_dummy_weight=False)
    validate_mtp_config(config)
    setattr(config, field, value)
    with pytest.raises(ValueError, match="Experimental"):
        validate_mtp_config(config)


def test_qwen_allows_radix_graphs_and_hybrid(monkeypatch):
    import freetoken.env

    monkeypatch.setattr(freetoken.env, "ENV", SimpleNamespace(DISABLE_OVERLAP_SCHEDULING=False))
    config = SimpleNamespace(
        max_running_req=1, tp_info=SimpleNamespace(size=1),
        cache_type="radix", cuda_graph_max_bs=1, cuda_graph_bs=None,
        moe_strategy="hybrid", moe_cpu_layers=None, use_dummy_weight=False,
        model_config=SimpleNamespace(model_type="qwen4_exp"),
    )
    validate_mtp_config(config)


def test_repeated_argmax_is_a_full_accept(monkeypatch):
    """Spaces/newlines/00 are real repeats. Identical verify rows that match the
    draft are a full accept plus bonus, not a broken-graph heuristic."""
    eng, decoder, req, batch, calls = _fixture(monkeypatch)
    _, token1 = _advance(2, 3)
    decoder.draft = torch.tensor([token1])

    def target(b, *, all_logits=False):
        calls.append(b.input_ids.tolist())
        row = torch.nn.functional.one_hot(torch.tensor(token1), 17).float()
        return torch.stack([row, row]), torch.tensor([[1.0], [1.0]])

    monkeypatch.setattr(decoder, "_target", target)
    monkeypatch.setattr(
        decoder, "_result",
        lambda r, token: token.reshape(-1).to(torch.int32),
    )
    out = decoder.try_forward(batch)
    assert out.tolist() == [token1, token1]
    assert len(calls) == 1


def test_context_clamps_when_host_lags_two_tokens():
    """Decode PLE used to IndexError at ids[device_len-2] after a 2-token MTP commit."""
    from freetoken.models.qwen4_exp.ple_disk import _context

    ids = torch.arange(4, dtype=torch.int64)
    assert _context(ids, 3, eos=99) == [1, 2]
    assert _context(ids, 4, eos=99) == [2, 3]
    assert _context(ids, 5, eos=99) == [3, 99]
    assert _context(ids, 6, eos=99) == [99, 99]
    assert _context(ids, 0, eos=99) == [99, 99]


def test_result_appends_committed_tokens_before_return(monkeypatch):
    eng = _engine()
    eng.stream = object()
    monkeypatch.setattr(torch.cuda, "Event", lambda **_: SimpleNamespace(record=lambda *_a, **_k: None))
    decoder = MTPDecoder(eng, None)
    req = Req(torch.tensor([1, 2, 3, 4]), 0, 3, 8, 7, SamplingParams(), None)
    out = decoder._result(req, torch.tensor([9, 10], dtype=torch.int32))
    assert req.input_ids.tolist() == [1, 2, 3, 4, 9, 10]
    assert req.device_len == 6
    assert out.host_appended == 2
    assert req.input_ids.numel() == req.device_len


def test_chunked_prefill_result_does_not_append_host(monkeypatch):
    from freetoken.scheduler.prefill import ChunkedReq

    eng = _engine()
    eng.stream = object()
    monkeypatch.setattr(torch.cuda, "Event", lambda **_: SimpleNamespace(record=lambda *_a, **_k: None))
    decoder = MTPDecoder(eng, None)
    req = ChunkedReq(torch.tensor([1, 2, 3, 4]), 0, 0, 8, 7, SamplingParams(), None)
    out = decoder._result(req, torch.tensor([9], dtype=torch.int32))
    assert req.input_ids.tolist() == [1, 2, 3, 4]
    assert out.host_appended == 0


def _drain_stub():
    from contextlib import contextmanager

    from freetoken.scheduler.scheduler import Scheduler

    @contextmanager
    def _region():
        yield

    sent = []
    stub = SimpleNamespace(
        cache_manager=SimpleNamespace(lazy_free_region=_region, cache_req=lambda *a, **k: None),
        decode_manager=SimpleNamespace(running_reqs=[], remove_req=lambda r: None),
        prefill_manager=SimpleNamespace(pending_list=[]),
        finished_reqs=set(),
        eos_token_ids=set(),
        toolcall_anchor_id=None,
        config=SimpleNamespace(page_size=1),
        status_reporter=SimpleNamespace(report_batch=lambda *a, **k: None),
        send_result=sent.extend,
        _kv_usage_pages=lambda: (0, 1),
        _mamba_slot_usage=lambda: None,
        _swa_token_usage=lambda: None,
        _gpu_mem_bytes=lambda: 0,
        _match_stop_str=lambda req, end=None: None,
        _free_req_resources=lambda req: None,
    )
    return stub, sent, Scheduler


def test_overlap_drain_uses_output_skip_when_accept_counts_differ():
    """Overlap runs N+1's _result (1-token reject) before draining N (2-token accept).

    A flag on the req would skip 1 while draining 2 and append N's second token
    again — host grows past device_len. The skip count must come from N's output.
    """
    from freetoken.engine.engine import ForwardOutput

    stub, sent, Scheduler = _drain_stub()
    req = Req(torch.tensor([1, 2, 3, 4]), 0, 3, 8, 7, SamplingParams(), None)
    # N: accept 2. Host already has those tokens (MTP _result).
    req.complete_n(2)
    req.append_host(torch.tensor([10, 11], dtype=torch.int32))
    # N+1 already launched: reject 1, overwriting any req-side skip flag.
    req.complete_n(1)
    req.append_host(torch.tensor([12], dtype=torch.int32))
    assert req.input_ids.tolist() == [1, 2, 3, 4, 10, 11, 12]
    event = SimpleNamespace(synchronize=lambda: None)
    batch = Batch([req], "decode")
    last_n = (
        SimpleNamespace(batch=batch),
        ForwardOutput(torch.tensor([10, 11]), torch.tensor([10, 11]), event, 2),
    )
    Scheduler._process_last_data(stub, last_n)
    assert req.input_ids.tolist() == [1, 2, 3, 4, 10, 11, 12]
    assert [m.next_token for m in sent] == [10, 11]

    sent.clear()
    last_n1 = (
        SimpleNamespace(batch=batch),
        ForwardOutput(torch.tensor([12]), torch.tensor([12]), event, 1),
    )
    Scheduler._process_last_data(stub, last_n1)
    assert req.input_ids.tolist() == [1, 2, 3, 4, 10, 11, 12]
    assert [m.next_token for m in sent] == [12]


class _FakeEvent:
    """Stand-in for torch.cuda.Event so the phase profiler is testable on CPU."""

    def __init__(self, enable_timing=False):
        self.enable_timing = enable_timing

    def record(self, stream=None):
        pass

    def query(self):
        return True

    def elapsed_time(self, other):
        return 1.0


def test_phase_profiler_accounts_a_reject_round(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MTP_PROFILE", "1")
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    eng, decoder, req, batch, calls = _fixture(monkeypatch)
    eng.stream = None
    assert decoder._prof_on
    _, expected = _advance(2, 3)
    decoder.draft = torch.tensor([(expected + 1) % 17], dtype=torch.int32)

    decoder.try_forward(batch)

    suffix = decoder._profile_suffix(decoder.rounds)
    assert "snapshot=" in suffix and "verify=" in suffix and "draft=" in suffix
    assert "restore=" in suffix and "replay=" in suffix
    assert "outcomes[" in suffix
    assert decoder._cnt["verify"] == 1 and decoder._cnt["draft"] == 1
    assert decoder._ms["verify"] == pytest.approx(1.0)
    assert decoder._outcomes["reject"] == 1


def test_phase_profiler_counts_a_full_accept(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MTP_PROFILE", "1")
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    eng, decoder, req, batch, calls = _fixture(monkeypatch)
    eng.stream = None
    state, expected = _advance(2, 3)
    decoder.draft = torch.tensor([expected], dtype=torch.int32)

    decoder.try_forward(batch)

    suffix = decoder._profile_suffix(decoder.rounds)
    assert "reverify=" not in suffix and "restore=" not in suffix
    assert decoder._outcomes["full"] == 1


def test_profiler_is_inert_when_disabled(monkeypatch):
    monkeypatch.delenv("FREETOKEN_MTP_PROFILE", raising=False)
    eng, decoder, req, batch, calls = _fixture(monkeypatch)
    assert not decoder._prof_on
    with decoder._phase("verify"):
        seen = 1
    assert seen == 1
    assert decoder._pending == []
    decoder._flush_phases()
    assert decoder._ms == {}


def test_phase_profiler_counts_a_partial_accept(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MTP_PROFILE", "1")
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    eng, decoder, req, batch, calls = _fixture(monkeypatch)
    eng.stream = None
    decoder.k = 2
    state1, token1 = _advance(2, 3)
    _, token2 = _advance(state1, token1)
    decoder.draft = torch.tensor([token1, (token2 + 1) % 17])

    decoder.try_forward(batch)

    suffix = decoder._profile_suffix(decoder.rounds)
    assert "reverify=" in suffix and "restore=" in suffix
    assert decoder._outcomes["partial"] == 1
    assert decoder._cnt["reverify"] == 1 and decoder._cnt["restore"] == 1


class _BrokenEvent(_FakeEvent):
    def record(self, stream=None):
        raise RuntimeError("out of event slots")


def test_profiling_failure_degrades_to_plain_serving(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MTP_PROFILE", "1")
    monkeypatch.setattr(torch.cuda, "Event", _BrokenEvent)
    eng, decoder, req, batch, calls = _fixture(monkeypatch)
    eng.stream = None
    _, expected = _advance(2, 3)
    decoder.draft = torch.tensor([expected], dtype=torch.int32)

    out = decoder.try_forward(batch)

    assert out is not None
    assert decoder._prof_on is False
    assert decoder._profile_suffix(decoder.rounds) == ""
    assert decoder._pending == []
