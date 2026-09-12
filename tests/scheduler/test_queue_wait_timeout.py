"""CPU-only tests for scheduler queue-wait expiry."""

from __future__ import annotations

import time
from types import SimpleNamespace

import torch
from freetoken.core import SamplingParams
from freetoken.message import ErrorReplyMsg
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.utils import PendingReq


def _pending(uid: int, *, enqueued_at: float = 0.0, chunked: bool = False) -> PendingReq:
    return PendingReq(
        uid=uid,
        input_ids=torch.arange(4, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=1),
        chunked_req=object() if chunked else None,
        enqueued_at=enqueued_at,
    )


def _scheduler(timeout: float, pending: list[PendingReq]):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SimpleNamespace(queue_wait_timeout=timeout)
    scheduler.prefill_manager = SimpleNamespace(pending_list=list(pending))
    scheduler.decode_manager = SimpleNamespace(running_reqs=[object(), object()])
    sent = []
    scheduler.send_result = sent.extend
    return scheduler, sent


def test_expire_fails_stale_waiters_and_keeps_chunked_and_unstamped():
    now = time.monotonic()
    stale = _pending(1, enqueued_at=now - 40)
    unstamped = _pending(2)  # enqueued_at=0 → never expire
    chunked = _pending(3, enqueued_at=now - 40, chunked=True)
    fresh = _pending(4, enqueued_at=now)
    scheduler, sent = _scheduler(30.0, [stale, unstamped, chunked, fresh])

    Scheduler._expire_queued_requests(scheduler)

    kept = [req.uid for req in scheduler.prefill_manager.pending_list]
    assert kept == [2, 3, 4]
    assert len(sent) == 1
    assert isinstance(sent[0], ErrorReplyMsg)
    assert sent[0].uid == 1
    assert sent[0].code == "server_busy"
    assert "30s" in sent[0].error


def test_expire_disabled_when_timeout_is_zero():
    now = time.monotonic()
    scheduler, sent = _scheduler(0.0, [_pending(1, enqueued_at=now - 1e6)])
    Scheduler._expire_queued_requests(scheduler)
    assert [req.uid for req in scheduler.prefill_manager.pending_list] == [1]
    assert sent == []
