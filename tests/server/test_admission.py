"""CPU-only tests for the HTTP admission cap (429 server_busy)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from freetoken.server.accounting import AdmissionClosedError, ServerBusyError, register_accounting_routes
from freetoken.server.api_server import FrontendManager


def _manager(*, max_running_req=2, max_queued_requests=1, queue_wait_timeout=30.0, maintenance="serving"):
    return FrontendManager(
        config=SimpleNamespace(
            served_model_name="model-a",
            max_running_req=max_running_req,
            max_queued_requests=max_queued_requests,
            queue_wait_timeout=queue_wait_timeout,
        ),
        send_tokenizer=None,
        recv_tokenizer=None,
        maintenance_state=maintenance,
    )


def test_new_user_rejects_when_running_plus_queued_cap_is_full():
    manager = _manager(max_running_req=2, max_queued_requests=1)
    assert manager.new_user() == 0
    assert manager.new_user() == 1
    assert manager.new_user() == 2  # 2 running + 1 queued
    with pytest.raises(ServerBusyError, match="cap 3") as exc_info:
        manager.new_user()
    assert exc_info.value.active == 3
    assert exc_info.value.cap == 3
    assert manager.stats.active == 3  # the rejected call never counted


def test_new_user_unlimited_queue_never_raises_busy():
    manager = _manager(max_running_req=1, max_queued_requests=-1)
    for _ in range(12):
        manager.new_user()
    assert manager.stats.active == 12


def test_new_user_still_refuses_when_engine_is_stopping():
    manager = _manager(maintenance="stopping")
    with pytest.raises(AdmissionClosedError, match="stopping"):
        manager.new_user()
    assert manager.stats.active == 0


def test_server_busy_handler_returns_429_with_retry_after():
    app = FastAPI()
    register_accounting_routes(app, lambda: None)

    @app.get("/boom")
    def boom():
        raise ServerBusyError(
            "server busy: 3 in flight, cap 3", retry_after=30, active=3, cap=3
        )

    response = TestClient(app).get("/boom")
    assert response.status_code == 429
    assert response.headers["retry-after"] == "30"
    body = response.json()["error"]
    assert body["code"] == "server_busy"
    assert body["type"] == "server_error"
    assert "cap 3" in body["message"]
