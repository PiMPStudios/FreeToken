"""MTP extra verify pages must return to the pool after a short commit.

cache_req only accounts [:cached_len]. Pages allocated for [device_len, device_len+k)
that a reject/partial never claims used to leak (idle integrity: free+cache != num_pages).
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.utils import div_ceil


def _pend(ids):
    t = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(input_ids=t, input_len=len(ids), mm_embeds=None)


def _admit(cm, table_idx, ids, output_len=16):
    handle = cm.match_req(_pend(ids)).cuda_handle
    req = Req(
        input_ids=torch.tensor(ids, dtype=torch.int32),
        table_idx=table_idx,
        cached_len=0,
        output_len=output_len,
        uid=table_idx,
        sampling_params=SamplingParams(),
        cache_handle=handle,
    )
    req.device_len = len(ids)
    cm.lock(handle)
    cm.allocate_paged([req])
    req.cached_len = len(ids)
    return req


def _extra_alloc(cm, req, k=2):
    req.mtp_extra_end_page = div_ceil(req.device_len + k, cm.page_size)
    slot = copy.copy(req)
    slot.cached_len = req.device_len
    slot.device_len = req.device_len + k
    cm.allocate_paged([slot])


def test_short_commit_releases_extra_page_radix():
    ps, n_pages = 4, 16
    page_table = torch.zeros(2, 64, dtype=torch.int32)
    cm = CacheManager(n_pages, ps, page_table, "radix")
    # 6 cached tokens (pages 0-1); pending token is still on page 1. Extra k=2
    # crosses into page 2. A 1-token commit must not keep page 2.
    req = _admit(cm, 0, list(range(6)))
    req.device_len = 7
    cm.allocate_paged([req])
    _extra_alloc(cm, req, k=2)
    assert req.mtp_extra_end_page == 3
    req.complete_n(1)
    cm.release_mtp_extra_pages([req])
    with cm.lazy_free_region():
        cm.cache_req(req, finished=True)
    cm.check_integrity()


def test_short_commit_without_release_leaks():
    ps, n_pages = 4, 16
    page_table = torch.zeros(2, 64, dtype=torch.int32)
    cm = CacheManager(n_pages, ps, page_table, "radix")
    req = _admit(cm, 0, list(range(6)))
    req.device_len = 7
    cm.allocate_paged([req])
    _extra_alloc(cm, req, k=2)
    req.complete_n(1)
    req.mtp_extra_end_page = None
    with cm.lazy_free_region():
        cm.cache_req(req, finished=True)
    try:
        cm.check_integrity()
    except RuntimeError as exc:
        assert "integrity check failed" in str(exc)
    else:
        raise AssertionError("expected a page leak without release_mtp_extra_pages")


def test_page_aligned_finish_releases_pending_page():
    """complete_n leaves device_len = cached_len+1. If that pending token is a
    new page, finish must not keep it (hybrid donate uses [:cached_len] only)."""
    ps, n_pages = 4, 16
    page_table = torch.zeros(2, 64, dtype=torch.int32)
    cm = CacheManager(n_pages, ps, page_table, "radix")
    req = _admit(cm, 0, list(range(7)))
    req.device_len = 8
    cm.allocate_paged([req])
    _extra_alloc(cm, req, k=2)
    req.complete_n(1)
    req.append_host(torch.tensor([99], dtype=torch.int32))
    assert req.cached_len == 8
    assert req.device_len == 9
    cm.release_mtp_extra_pages([req])
    with cm.lazy_free_region():
        cm.cache_req(req, finished=True)
    cm.check_integrity()


def test_full_commit_keeps_extra_page():
    ps, n_pages = 4, 16
    page_table = torch.zeros(2, 64, dtype=torch.int32)
    cm = CacheManager(n_pages, ps, page_table, "radix")
    req = _admit(cm, 0, list(range(6)))
    req.device_len = 7
    cm.allocate_paged([req])
    _extra_alloc(cm, req, k=2)
    req.complete_n(3)
    cm.release_mtp_extra_pages([req])
    with cm.lazy_free_region():
        cm.cache_req(req, finished=True)
    cm.check_integrity()


def test_short_commit_releases_extra_page_hybrid():
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate=True,
    )
    pool = LinearStatePool(
        group=g, num_slots=16, dtype=torch.bfloat16,
        device=torch.device("cpu"), tp_size=1,
    )
    ps, n_pages = 4, 16
    page_table = torch.zeros(2, 64, dtype=torch.int32)
    cm = CacheManager(n_pages, ps, page_table, "hybrid_radix", linear_state_pool=pool)
    req = _admit(cm, 0, list(range(6)))
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    req.mamba_next_track_idx = 1
    req.device_len = 7
    cm.allocate_paged([req])
    _extra_alloc(cm, req, k=2)
    req.complete_n(1)
    cm.release_mtp_extra_pages([req])
    with cm.lazy_free_region():
        cm.cache_req(req, finished=True)
    cm.check_integrity()


def test_page_aligned_finish_releases_pending_page_hybrid():
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate=True,
    )
    pool = LinearStatePool(
        group=g, num_slots=16, dtype=torch.bfloat16,
        device=torch.device("cpu"), tp_size=1,
    )
    ps, n_pages = 4, 16
    page_table = torch.zeros(2, 64, dtype=torch.int32)
    cm = CacheManager(n_pages, ps, page_table, "hybrid_radix", linear_state_pool=pool)
    req = _admit(cm, 0, list(range(7)))
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    req.mamba_next_track_idx = 1
    req.device_len = 8
    cm.allocate_paged([req])
    _extra_alloc(cm, req, k=2)
    req.complete_n(1)
    req.append_host(torch.tensor([99], dtype=torch.int32))
    cm.release_mtp_extra_pages([req])
    with cm.lazy_free_region():
        cm.cache_req(req, finished=True)
    cm.check_integrity()
