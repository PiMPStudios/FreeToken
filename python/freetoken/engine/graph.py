from __future__ import annotations

import copy
import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from freetoken.attention.linear import FLAMetadata
from freetoken.core import Batch, Req, get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.utils import init_logger, mem_GB
from freetoken.utils.progress import emit_progress
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.models import BaseLLMModel
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    # [3, bs] t/h/w rope positions; allocated only for mrope models (else None).
    mrope_positions: torch.Tensor | None
    logits: torch.Tensor
    table_idx: torch.Tensor  # per-request slot id for GatedDeltaNet state gather/scatter
    # Decode GDN query indptr = arange(bs+1); a constant per captured bs, filled once.
    fla_cu_seqlens: torch.Tensor

    @classmethod
    def init(
        cls, bs: int, vocab_size: int, device: torch.device, mrope: bool = False
    ) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            mrope_positions=(
                torch.zeros(3, bs, dtype=torch.int32, device=device) if mrope else None
            ),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
            table_idx=torch.zeros(bs, dtype=torch.int32, device=device),
            fla_cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        from freetoken.attention.linear import FLAMetadata

        _slice = slice(batch.padded_size)
        bs = batch.padded_size
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]
        if self.mrope_positions is not None:
            batch.mrope_positions = self.mrope_positions[:, _slice]
        batch.linear_table_idx = self.table_idx[_slice]
        # Decode GDN metadata reads the persistent cu_seqlens (constant arange) and the
        # persistent table_idx slot map, so the captured kernels see stable addresses.
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.fla_cu_seqlens[: bs + 1], cache_indices=self.table_idx[_slice]
        )

    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        if batch.out_loc is not None:
            self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions
        if self.mrope_positions is not None:
            self.mrope_positions[:, _slice] = batch.mrope_positions
        if batch.linear_table_idx is not None:
            self.table_idx[_slice] = batch.linear_table_idx


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    candidates = [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
    return [bs for bs in candidates if bs <= cuda_graph_max_bs]


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        moe_offload_cache: OffloadMoeCache | None = None,
        mrope: bool = False,
        mtp_verify: bool = False,
        mtp_verify_tokens: int = 2,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.moe_offload_cache = moe_offload_cache
        self.mrope = mrope
        self.stream = stream
        self.device = device
        self.mtp_verify_graph = None
        self.verify_buffer = None
        self.verify_tokens = max(2, int(mtp_verify_tokens))
        self._capture_graphs(max_seq_len, vocab_size, model, mtp_verify=mtp_verify)

    def _reset_moe_offload_cache(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.reset()

    def _capture_graphs(
        self, max_seq_len: int, vocab_size: int, model: BaseLLMModel, *, mtp_verify: bool = False
    ):
        # Mark the post-weights "warmup" phase for /health: this stretch (graph capture — or the
        # remaining readiness work when graphs are disabled) moves no bytes, so without this the
        # loader would sit at 100% (last byte bar) until the ready ack. total=0 ⇒ the desktop
        # reads it as an indeterminate phase and animates the bar. Must precede the
        # graphs-disabled early return so that config gets the phase too.
        emit_progress("Capturing CUDA graphs / warming up", 0, 0)
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(
            max_seq_len=max_seq_len,
            bs_list=self.graph_bs_list,
            verify_tokens=self.verify_tokens,
        )

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(
            self.max_graph_bs, vocab_size, self.device, mrope=self.mrope
        )
        self._reset_moe_offload_cache()

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            # capture on the dummy linear-state slot so GatedDeltaNet gather/scatter
            # touches scratch (real slot indices are written by copy_from on replay). Hybrid-
            # radix decouples the GDN slot from table_idx -> use the GDN padding slot.
            dummy_slot = (self.dummy_req.linear_slot_idx
                          if self.dummy_req.linear_slot_idx is not None
                          else self.dummy_req.table_idx)
            self.buffer.table_idx[:bs].fill_(dummy_slot)
            with get_global_ctx().forward_batch(batch):
                self.buffer.logits[:bs] = model.forward()
                # Keep the offload cache warmed for capture. Resetting here forces
                # CUDA graph capture to replay cold-cache expert copies.
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
                self._reset_moe_offload_cache()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph

        if mtp_verify and self.max_graph_bs > 0:
            self._capture_mtp_verify(model, vocab_size, pool)

        self._reset_moe_offload_cache()
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def _capture_mtp_verify(self, model: BaseLLMModel, vocab_size: int, pool) -> None:
        """Capture a 1-request, T-token speculative-verify graph of the target model."""
        t = self.verify_tokens
        inner = getattr(model, "model", None)
        decode_buf = getattr(inner, "_mtp_hidden_decode", None) if inner is not None else None
        verify_buf = getattr(inner, "_mtp_hidden_verify", None) if inner is not None else None
        if verify_buf is not None:
            inner._mtp_hidden_buf = verify_buf
        dummy = copy.copy(self.dummy_req)
        dummy.cached_len = 1
        dummy.device_len = 1 + t
        dummy.max_device_len = 2 + t
        dummy.input_ids = torch.zeros(dummy.max_device_len, dtype=torch.int32)
        batch = Batch([dummy], "prefill")
        batch.padded_reqs = batch.reqs
        batch.speculative_verify = True
        self.verify_buffer = GraphCaptureBuffer.init(
            t, vocab_size, self.device, mrope=self.mrope
        )
        buf = self.verify_buffer
        buf.input_ids.zero_()
        buf.positions.copy_(
            torch.arange(1, 1 + t, dtype=torch.int32, device=self.device)
        )
        # mrope models feed [3, n] positions; for text-only verify all three t/h/w rows
        # equal the sequence index (the captured values are placeholders, replay overwrites).
        if buf.mrope_positions is not None:
            buf.mrope_positions.copy_(buf.positions.unsqueeze(0).expand(3, -1))
        dummy_slot = int(get_global_ctx().page_table[dummy.table_idx, 0].item())
        buf.out_loc.fill_(dummy_slot)
        slot = (
            dummy.linear_slot_idx if dummy.linear_slot_idx is not None else dummy.table_idx
        )
        buf.table_idx[:1].fill_(slot)
        batch.input_ids = buf.input_ids
        batch.out_loc = buf.out_loc
        batch.positions = buf.positions
        if buf.mrope_positions is not None:
            batch.mrope_positions = buf.mrope_positions
        batch.linear_table_idx = buf.table_idx[:1]
        batch.active_table_idx = torch.tensor(
            [dummy.table_idx], dtype=torch.int64, device=self.device
        )
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=torch.tensor([0, t], dtype=torch.int32, device=self.device),
            cache_indices=buf.table_idx[:1],
            has_initial_state=torch.tensor([True], dtype=torch.bool, device=self.device),
        )
        self.attn_backend.prepare_for_capture(batch)
        graph = torch.cuda.CUDAGraph()
        with get_global_ctx().forward_batch(batch), model.forward_host_ctx(batch, True):
            logits = model.forward()
            if logits.shape[0] != t:
                raise RuntimeError(
                    f"MTP verify graph warmup returned logits {tuple(logits.shape)}, "
                    f"expected ({t}, vocab). Last-token LM head would clone rows."
                )
            buf.logits[:t].copy_(logits)
            with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                buf.logits[:t].copy_(model.forward())
            self._reset_moe_offload_cache()
        if decode_buf is not None:
            inner._mtp_hidden_buf = decode_buf
        self.mtp_verify_graph = graph
        self._verify_fla = batch.fla_metadata
        logger.info_rank0(f"Captured {t}-token MTP verify CUDA graph")

    def replay_mtp_verify(self, batch: Batch) -> torch.Tensor:
        assert self.mtp_verify_graph is not None and self.verify_buffer is not None
        t = self.verify_tokens
        buf = self.verify_buffer
        buf.input_ids.copy_(batch.input_ids)
        buf.out_loc.copy_(batch.out_loc)
        buf.positions.copy_(batch.positions)
        if buf.mrope_positions is not None:
            buf.mrope_positions.copy_(batch.mrope_positions)
        if batch.linear_table_idx is not None:
            buf.table_idx[:1].copy_(batch.linear_table_idx.reshape(-1)[:1])
        batch.input_ids = buf.input_ids
        batch.out_loc = buf.out_loc
        batch.positions = buf.positions
        if buf.mrope_positions is not None:
            batch.mrope_positions = buf.mrope_positions
        batch.linear_table_idx = buf.table_idx[:1]
        batch.fla_metadata = self._verify_fla
        self.attn_backend.prepare_for_replay(batch)
        self.mtp_verify_graph.replay()
        return buf.logits[:t]

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.buffer.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        # Drop the CUDAGraph objects (and the shared mempool they hold) AND the static
        # GraphCaptureBuffer tensors ([max_bs, vocab] logits + input/out_loc/positions/...).
        # Dropping the references is the load-bearing step; without it a runtime rebuild's
        # free-before-alloc cannot reclaim this GPU memory. empty_cache() is left to the
        # caller / next capture (GraphRunner._capture_graphs already runs it).
        self.graph_map = {}
        self.buffer = None
        self.mtp_verify_graph = None
        self.verify_buffer = None
        gc.collect()
