"""Single-layer Qwen4Exp MTP with the checkpoint's BF16 draft experts."""

from dataclasses import replace

import torch
from freetoken.layers import BaseOP, GemmaPlusOneRMSNorm, LinearReplicated, OPList
from freetoken.layers.moe import MoELayer
from freetoken.moe.fused import fused_topk

from .hc import GatedResidual
from .model import Qwen4ExpDecoderLayer
from .weight import _try_fuse


class _DraftExperts(MoELayer):
    def forward(self, hidden_states, router_logits=None):
        weights, ids = fused_topk(
            hidden_states, router_logits, self.top_k, self.renormalize
        )
        return self.routed_forward(hidden_states, weights, ids)


class Qwen4ExpMTP(BaseOP):
    def __init__(self, config):
        args = config.qwen4_args
        d, n = config.hidden_size, args.hc_count
        self._hc_count = n
        self.fc_embedding = LinearReplicated(d, d, has_bias=False)
        self.fc_hidden = LinearReplicated(d, d, has_bias=False)
        self.pre_fc_norm_embedding = GemmaPlusOneRMSNorm(d, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaPlusOneRMSNorm(d * n, eps=config.rms_norm_eps)
        draft_config = replace(config, expert_quant="none", moe_backend="fused")
        layer = Qwen4ExpDecoderLayer(draft_config, config.num_layers)
        layer.mlp.experts = _DraftExperts(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=d,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
        )
        self.layers = OPList([layer])
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)

    def forward(self, hidden, embedding, batch):
        token = self.fc_embedding.forward(self.pre_fc_norm_embedding.forward(embedding))
        h = self.pre_fc_norm_hidden.forward(hidden)
        h = self.fc_hidden.forward(h.reshape(-1, self._hc_count, token.shape[-1]))
        h = (h + token[:, None, :]).flatten(1)
        h = self.layers.op_list[0].forward(h, batch)
        # Pre-mixer multi-stream is the next draft step's hidden (vLLM scheme A).
        self.last_multi = h
        return self.hyper_connection_mixer.mix(h)[0]

    def load(self, reader, device):
        state, pending = {}, {}
        for raw in reader._weight_map:
            if not raw.startswith("mtp."):
                continue
            name = raw.removeprefix("mtp.")
            tensor = reader.get(raw).to(device)
            fused = _try_fuse(name, tensor, pending)
            if fused is None:
                state[name] = tensor
            elif fused:
                state[fused[0]] = fused[1]
        if pending:
            raise ValueError(f"Incomplete MTP projections: {sorted(pending)}")
        self.load_state_dict(state)
