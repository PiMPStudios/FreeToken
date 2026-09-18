"""GLM-5.3's plain residual MTP block, with independent FP8 draft experts."""

from dataclasses import replace

import torch
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm
from freetoken.layers.moe import MoELayer

from .attention import Glm5NextAttention
from .moe import Glm5NextSparseBlock
from .weight import _iter_dsa_layer


class _DraftExperts(MoELayer):
    def __init__(self, config):
        super().__init__(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            weight_format="fp8_block",
        )
        self._limit = config.swiglu_limit

    def routed_forward(self, hidden_states, topk_weights, topk_ids):
        from freetoken.kernel.triton.fp8_blockscale_moe import (
            fused_experts_decode_fp8_blockscale,
        )

        limit = self._limit
        return fused_experts_decode_fp8_blockscale(
            hidden_states, self.gate_up_proj, self.gate_up_scale_inv,
            self.down_proj, self.down_scale_inv, topk_weights, topk_ids,
            activation="swiglu_clamp" if limit is not None else "silu",
            act_alpha=1.0,
            act_limit=float("inf") if limit is None else limit,
        )


class _SharedHead(BaseOP):
    def __init__(self, config):
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Glm5NextMTP(BaseOP):
    def __init__(self, config):
        config = replace(config, attn_quant="none", dense_quant="none", moe_strategy="fused")
        self._config = config
        d, eps = config.hidden_size, config.rms_norm_eps
        self.enorm = RMSNorm(d, eps=eps)
        self.hnorm = RMSNorm(d, eps=eps)
        self.eh_proj = LinearReplicated(2 * d, d, has_bias=False)
        self.input_layernorm = RMSNorm(d, eps=eps)
        self.post_attention_layernorm = RMSNorm(d, eps=eps)
        self.self_attn = Glm5NextAttention(config, config.num_layers)
        self.mlp = Glm5NextSparseBlock(replace(config, moe_strategy="offload"), config.num_layers)
        self.mlp.experts = _DraftExperts(config)
        self.shared_head = _SharedHead(config)

    def forward(self, hidden, embedding, batch):
        embedding = embedding.masked_fill((batch.positions == 0).unsqueeze(-1), 0)
        h = self.eh_proj.forward(torch.cat([
            self.enorm.forward(embedding), self.hnorm.forward(hidden),
        ], dim=-1))
        h = h + self.self_attn.forward(self.input_layernorm.forward(h))
        h = h + self.mlp.forward(self.post_attention_layernorm.forward(h))
        return self.shared_head.norm.forward(h)

    def load(self, reader, device, packed=None):
        """The GLM draft head keeps the checkpoint's layer layout as-is, so the packed mapping is unused here."""
        config = self._config
        layer = config.num_layers
        prefix = f"model.language_model.layers.{layer}."
        target = f"model.layers.{layer}."
        expected = self.state_dict()
        state = {
            key.removeprefix(target): tensor.to(device)
            for key, tensor in _iter_dsa_layer(reader, layer, False)
        }
        for key, tensor in expected.items():
            if key in state or key.startswith("mlp.experts."):
                continue
            raw = prefix + key.replace("mlp.e_score_correction_bias", "mlp.gate.e_score_correction_bias")
            state[key] = reader.get(raw).to(device=device, dtype=tensor.dtype)
        for key in ("gate_up_proj", "down_proj", "gate_up_scale_inv", "down_scale_inv"):
            full_key = "mlp.experts." + key
            shape = expected[full_key]
            state[full_key] = torch.empty(shape.shape, dtype=shape.dtype, device=device)
        for expert in range(config.num_experts):
            for proj, dest, start in (
                ("gate_proj", "gate_up", 0),
                ("up_proj", "gate_up", config.moe_intermediate_size),
                ("down_proj", "down", 0),
            ):
                src = f"{prefix}mlp.experts.{expert}.{proj}"
                w = reader.get(src + ".weight")
                s = reader.get(src + ".weight_scale")
                if w.dtype != torch.float8_e4m3fn:
                    raise ValueError("GLM MTP currently requires block-FP8 draft experts")
                state[f"mlp.experts.{dest}_proj"][expert, start:start + w.shape[0]].copy_(w)
                state[f"mlp.experts.{dest}_scale_inv"][expert, start // 128:start // 128 + s.shape[0]].copy_(s)
        self.load_state_dict(state)
        self.self_attn.prepare_for_runtime()
