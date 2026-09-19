"""Single-layer Qwen4Exp MTP with the checkpoint's BF16 draft experts."""

from dataclasses import replace

import torch
from freetoken.layers import BaseOP, GemmaPlusOneRMSNorm, LinearReplicated, OPList
from freetoken.layers.moe import MoELayer
from freetoken.moe.fused import fused_topk

from .hc import GatedResidual
from .model import Qwen4ExpDecoderLayer
from .weight import _DenseFuser


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
        # The draft head is fully BF16: the checkpoint's quantized experts are swapped for
        # independent BF16 draft experts and the qwen dense projections are BF16. Build the
        # draft layer unquantized (quant=None) so make_moe_layer does not pick the
        # checkpoint's NVFP4 expert kernel, which is unusable for a resident (fused) layer.
        draft_config = replace(config, expert_quant="none", moe_strategy="fused", quant=None)
        layer = Qwen4ExpDecoderLayer(draft_config, config.num_layers)
        layer.mlp.experts = _DraftExperts(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=d,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
        )
        self.layers = OPList([layer])
        # Build the HC mixer from the same unquantized config so its input_mix_weight_*
        # modules stay BF16, matching the checkpoint's BF16 MTP tensors.
        self.hyper_connection_mixer = GatedResidual(draft_config, use_combine=False)

    def forward(self, hidden, embedding, batch):
        token = self.fc_embedding.forward(self.pre_fc_norm_embedding.forward(embedding))
        h = self.pre_fc_norm_hidden.forward(hidden)
        h = self.fc_hidden.forward(h.reshape(-1, self._hc_count, token.shape[-1]))
        h = (h + token[:, None, :]).flatten(1)
        h = self.layers.op_list[0].forward(h, batch)
        # Pre-mixer multi-stream is the next draft step's hidden (vLLM scheme A).
        self.last_multi = h
        return self.hyper_connection_mixer.mix(h)[0]

    def load(self, reader, device, packed):
        from freetoken.layers.quantization import get_quant_config

        # The draft layer reuses the target's layer class, so it merges the same packed parts the target's dense reader does; the QuantConfig picks the GDN in_proj layout.
        fuser = _DenseFuser(get_quant_config(), packed)
        state: dict[str, torch.Tensor] = {}
        for raw in reader._weight_map:
            if not raw.startswith("mtp."):
                continue
            name = raw.removeprefix("mtp.")
            tensor = reader.get(raw).to(device)
            fused = fuser.fuse(name, tensor)
            if fused is None:
                state[name] = fuser.check_unfused(name, tensor)
            else:
                state.update(fused)
        if fuser.buf:
            raise ValueError(f"Incomplete MTP projections: {sorted(k[0] + '.' + k[1] for k in fuser.buf)}")
        self.load_state_dict(state)
