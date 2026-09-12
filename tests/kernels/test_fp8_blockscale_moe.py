"""The block-fp8 MoE kernels against a torch reference, for each gate-up epilogue they accept."""

import pytest
import torch
import torch.nn.functional as F

from freetoken.kernel.triton.fp8_blockscale_moe import fused_experts_decode_fp8_blockscale

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

E, H, I, TOPK, M = 4, 256, 256, 2, 8
BLOCK = 128


def _quant_block(w):
    n, k = w.shape
    blocks = w.float().view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    scale = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12) / 448.0
    q = (blocks / scale).to(torch.float8_e4m3fn).view(n, k)
    return q, scale.squeeze(1).squeeze(-1).to(torch.bfloat16)


def _dequant(q, s):
    n, k = q.shape
    return (q.float().view(n // BLOCK, BLOCK, k // BLOCK, BLOCK) * s.float()[:, None, :, None]).view(n, k)


def _experts():
    torch.manual_seed(0)
    gu = [_quant_block(torch.randn(2 * I, H, device="cuda") / H**0.5) for _ in range(E)]
    dn = [_quant_block(torch.randn(H, I, device="cuda") / I**0.5) for _ in range(E)]
    stack = lambda xs, i: torch.stack([x[i] for x in xs]).contiguous()
    return stack(gu, 0), stack(gu, 1), stack(dn, 0), stack(dn, 1)


def _routing():
    ids = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(M)]).to(torch.int32)
    w = torch.softmax(torch.randn(M, TOPK, device="cuda"), dim=-1)
    return w, ids


def _reference(x, gate_up, gate_up_scale, down, down_scale, w, ids, activation, alpha, limit):
    from freetoken.layers import gated_act_and_mul

    out = torch.zeros(M, H, device="cuda", dtype=torch.float32)
    for e in range(E):
        gu = _dequant(gate_up[e], gate_up_scale[e])
        dn = _dequant(down[e], down_scale[e])
        for t, k in zip(*torch.nonzero(ids == e, as_tuple=True)):
            h = (x[t].float() @ gu.T).to(x.dtype).view(1, -1)
            a = torch.empty(1, I, device="cuda", dtype=x.dtype)
            gated_act_and_mul(activation, h, a, alpha=alpha, limit=limit)
            out[t] += w[t, k] * (a[0].float() @ dn.T)
    return out.to(x.dtype)


@pytest.mark.parametrize("activation, alpha, limit", [("silu", 1.0, float("inf")), ("swiglu_clamp", 1.0, 0.5), ("gelu_tanh", 1.0, float("inf"))])
def test_fp8_block_moe_epilogues_match_the_reference(activation, alpha, limit):
    from freetoken.kernel.triton.fp8_blockscale_moe import fused_experts_fp8_blockscale

    gate_up, gate_up_scale, down, down_scale = _experts()
    w, ids = _routing()
    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    ref = _reference(x, gate_up, gate_up_scale, down, down_scale, w, ids, activation, alpha, limit).float()
    plain = _reference(x, gate_up, gate_up_scale, down, down_scale, w, ids, "silu", 1.0, float("inf")).float()
    if activation != "silu":
        assert not torch.allclose(ref, plain, rtol=1e-2, atol=1e-3), "the epilogue under test must change the result"

    decode = fused_experts_decode_fp8_blockscale(x, gate_up, gate_up_scale, down, down_scale, w, ids, activation, alpha, limit).float()
    # W8A16 decode: only bf16 accumulation-order differences remain
    assert torch.nn.functional.cosine_similarity(decode.flatten(), ref.flatten(), dim=0) > 0.999
    assert (decode - ref).abs().max() <= 2e-2 * ref.abs().max() + 1e-3

    prefill = fused_experts_fp8_blockscale(x, gate_up, gate_up_scale, down, down_scale, w, ids, E, activation, alpha, limit).float()
    # W8A8 prefill quantizes the activations per 128-group, so the tolerance is the fp8 activation error
    assert torch.nn.functional.cosine_similarity(prefill.flatten(), ref.flatten(), dim=0) > 0.99
    assert (prefill - ref).abs().max() <= 8e-2 * ref.abs().max() + 1e-3


@pytest.mark.parametrize("limit", [None, 10.0])
def test_fp8_draft_experts_match_dequantized_reference(limit):
    """The MTP draft experts keep GLM's clamped SwiGLU semantics through the shared decode kernel."""
    torch.manual_seed(81)
    device = "cuda"
    m, h, inter, experts = 2, 256, 128, 3
    x = (torch.randn(m, h, device=device) * 2).bfloat16()
    gu = torch.randn(experts, 2 * inter, h, device=device).to(torch.float8_e4m3fn)
    dn = torch.randn(experts, h, inter, device=device).to(torch.float8_e4m3fn)
    gs = (torch.rand(experts, 2, 2, device=device) * 0.1 + 0.25).bfloat16()
    ds = (torch.rand(experts, 2, 1, device=device) * 0.1 + 0.1).bfloat16()
    ids = torch.tensor([[0, 2], [2, 1]], dtype=torch.int32, device=device)
    weights = torch.tensor([[0.7, 0.3], [0.6, 0.4]], device=device)
    result = fused_experts_decode_fp8_blockscale(
        x, gu, gs, dn, ds, weights, ids,
        activation="silu" if limit is None else "swiglu_clamp",
        act_alpha=1.0,
        act_limit=float("inf") if limit is None else limit,
    )
    gu_ref = gu.float() * gs.float().repeat_interleave(128, 1).repeat_interleave(128, 2)
    dn_ref = dn.float() * ds.float().repeat_interleave(128, 1).repeat_interleave(128, 2)
    expected = []
    for row in range(m):
        routed = []
        for k in range(2):
            e = int(ids[row, k])
            gate, up = F.linear(x[row].float(), gu_ref[e]).bfloat16().float().chunk(2)
            if limit is not None:
                gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
            activated = (F.silu(gate) * up).bfloat16().float()
            routed.append((F.linear(activated, dn_ref[e]) * weights[row, k]).bfloat16())
        expected.append(torch.stack(routed).float().sum(0).bfloat16())
    torch.testing.assert_close(result, torch.stack(expected), rtol=0.02, atol=0.5)
