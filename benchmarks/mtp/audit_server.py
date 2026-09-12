"""Run the test server with a bounded sequential-vs-batched verifier audit.

Set MTP_AUDIT_OUTPUT to a repository-local JSONL path and optionally
MTP_AUDIT_ROUNDS (default 64). Pass the usual ft serve arguments. Audited
requests perform extra target forwards and must be excluded from timing.
"""

import json
import os
from pathlib import Path

import torch
from freetoken.speculative.mtp import MTPDecoder, StateSnapshot


_target = MTPDecoder._target


def _error(actual, expected):
    actual, expected = actual.float(), expected.float()
    delta = actual - expected
    return {"max_abs": delta.abs().max().item(),
            "relative_rms": (delta.square().mean().sqrt()
                             / expected.square().mean().sqrt().clamp_min(1e-12)).item()}


def _audit_target(self, batch, *, all_logits=False):
    path = os.environ.get("MTP_AUDIT_OUTPUT")
    limit = int(os.environ.get("MTP_AUDIT_ROUNDS", "64"))
    if not path or not all_logits or self.rounds >= limit:
        return _target(self, batch, all_logits=all_logits)

    snapshot = StateSnapshot(self.engine)
    req = batch.reqs[0]
    sequential = []
    for offset in range(len(batch.input_ids)):
        one = self._batch(req, req.cached_len + offset, batch.input_ids[offset:offset + 1])
        one.phase = "decode"
        one.active_table_idx = torch.tensor([req.table_idx], dtype=torch.int32,
                                            device=batch.input_ids.device)
        slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
        one.linear_table_idx = torch.tensor([slot], dtype=torch.int32,
                                            device=batch.input_ids.device)
        self.engine.attn_backend.prepare_metadata(one)
        logits, _ = _target(self, one)
        sequential.append(logits.clone())
    pool = self.engine.linear_state_pool
    states = {"conv": pool.conv_states, "recurrent": pool.recurrent_states,
              **pool.slot_states}
    expected_states = {key: value.clone() for key, value in states.items()}
    snapshot.restore()
    logits, hidden = _target(self, batch, all_logits=True)
    sequential = torch.cat(sequential)
    sequential_best = sequential.float().topk(2, dim=-1)
    batched_best = logits.float().topk(2, dim=-1)
    record = {"uid": req.uid, "position": req.cached_len,
              "inputs": batch.input_ids.tolist(),
              "sequential_argmax": sequential.argmax(-1).tolist(),
              "batched_argmax": logits.argmax(-1).tolist(),
              "sequential_margin": (sequential_best.values[:, 0]
                                    - sequential_best.values[:, 1]).tolist(),
              "batched_margin": (batched_best.values[:, 0]
                                 - batched_best.values[:, 1]).tolist(),
              "logits": _error(logits, sequential),
              "state": {key: _error(states[key], value)
                        for key, value in expected_states.items() if value.numel()}}
    with Path(path).open("a") as output:
        output.write(json.dumps(record) + "\n")
    return logits, hidden


MTPDecoder._target = _audit_target


if __name__ == "__main__":
    from freetoken.cli import main

    main()
