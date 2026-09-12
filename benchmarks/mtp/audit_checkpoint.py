"""Reconcile the MTP module with safetensors headers without reading weight data."""

import argparse
import json
import struct
from pathlib import Path

import torch
from freetoken.distributed import DistributedInfo, set_tp_info
from freetoken.engine.config import EngineConfig
from freetoken.layers import set_rope_device
from freetoken.utils import torch_dtype


class HeaderReader:
    def __init__(self, path):
        self.path = path
        self._weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}
        self.read_keys = set()

    def get(self, key):
        filename = self._weight_map[key]
        if filename not in self.headers:
            with (self.path / filename).open("rb") as stream:
                length = struct.unpack("<Q", stream.read(8))[0]
                self.headers[filename] = json.loads(stream.read(length))
        spec = self.headers[filename][key]
        dtype = {"BF16": torch.bfloat16, "F32": torch.float32,
                 "F8_E4M3": torch.float8_e4m3fn}[spec["dtype"]]
        self.read_keys.add(key)
        return torch.empty(spec["shape"], dtype=dtype, device="meta")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_tp_info(rank=0, size=1)
    set_rope_device(torch.device("cpu"))
    config = EngineConfig(str(args.model), DistributedInfo(0, 1), torch.bfloat16,
                          experimental_mtp=True).model_config
    if config.model_type == "qwen4_exp":
        from freetoken.models.qwen4_exp.mtp import Qwen4ExpMTP as Head
        prefix = "mtp."
    else:
        from freetoken.models.glm5_next.mtp import Glm5NextMTP as Head
        prefix = f"model.language_model.layers.{config.num_layers}."
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        head = Head(config)
    reader = HeaderReader(args.model)
    head.load(reader, "meta")
    unused = sorted(k for k in reader._weight_map if k.startswith(prefix) and k not in reader.read_keys)
    if unused:
        raise ValueError(f"Unconsumed MTP checkpoint tensors: {unused}")
    report = {"model": str(args.model), "model_type": config.model_type,
              "checkpoint_tensors": len(reader.read_keys),
              "module_tensors": len(head.state_dict()),
              "unused_tensors": unused, "schema_load": "passed",
              "weight_values_read": False}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
