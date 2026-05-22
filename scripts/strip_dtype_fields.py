#!/usr/bin/env python3
"""Strip null scale_dtype/zp_dtype from compressed-tensors quantization configs.

Why: compressed-tensors==0.12.2 (the latest version that imports on Python 3.9)
uses a pydantic schema with extra=forbid. Newer llmcompressor releases write
scale_dtype/zp_dtype=null into the config, which the older schema rejects.
Since the values are null, dropping them is information-preserving.
"""
import json
import shutil
import sys
from pathlib import Path

TARGETS = [
    "compressed-models/quantized/Qwen3-8B-awq--arc-gsm8k-math/config.json",
    "compressed-models/quantized/Qwen2.5-7B-Instruct-W8A8-Dynamic-Per-Token/config.json",
    "compressed-models/pruned/Qwen2.5-7B-InstructWANDA-2of4-W8A8-FP8-Dynamic/config.json",
    "compressed-models/pruned/Meta-Llama-3-8B-InstructWANDA-2of4-W8A8-FP8-Dynamic/config.json",
    "compressed-models/pruned/Meta-Llama-3-8B-Instruct2of4-W8A8-FP8-Dynamic-Per-Token/config.json",
    "compressed-models/pruned/Qwen2.5-7B-InstructSPARSEGPT-2of4-W8A8-FP8-Dynamic/config.json",
]
ROOT = Path("/home/geiger/gwb082/Jonathans_Thesis")
FIELDS = ("scale_dtype", "zp_dtype")


def strip(node, path=""):
    changes = []
    if isinstance(node, dict):
        for k in list(node.keys()):
            if k in FIELDS:
                if node[k] is None:
                    del node[k]
                    changes.append(f"{path}.{k}")
                else:
                    raise ValueError(f"refusing to drop non-null {path}.{k}={node[k]!r}")
            else:
                changes += strip(node[k], f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            changes += strip(v, f"{path}[{i}]")
    return changes


def main(apply=False):
    for rel in TARGETS:
        path = ROOT / rel
        if not path.exists():
            print(f"SKIP  {path}: not found")
            continue
        cfg = json.loads(path.read_text())
        changes = strip(cfg)
        print(f"{'APPLY' if apply else 'DRY '} {path}: stripped {len(changes)} fields")
        for c in changes:
            print(f"  - {c}")
        if apply and changes:
            backup = path.with_suffix(".json.bak")
            if not backup.exists():
                shutil.copy2(path, backup)
            path.write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
