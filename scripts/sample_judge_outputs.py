#!/usr/bin/env python3
"""Pool judge outputs across subtasks of each TrustLLM benchmark and sample N examples per benchmark.

Output: one combined JSON for hand-validation of LLM-as-judge labels.
Each sample is tagged with _benchmark / _subtask / _judge_file for traceability.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path("/weka/geiger/gwb082/Jonathans_Thesis/LLMCBench/TrustLLM/saved_evaluations/qwen-2.5-7b-smooth")
N_PER_SUBTASK = 10
SEED = 42
OUT = ROOT / "judge_validation_sample.json"


def main():
    rng = random.Random(SEED)
    combined = []
    summary = {}

    subtask_dirs = sorted([d for d in ROOT.iterdir() if d.is_dir()])
    for sub in subtask_dirs:
        pool = []
        for f in sorted(sub.glob("*.json")):
            try:
                data = json.loads(f.read_text())
            except Exception as e:
                print(f"  skip {f}: {e}")
                continue
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "eval_res" in item:
                        tagged = {
                            "_subtask": sub.name,
                            "_judge_file": f.name,
                            **item,
                        }
                        pool.append(tagged)
        if not pool:
            continue  # no judge file in this subtask

        n = min(N_PER_SUBTASK, len(pool))
        picks = rng.sample(pool, n)
        combined.extend(picks)

        src_counts = defaultdict(int)
        for p in picks:
            src_counts[p["_judge_file"]] += 1
        summary[sub.name] = {
            "pool_size": len(pool),
            "sampled": len(picks),
            "from": dict(src_counts),
        }

    OUT.write_text(json.dumps(combined, ensure_ascii=False, indent=2))

    print(f"\nWrote {len(combined)} examples to {OUT}")
    print(f"\nPer-subtask breakdown ({len(summary)} subtasks with judge files):")
    for sub, info in summary.items():
        srcs = ", ".join(f"{c} from {s}" for s, c in info["from"].items())
        print(f"  {sub}: sampled {info['sampled']} of {info['pool_size']}  ({srcs})")


if __name__ == "__main__":
    main()
