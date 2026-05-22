#!/usr/bin/env python3
"""Pool judge outputs by JUDGE TYPE (longformer vs gpt) and sample N per judge.

Outputs separate JSONs, one per judge, so you can validate each judge independently.
Stratified evenly across the subtasks that used that judge.
"""
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path("/weka/geiger/gwb082/Jonathans_Thesis/LLMCBench/TrustLLM/saved_evaluations/qwen-2.5-7b-smooth")
N_PER_JUDGE = 100
SEED = 42

# Map filename suffix -> canonical judge label
# (safety uses 'hf_eval_progress.json' but it's still the longformer classifier under the hood)
JUDGE_OF_FILE = {
    "longformer_eval.json": "longformer",
    "hf_eval_progress.json": "longformer",  # safety overrides filename but uses longformer
    "eval_progress.json": "gpt",
    "perspective_eval_progress.json": "perspective",  # not run here
}


def load_pool_by_subtask(judge_label):
    """Return {subtask: [tagged items]} for a given judge label."""
    by_subtask = defaultdict(list)
    for sub in sorted(ROOT.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.glob("*.json")):
            if JUDGE_OF_FILE.get(f.name) != judge_label:
                continue
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "eval_res" in item and item.get("eval_res") is not None:
                        by_subtask[sub.name].append({
                            "_judge": judge_label,
                            "_subtask": sub.name,
                            "_judge_file": f.name,
                            **item,
                        })
    return by_subtask


def stratified_sample(by_subtask, n_total, rng):
    """Sample ~n_total examples, distributed evenly across subtasks (largest-remainder)."""
    subtasks = list(by_subtask.keys())
    if not subtasks:
        return []
    base = n_total // len(subtasks)
    remainder = n_total - base * len(subtasks)
    # Sort by pool size desc so larger pools absorb remainder
    sized = sorted(subtasks, key=lambda s: -len(by_subtask[s]))
    quota = {s: base for s in subtasks}
    for s in sized[:remainder]:
        quota[s] += 1

    # If a subtask's pool is smaller than its quota, redistribute the shortfall
    shortfall = 0
    final = {}
    for s, q in quota.items():
        take = min(q, len(by_subtask[s]))
        final[s] = take
        shortfall += q - take
    # spread shortfall to subtasks that still have room
    if shortfall:
        for s in sized:
            room = len(by_subtask[s]) - final[s]
            if room > 0:
                add = min(room, shortfall)
                final[s] += add
                shortfall -= add
                if shortfall == 0:
                    break

    out = []
    for s, q in final.items():
        out.extend(rng.sample(by_subtask[s], q))
    return out


def main():
    rng = random.Random(SEED)
    judges = ["longformer", "gpt"]
    summary = {}

    for j in judges:
        by_sub = load_pool_by_subtask(j)
        picks = stratified_sample(by_sub, N_PER_JUDGE, rng)
        out_path = ROOT / f"judge_validation_{j}.json"
        out_path.write_text(json.dumps(picks, ensure_ascii=False, indent=2))
        counts = defaultdict(int)
        for p in picks:
            counts[p["_subtask"]] += 1
        summary[j] = {
            "pool_total": sum(len(v) for v in by_sub.values()),
            "subtasks": {s: len(v) for s, v in by_sub.items()},
            "sampled": len(picks),
            "out_path": str(out_path),
            "from": dict(counts),
        }

    print(f"\n{'='*60}")
    for j, info in summary.items():
        print(f"\nJudge: {j.upper()}")
        print(f"  pool: {info['pool_total']} eval'd items across {len(info['subtasks'])} subtasks")
        print(f"  sampled: {info['sampled']} → {info['out_path']}")
        print(f"  per-subtask draw:")
        for s, c in sorted(info["from"].items()):
            pool_size = info["subtasks"][s]
            print(f"      {c:>3}/{pool_size:>4}  {s}")


if __name__ == "__main__":
    main()
