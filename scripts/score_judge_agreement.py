#!/usr/bin/env python3
"""Score judge-vs-human agreement for the validation samples.

For each per-judge file:
  - Overall accuracy + Cohen's kappa + bootstrap 95% CI on accuracy
  - Per-subtask breakdown (accuracy, n)
  - Up to 10 disagreement examples for qualitative inspection

Comparison is case-insensitive whitespace-trimmed exact match.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path("/weka/geiger/gwb082/Jonathans_Thesis/LLMCBench/TrustLLM/saved_evaluations/qwen-2.5-7b-smooth")
FILES = {
    "longformer": ROOT / "judge_validation_longformer.json",
    "gpt-4o":     ROOT / "judge_validation_gpt.json",
}
BOOTSTRAP_N = 2000
SEED = 42
N_DISAGREE_SHOW = 10


def norm(s):
    return str(s).strip().lower() if s is not None else ""


def cohen_kappa(y1, y2):
    labels = sorted(set(y1) | set(y2))
    n = len(y1)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(y1, y2)) / n
    pe = 0.0
    for lab in labels:
        p1 = sum(a == lab for a in y1) / n
        p2 = sum(b == lab for b in y2) / n
        pe += p1 * p2
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def bootstrap_acc_ci(y1, y2, n_boot=BOOTSTRAP_N, seed=SEED):
    rng = random.Random(seed)
    n = len(y1)
    if n == 0:
        return (float("nan"), float("nan"))
    accs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        accs.append(sum(y1[i] == y2[i] for i in idx) / n)
    accs.sort()
    return (accs[int(0.025 * n_boot)], accs[int(0.975 * n_boot)])


def score_file(judge_name, path):
    data = json.loads(path.read_text())

    annotated, skipped = [], 0
    for d in data:
        h = norm(d.get("human_label"))
        if h == "":
            skipped += 1
            continue
        annotated.append({**d, "_h_norm": h, "_e_norm": norm(d.get("eval_res"))})

    y_h = [d["_h_norm"] for d in annotated]
    y_e = [d["_e_norm"] for d in annotated]

    print(f"\n{'=' * 68}")
    print(f"JUDGE: {judge_name}   (file: {path.name})")
    print(f"{'=' * 68}")
    print(f"  annotated: {len(annotated)}  |  skipped (blank human_label): {skipped}")

    if not annotated:
        print("  No annotations to score.")
        return

    acc = sum(a == b for a, b in zip(y_h, y_e)) / len(annotated)
    kappa = cohen_kappa(y_h, y_e)
    lo, hi = bootstrap_acc_ci(y_h, y_e)

    print(f"\n  OVERALL")
    print(f"    accuracy   = {acc:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"    Cohen's κ  = {kappa:.3f}")

    # Per-subtask
    by_sub = defaultdict(list)
    for d in annotated:
        by_sub[d["_subtask"]].append((d["_h_norm"], d["_e_norm"]))
    print(f"\n  PER SUBTASK")
    print(f"    {'subtask':<40} {'n':>4}  {'acc':>6}  {'κ':>6}")
    for sub in sorted(by_sub):
        pairs = by_sub[sub]
        h = [p[0] for p in pairs]
        e = [p[1] for p in pairs]
        a = sum(x == y for x, y in pairs) / len(pairs)
        k = cohen_kappa(h, e) if len(set(h) | set(e)) > 1 else float("nan")
        k_str = f"{k:6.3f}" if k == k else "   n/a"
        print(f"    {sub:<40} {len(pairs):>4}  {a:>6.3f}  {k_str}")

    # Disagreements (sample)
    disagree = [d for d in annotated if d["_h_norm"] != d["_e_norm"]]
    print(f"\n  DISAGREEMENTS: {len(disagree)} of {len(annotated)}  (showing up to {N_DISAGREE_SHOW})")
    for i, d in enumerate(disagree[:N_DISAGREE_SHOW]):
        prompt = str(d.get("prompt", ""))[:120].replace("\n", " ")
        res = str(d.get("res", ""))[:120].replace("\n", " ")
        print(f"    [{i+1}] {d['_subtask']}")
        print(f"        human={d.get('human_label')!r}  judge={d.get('eval_res')!r}")
        print(f"        prompt: {prompt}")
        print(f"        res:    {res}")


def main():
    for name, path in FILES.items():
        score_file(name, path)


if __name__ == "__main__":
    main()
