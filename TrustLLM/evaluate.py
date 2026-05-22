import os
import json
import shutil
import argparse
import tempfile

from trustllm_pkg.trustllm.task.pipeline import (
    run_truthfulness,
    run_safety,
    run_fairness,
    run_robustness,
    run_privacy,
    run_ethics,
)
from trustllm_pkg.trustllm import config


# Judge-output filenames the pipeline writes into ./saved_evaluations/ (cwd).
# These are shared across subtasks → must be snapshotted between calls.
JUDGE_FILES = [
    "longformer_eval.json",
    "hf_eval_progress.json",
    "eval_progress.json",
    "perspective_eval_progress.json",
    "eval_res.json",
]
SAVE_DIR = "saved_evaluations"


def parse_args():
    p = argparse.ArgumentParser(description="Run TrustLLM evaluation for a given model")
    p.add_argument("--model_name", type=str, required=True,
                   help="Model name or path (used for result directory)")
    p.add_argument("--subset", type=int, default=None,
                   help="If set, evaluate only the first N items of each dataset (for judge validation)")
    return p.parse_args()


def snapshot_judge_outputs(model_tag, subtask_tag):
    """Move any judge files written during the last call into a per-subtask folder."""
    out_dir = os.path.join(SAVE_DIR, model_tag, subtask_tag)
    os.makedirs(out_dir, exist_ok=True)
    moved = []
    for fn in JUDGE_FILES:
        src = os.path.join(SAVE_DIR, fn)
        if os.path.exists(src) and os.path.getsize(src) > 2:  # skip empty {}/[]
            shutil.move(src, os.path.join(out_dir, fn))
            moved.append(fn)
    # Clean up empty stubs the pipeline leaves behind
    for fn in JUDGE_FILES:
        src = os.path.join(SAVE_DIR, fn)
        if os.path.exists(src):
            os.remove(src)
    if moved:
        print(f"  [snapshot] {subtask_tag}: {moved} -> {out_dir}")


def maybe_subset(path, n, subset_root):
    """If --subset N is set, write a truncated copy of `path` and return the new path."""
    if not n or not path or not os.path.exists(path):
        return path
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        # Not a flat list (rare in TrustLLM inputs) — skip subsetting silently
        return path
    truncated = data[:n]
    new_path = os.path.join(subset_root, os.path.basename(path))
    os.makedirs(subset_root, exist_ok=True)
    with open(new_path, "w", encoding="utf-8") as f:
        json.dump(truncated, f, ensure_ascii=False, indent=2)
    return new_path


def run_subtask(name, model_tag, fn, **kwargs):
    """Call a pipeline runner with one subtask's path set, then snapshot its judge output."""
    print(f"\n>>> {name}")
    try:
        result = fn(**kwargs)
    except Exception as e:
        print(f"  [error] {name}: {e}")
        result = {"error": str(e)}
    snapshot_judge_outputs(model_tag, name)
    return result


def main():
    args = parse_args()
    model_path = args.model_name
    current_model = config.model_info["model_mapping"].get(model_path, model_path)
    model_tag = current_model.replace("/", "__")
    print(f"Evaluating model: {current_model}  (subset={args.subset})")

    base = f"/home/geiger/gwb082/Jonathans_Thesis/LLMCBench/TrustLLM/generation_results/{current_model}"
    subset_root = tempfile.mkdtemp(prefix="trustllm_subset_") if args.subset else None

    def p(category, fname):
        return maybe_subset(os.path.join(base, category, fname), args.subset, subset_root)

    results = {}

    # ---------- Truthfulness (one subtask per call) ----------
    results["truthfulness_internal"] = run_subtask(
        "truthfulness_internal", model_tag, run_truthfulness,
        internal_path=p("truthfulness", "internal.json"))
    results["truthfulness_external"] = run_subtask(
        "truthfulness_external", model_tag, run_truthfulness,
        external_path=p("truthfulness", "external.json"))
    results["truthfulness_hallucination"] = run_subtask(
        "truthfulness_hallucination", model_tag, run_truthfulness,
        hallucination_path=p("truthfulness", "hallucination.json"))
    results["truthfulness_sycophancy"] = run_subtask(
        "truthfulness_sycophancy", model_tag, run_truthfulness,
        sycophancy_path=p("truthfulness", "sycophancy.json"))
    results["truthfulness_advfact"] = run_subtask(
        "truthfulness_advfact", model_tag, run_truthfulness,
        advfact_path=p("truthfulness", "golden_advfactuality.json"))

    # ---------- Safety (split — these all share hf_eval_progress.json) ----------
    results["safety_jailbreak"] = run_subtask(
        "safety_jailbreak", model_tag, run_safety,
        jailbreak_path=p("safety", "jailbreak.json"),
        jailbreak_eval_type="total")
    results["safety_exaggerated"] = run_subtask(
        "safety_exaggerated", model_tag, run_safety,
        exaggerated_safety_path=p("safety", "exaggerated_safety.json"))
    results["safety_misuse"] = run_subtask(
        "safety_misuse", model_tag, run_safety,
        misuse_path=p("safety", "misuse.json"))
    # toxicity disabled in original — leave off

    # ---------- Fairness ----------
    results["fairness_stereotype_recognition"] = run_subtask(
        "fairness_stereotype_recognition", model_tag, run_fairness,
        stereotype_recognition_path=p("fairness", "stereotype_recognition.json"))
    results["fairness_stereotype_agreement"] = run_subtask(
        "fairness_stereotype_agreement", model_tag, run_fairness,
        stereotype_agreement_path=p("fairness", "stereotype_agreement.json"))
    results["fairness_stereotype_query"] = run_subtask(
        "fairness_stereotype_query", model_tag, run_fairness,
        stereotype_query_test_path=p("fairness", "stereotype_query_test.json"))
    results["fairness_disparagement"] = run_subtask(
        "fairness_disparagement", model_tag, run_fairness,
        disparagement_path=p("fairness", "disparagement.json"))
    results["fairness_preference"] = run_subtask(
        "fairness_preference", model_tag, run_fairness,
        preference_path=p("fairness", "preference.json"))

    # ---------- Robustness ----------
    results["robustness_advglue"] = run_subtask(
        "robustness_advglue", model_tag, run_robustness,
        advglue_path=p("robustness", "AdvGLUE.json"))
    results["robustness_advinstruction"] = run_subtask(
        "robustness_advinstruction", model_tag, run_robustness,
        advinstruction_path=p("robustness", "AdvInstruction.json"))
    results["robustness_ood_detection"] = run_subtask(
        "robustness_ood_detection", model_tag, run_robustness,
        ood_detection_path=p("robustness", "ood_detection.json"))
    results["robustness_ood_generalization"] = run_subtask(
        "robustness_ood_generalization", model_tag, run_robustness,
        ood_generalization_path=p("robustness", "ood_generalization.json"))

    # ---------- Privacy ----------
    results["privacy_confAIde"] = run_subtask(
        "privacy_confAIde", model_tag, run_privacy,
        privacy_confAIde_path=p("privacy", "privacy_awareness_confAIde.json"))
    results["privacy_awareness_query"] = run_subtask(
        "privacy_awareness_query", model_tag, run_privacy,
        privacy_awareness_query_path=p("privacy", "privacy_awareness_query.json"))
    results["privacy_leakage"] = run_subtask(
        "privacy_leakage", model_tag, run_privacy,
        privacy_leakage_path=p("privacy", "privacy_leakage.json"))

    # ---------- Ethics ----------
    results["ethics_explicit"] = run_subtask(
        "ethics_explicit", model_tag, run_ethics,
        explicit_ethics_path=p("ethics", "explicit_moralchoice.json"))
    results["ethics_implicit_social_norm"] = run_subtask(
        "ethics_implicit_social_norm", model_tag, run_ethics,
        implicit_ethics_path_social_norm=p("ethics", "implicit_SocialChemistry101.json"))
    results["ethics_implicit_ETHICS"] = run_subtask(
        "ethics_implicit_ETHICS", model_tag, run_ethics,
        implicit_ethics_path_ETHICS=p("ethics", "implicit_ETHICS.json"))
    results["ethics_awareness"] = run_subtask(
        "ethics_awareness", model_tag, run_ethics,
        awareness_path=p("ethics", "awareness.json"))

    # ---------- Persist aggregated scores ----------
    scores_dir = os.path.join(SAVE_DIR, model_tag)
    os.makedirs(scores_dir, exist_ok=True)
    suffix = f"_subset{args.subset}" if args.subset else ""
    scores_path = os.path.join(scores_dir, f"scores{suffix}.json")
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n========== FINAL RESULTS for {current_model} ==========")
    print(f"Aggregated scores written to: {scores_path}")
    print(f"Per-subtask judge outputs under: {scores_dir}/<subtask>/")
    for name, result in results.items():
        print(f"  {name}: {result}")

    if subset_root and os.path.exists(subset_root):
        shutil.rmtree(subset_root, ignore_errors=True)


if __name__ == "__main__":
    main()
