import os
import argparse
from trustllm_pkg.trustllm.task.pipeline import (
    run_truthfulness,
    run_safety,
    run_fairness,
    run_robustness,
    run_privacy,
    run_ethics,
)
from trustllm_pkg.trustllm import config

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TrustLLM evaluation for a given model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name or path (used for result directory)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model_name

    #### CHECK IF model_name IS IN MAPPING ####
    model_mapping = config.model_info["model_mapping"]
    current_model = model_mapping.get(model_path, model_path)

    print(f"Evaluating model: {current_model}")

    result_base_dir = f"/home/geiger/gwb082/Jonathans_Thesis/LLMCBench/TrustLLM/generation_results/{current_model}"

    # # ---------- Truthfulness ----------
    # truthfulness_dir = os.path.join(result_base_dir, "truthfulness")

    # truthfulness_results = run_truthfulness(
    #     internal_path=os.path.join(truthfulness_dir, "internal.json"),
    #     external_path=os.path.join(truthfulness_dir, "external.json"),
    #     hallucination_path=os.path.join(truthfulness_dir, "hallucination.json"),
    #     sycophancy_path=os.path.join(truthfulness_dir, "sycophancy.json"),
    #     advfact_path=os.path.join(truthfulness_dir, "golden_advfactuality.json"),
    # )

    # print("Truthfulness Results:", truthfulness_results)



    # # ---------- Safety ----------
    # safety_dir = os.path.join(result_base_dir, "safety")

    # safety_results = run_safety(
    #     jailbreak_path=os.path.join(safety_dir, "jailbreak.json"),
    #     exaggerated_safety_path=os.path.join(safety_dir, "exaggerated_safety.json"),
    #     misuse_path=os.path.join(safety_dir, "misuse.json"),
    #     toxicity_eval=False,
    #     toxicity_path=os.path.join(safety_dir, "toxicity.json"),
    #     jailbreak_eval_type="total",
    # )

    # # ---------- Fairness ----------
    # fairness_dir = os.path.join(result_base_dir, "fairness")

    # fairness_results = run_fairness(
    #     stereotype_recognition_path=os.path.join(fairness_dir, "stereotype_recognition.json"),
    #     stereotype_agreement_path=os.path.join(fairness_dir, "stereotype_agreement.json"),
    #     stereotype_query_test_path=os.path.join(fairness_dir, "stereotype_query_test.json"),
    #     disparagement_path=os.path.join(fairness_dir, "disparagement.json"),
    #     preference_path=os.path.join(fairness_dir, "preference.json"),
    # )

    # # ---------- Robustness ----------
    # robustness_dir = os.path.join(result_base_dir, "robustness")

    # robustness_results = run_robustness(
    #     advglue_path=os.path.join(robustness_dir, "AdvGLUE.json"),
    #     advinstruction_path=os.path.join(robustness_dir, "AdvInstruction.json"),
    #     ood_detection_path=os.path.join(robustness_dir, "ood_detection.json"),
    #     ood_generalization_path=os.path.join(robustness_dir, "ood_generalization.json"),
    # )

    # ---------- Privacy ----------
    privacy_dir = os.path.join(result_base_dir, "privacy")

    privacy_results = run_privacy(
        privacy_confAIde_path=os.path.join(privacy_dir, "privacy_awareness_confAIde.json"),
        privacy_awareness_query_path=os.path.join(privacy_dir, "privacy_awareness_query.json"),
        privacy_leakage_path=os.path.join(privacy_dir, "privacy_leakage.json"),
    )
    
    print(f"Privacy Results for {current_model}:", privacy_results)

    # # ---------- Ethics ----------
    # ethics_dir = os.path.join(result_base_dir, "ethics")

    # ethics_results = run_ethics(
    #     explicit_ethics_path=os.path.join(ethics_dir, "explicit_moralchoice.json"),
    #     implicit_ethics_path_social_norm=os.path.join(ethics_dir, "implicit_SocialChemistry101.json"),
    #     implicit_ethics_path_ETHICS=os.path.join(ethics_dir, "implicit_ETHICS.json"),
    #     awareness_path=os.path.join(ethics_dir, "awareness.json"),
    # )
    


    # all_results = {
    #     "truthfulness": truthfulness_results,
    #     "safety": safety_results,
    #     "fairness": fairness_results,
    #     "robustness": robustness_results,
    #     "privacy": privacy_results,
    #     "ethics": ethics_results,
    # }

    # print(f"\n========== FINAL RESULTS for {current_model} ==========")
    # for name, result in all_results.items():
    #     print(f"\n--- {name.upper()} ---")
    #     print(result)


if __name__ == "__main__":
    main()