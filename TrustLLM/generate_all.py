from trustllm_pkg.trustllm.generation.generation import LLMGeneration

    


MODEL_PATH = "/home/geiger/gwb082/Jonathans_Thesis/compressed-models/quantized/Qwen2.5-7B-Instruct-W8A8-Dynamic-Per-Token"
DATA_ROOT  = "/home/geiger/gwb082/Jonathans_Thesis/LLMCBench/TrustLLM/dataset/dataset"

COMMON = dict(
    model_path=MODEL_PATH,
    data_path=DATA_ROOT,
    online_model=True,
    use_deepinfra=False,
    use_replicate=False,
    repetition_penalty=1.0,
    num_gpus=1,
    max_new_tokens=128,
    debug=False,
    device="cuda",   # Slurm remaps GPUs via CUDA_VISIBLE_DEVICES
)

for test_type in ["truthfulness", "safety", "fairness", "robustness", "privacy", "ethics"]:
    llm_gen = LLMGeneration(test_type=test_type, **COMMON)
    llm_gen.generation_results()

