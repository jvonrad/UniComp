import os
import argparse

from pathlib import Path

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.pruning import WandaPruningModifier
from llmcompressor.modifiers.pruning import SparseGPTModifier
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation


# ======================
# Configuration
# ======================

BASE_SAVE_DIR = "/home/geiger/gwb082/Jonathans_Thesis/compressed-models/"

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 1024
SEED = 42

# Just for naming the output dir
DATASET_TAGS = ["arc", "gsm8k", "math"]


# ======================
# Model / Tokenizer
# ======================

def load_model_and_tokenizer(model_id: str, device: str = "cuda"):
    """Load the base model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        trust_remote_code=True,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    return model, tokenizer


# ======================
# Dataset loaders
# ======================

def load_commonsenseqa(num_samples: int):
    """
    Load CommonsenseQA as a calibration dataset.
    We keep the 'question' field for later use.
    """
    ds = load_dataset(
        "tau/commonsense_qa",
        "default",
        split=f"train[:{num_samples}]",
    )
    # CommonsenseQA already has a "question" field.
    return ds

def load_c4(num_samples: int):
    ds = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
    )
    ds = ds.take(num_samples)
    return ds

def load_arc(num_samples: int, seed: int = SEED):
    ds = load_dataset(
        "allenai/ai2_arc",
        "ARC-Easy",
        split=f"train[:{num_samples}]"
    ).shuffle(seed=seed)
    
    return ds

def load_gsm8k(num_samples: int, seed: int = SEED):
    """Load GSM8K calibration subset with 'question' field."""
    ds = load_dataset(
        "openai/gsm8k",
        "main",
        split=f"train[:{num_samples}]",
    ).shuffle(seed=seed)
    # GSM8K already has "question"
    return ds


def load_math(num_samples: int, seed: int = SEED):
    """
    Load Hendrycks MATH across categories, merge,
    and expose a 'question' field.
    """
    categories = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]

    math_datasets = []
    for cat in categories:
        ds = load_dataset("EleutherAI/hendrycks_math", cat, split="train")
        math_datasets.append(ds)

    full_math = concatenate_datasets(math_datasets)
    full_math = full_math.shuffle(seed=seed).select(range(num_samples))

    # Rename "problem" -> "question" for consistency
    full_math = full_math.rename_column("problem", "question")
    return full_math


def build_calibration_dataset(tokenizer, num_calibration_samples: int):
    """
    Build the combined calibration dataset:
      ARC + GSM8K + MATH 
    All examples will have a 'text' field with a chat-formatted prompt.
    """
    proportion = num_calibration_samples // 3

    arc_ds = load_arc(proportion)
    gsm8k_ds = load_gsm8k(proportion)
    math_ds = load_math(proportion)
    combined = concatenate_datasets([arc_ds, gsm8k_ds, math_ds])

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                [{"role": "user", "content": example["question"]}],
                tokenize=False,
            )
        }

    combined = combined.map(preprocess)
    return combined


# ======================
# Quantization
# ======================

def build_save_dir(base_dir: str, model_id: str, dataset_tags, compression_method: str, mask_structure: str):
    """Construct a descriptive save directory path."""
    model_name = model_id.rstrip("/").split("/")[-1]
    tag_str = "-".join(dataset_tags)
    if not compression_method == "sparsegpt" and not compression_method == "wanda":
        mask_structure = ""
    elif mask_structure == "2:4":
        mask_structure = "50"
    else:
        mask_structure = "2-out-of-4"
    save_dir = Path(base_dir) / f"{model_name}-{compression_method}-{mask_structure}-{tag_str}"
    os.makedirs(save_dir, exist_ok=True)
    return str(save_dir)




# ======================
# Sanity check generation
# ======================

def sample_generation(model, tokenizer, prompt: str = "Hello my name is", max_new_tokens: int = 100):
    """Do a simple generation with the quantized model to sanity check behavior."""
    print("\n\n========== SAMPLE GENERATION ==============")

    # Patch for autoregressive generation
    dispatch_for_generation(model)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print("==========================================\n\n")


# ======================
# Main
# ======================

def main():
    # 0. Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--compression_method", required=False, type=str, default="awq", choices=["awq", "wanda", "sparsegpt"])
    ap.add_argument("--mask_structure", required=False, type=str, default="0:0", help="Mask structure for SparseGPT pruning, e.g., '0:0' for unstructured.")
    args = ap.parse_args()
    MODEL_ID = args.model
    mask_structure = args.mask_structure
    if args.compression_method != "sparsegpt" and args.compression_method != "wanda":
        base_dir = os.path.join(BASE_SAVE_DIR, "quantized")   
    else:
        base_dir = os.path.join(BASE_SAVE_DIR, "pruned") 
    
    
    # 1. Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model, device="cuda")

    # 2. Build calibration dataset
    calibration_dataset = build_calibration_dataset(tokenizer, NUM_CALIBRATION_SAMPLES)

    # 3. Build save dir
    save_dir = build_save_dir(base_dir, MODEL_ID, DATASET_TAGS, args.compression_method, mask_structure)
    print(f"Saving quantized model to: {save_dir}")

    # 4. COMPRESSION
    
    
    recipe_awq = [
        AWQModifier(
            ignore=["lm_head"],
            scheme="W4A16",
            targets=["Linear"],
        ),
    ]
    
    recipe_wanda = [
        WandaPruningModifier(
            sparsity=0.5,# 50% weights zeroed
            mask_structure=mask_structure,
            targets=["Linear"],    # prune Linear layers
            ignore=["lm_head"],    # usually keep output head dense
        )
    ]

    recipe_sparsegpt = [
        SparseGPTModifier(
            sparsity=0.5,              # 50% zeros
            # SparseGPT expects mask_structure as "N:M" string; "0:0" means unstructured.
            mask_structure=mask_structure,
            targets=["Linear"],        # prune Linear layers
            ignore=["lm_head"],        # often ignored
            sequential_update=True,    # helps memory: prunes layer-by-layer
        )
    ]
    SAVE_COMPRESSED = True
    if args.compression_method == "wanda":
        recipe = recipe_wanda
    elif args.compression_method == "sparsegpt":
        recipe = recipe_sparsegpt
    else:
        SAVE_COMPRESSED = False
        recipe = recipe_awq  # or handle other methods

    
    #### Comment out 
    
    oneshot(
        model=model,
        dataset=calibration_dataset,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        output_dir=save_dir,
        save_compressed=SAVE_COMPRESSED,   # <-- for pruning you usually DO want to save
    )


    # 5. Sanity check generation
    sample_generation(model, tokenizer)

    # 6. Save tokenizer
    tokenizer.save_pretrained(save_dir)
    print(f"Tokenizer saved to: {save_dir}")


if __name__ == "__main__":
    main()
