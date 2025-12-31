# train_minitron_depth_chat.py
import os
import random

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig

# 1) Base model (Depth)
BASE_MODEL = "nvidia/Llama-3.1-Minitron-4B-Depth-Base"  # :contentReference[oaicite:7]{index=7}
OUT_DIR = "./Llama-3.1-Minitron-4B-Depth-Chat"

SEED = 42
MAX_SEQ_LEN = 4096
NUM_SAMPLES = 64_000  # rasyosef used 64k from OpenHermes-2.5 :contentReference[oaicite:8]{index=8}

random.seed(SEED)
torch.manual_seed(SEED)

# 2) Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
)

# 3) Patch tokenizer like rasyosef: add <|im_start|>/<|im_end|>, set BOS/EOS/PAD, set chat_template
# rasyosef tokenizer_config shows these tokens and this template :contentReference[oaicite:9]{index=9}
specials = {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}
num_added = tokenizer.add_special_tokens(specials)

tokenizer.bos_token = "<|im_start|>"
tokenizer.eos_token = "<|im_end|>"
# pad = eos is common for causal LM training
tokenizer.pad_token = tokenizer.eos_token

tokenizer.chat_template = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

# Resize embeddings if we added new tokens
if num_added > 0:
    model.resize_token_embeddings(len(tokenizer))

# Make sure model config EOS/BOS/PAD align (important for generation stop!)
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# 4) Load OpenHermes-2.5 and map to text via chat_template
# Dataset has fields like conversations + system_prompt :contentReference[oaicite:10]{index=10}
ds = load_dataset("teknium/OpenHermes-2.5", split="train")

def to_role(fr: str) -> str:
    fr = (fr or "").lower()
    if fr in ("human", "user"):
        return "user"
    if fr in ("gpt", "assistant"):
        return "assistant"
    if fr == "system":
        return "system"
    # fallback: treat unknown as user
    return "user"

def format_example(ex):
    # OpenHermes has "system_prompt" often :contentReference[oaicite:11]{index=11}
    sys_prompt = ex.get("system_prompt") or "You are a helpful assistant."
    conv = ex.get("conversations") or []

    messages = []
    # If the conversation already contains a system message, keep it.
    # Otherwise prepend system_prompt.
    has_system = any((m.get("from") or "").lower() == "system" for m in conv)
    if not has_system:
        messages.append({"role": "system", "content": sys_prompt})

    for m in conv:
        role = to_role(m.get("from"))
        content = m.get("value") or ""
        messages.append({"role": role, "content": content})

    # IMPORTANT: during training we want the assistant responses included,
    # so add_generation_prompt=False.
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

# Optional: drop rows that say "skip_prompt_formatting" if you want strictly consistent formatting
# (field exists in OpenHermes) :contentReference[oaicite:12]{index=12}
if "skip_prompt_formatting" in ds.column_names:
    ds = ds.filter(lambda x: not x.get("skip_prompt_formatting", False))

ds = ds.shuffle(seed=SEED)
ds = ds.select(range(min(NUM_SAMPLES, len(ds))))
ds = ds.map(format_example, remove_columns=ds.column_names)

# 5) Train (simple full SFT like rasyosef did; they mention single A100 40GB) :contentReference[oaicite:13]{index=13}
sft_args = SFTConfig(
    output_dir=OUT_DIR,
    seed=SEED,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="no",      # no intermediate checkpoints
    bf16=True,
    report_to="none",
    optim="adamw_torch",

    # ✅ moved here in new TRL
    dataset_text_field="text",   # defaults to "text" anyway :contentReference[oaicite:1]{index=1}
    max_length=MAX_SEQ_LEN,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=ds,
    args=sft_args,
)

trainer.train()

trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("Saved to:", OUT_DIR)
