from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch

model_id = "/home/geiger/gwb082/LLMs/Qwen/Qwen3-0.6B-Base"
out_dir  = "/home/geiger/gwb082/Jonathans_Thesis/compressed-models/quantized/qwen3-0-6b-awq4"

# load model in full precision first
model = AutoAWQForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    safetensors=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

# ► ONE config dict ◄ – include every AWQ option you want
quant_cfg = {
    "w_bit": 4,             # weight bits
    "q_group_size": 128,    # group size
    "zero_point": True      # Symmetric / asymmetric
    # "fuse_qkv": True,     # add these if you need kernel fusion
    # "fuse_mlp": True,
}

# run calibration + quantisation
model.quantize(
    tokenizer=tokenizer,
    quant_config=quant_cfg        # ← pass the dict **once**
)

# save the 4-bit model
model.save_quantized(out_dir)
tokenizer.save_pretrained(out_dir)

print("✓ AWQ-quantised model written to", out_dir)
