#!/bin/bash
#SBATCH -J evaluate_unicomp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1           # ein Task (Prozess) pro GPU
#SBATCH --cpus-per-task=8             # pro Prozess
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1                  # vier GPUs
#SBATCH --mem=80G                   # Gesamt-RAM (optional anpassen)
#SBATCH --time=0-04:00:00             # z.B. 2 Tage
#SBATCH --output=logs/distill.%j.out
#SBATCH --error=logs/distill.%j.err
#SBATCH --mail-user=xx
#SBATCH --mail-type=END


# Fail-Fast / Debug
echo "Starte Job $SLURM_JOB_ID am $(date)"

# -------- Conda & HF setup --------
export PATH="$HOME/miniconda/bin:$PATH"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate thesis

export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
CODE_DIR="/home/geiger/gwb082/Jonathans_Thesis/LLMCBench"
MODEL_DIR="/home/geiger/gwb082/LLMs"

# =========================
# Base / Full models
# =========================
LLAMA_3_8B="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/full/Llama-3.1-8B"
LLAMA_3_8B_INSTRUCT="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/full/Llama-3.1-8B-Instruct"
QWEN_2_5_7B_INSTRUCT="Qwen/Qwen2.5-7B-Instruct"
PROMETHEUS="prometheus-eval/prometheus-7b-v2.0"


# =========================
# Distilled models
# =========================
LRC_1_5B_BASE="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/distilled/LRC-1.5B-Base"
LRC_1_7B_BASE="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/distilled/LRC-1.7B-Base"
LRC_4B_BASE="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/distilled/LRC-4B-Base"
LRC_4B_SFT="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/distilled/LRC-4B-SFT"
LLAMA_3_1_MINITRON_4B_DEPTH_BASE="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/distilled/Llama-3.1-Minitron-4B-Depth-Base"
LLAMA_3_1_MINITRON_4B_DEPTH_IT="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/distilled/Llama-3.1-Minitron-4B-Depth-Chat"
LLAMA_3_1_MINITRON_4B_WIDTH_BASE="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/distilled/Llama-3.1-Minitron-4B-Width-Base"
LLAMA_3_1_MINITRON_4B_WIDTH_IT="rasyosef/Llama-3.1-Minitron-4B-Chat"
BOOMERANG_QWEN3_4_9B="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/distilled/boomerang-qwen3-4.9B"

# =========================
# Pruned models
# =========================
LLAMA_3_8B_INSTRUCT_WANDA_50="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Llama-3.1-8B-Instruct-wanda-0.5"
LLAMA_3_8B_INSTRUCT_WANDA_2_OUT_OF_4="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Llama-3.1-8B-Instruct-wanda-2-out-of-4"
LLAMA_3_8B_VLLM_WANDA_2_OUT_OF_4="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Meta-Llama-3-8B-InstructWANDA-2of4-W8A8-FP8-Dynamic"
LLAMA_3_8B_INSTRUCT_SPARSEGPT_50="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Llama-3.1-8B-Instruct-sparsegpt-0.5"
LLAMA_3_8B_INSTRUCT_SPARSEGPT_2_OUT_OF_4="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Llama-3.1-8B-Instruct-sparsegpt-2-out-of-4"
LLAMA_3_8B_VLLM_SPARSEGPT_2_OUT_OF_4="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Meta-Llama-3-8B-Instruct2of4-W8A8-FP8-Dynamic-Per-Token"
LLAMA_3_3B_INSTRUCT_SPARSEGPT_50="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Llama-3.2-3B-Instruct-sparsegpt-0.5"
LLAMA_3_3B_INSTRUCT_WANDA_50="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Llama-3.2-3B-Instruct-wanda-0.5"

QWEN_2_5_7B_INSTRUCT_WANDA_50="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/qwen-2.5-7b-it-wanda-0.5"
QWEN_2_5_7B_INSTRUCT_WANDA_2_OUT_OF_4="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/qwen-2.5-7b-it-wanda-2-out-of-4"
QWEN_2_5_7B_INSTRUCT_VLLM_WANDA_2_OUT_OF_4="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Qwen2.5-7B-InstructWANDA-2of4-W8A8-FP8-Dynamic"
QWEN_2_5_7B_INSTRUCT_SPARSEGPT_50="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/qwen-2.5-7b-it-sparsegpt-0.5"
QWEN_2_5_7B_INSTRUCT_SPARSEGPT_2_OUT_OF_4="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/qwen-2.5-7b-it-sparsegpt-2-out-of-4"
QWEN_2_5_7B_INSTRUCT_VLLM_SPARSEGPT_2_OUT_OF_4="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Qwen2.5-7B-InstructSPARSEGPT-2of4-W8A8-FP8-Dynamic"
QWEN_2_5_3B_INSTRUCT_SPARSEGPT_50="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/qwen-2.5-3b-it-sparsegpt-0.5"
QWEN_2_5_3B_INSTRUCT_WANDA_50="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/qwen-2.5-3b-it-wanda-0.5"

# =========================
# Quantized models
# =========================
LLAMA_3_8B_AWQ="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
LLAMA_3_8B_GPTQ="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4" 
LLAMA_3_8B_SMOOTHQUANT="/home/geiger/gwb082/Jonathans_Thesis/LLMCBench/llm-compressor/examples/quantization_w8a8_int8/Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token"

LLAMA_3_8B_INT8="/home/geiger/gwb082/Jonathans_Thesis/LLMCBench/llm-compressor/examples/quantization_w8a8_fp8/Llama-3.1-8B-Instruct-FP8-Dynamic"
QWEN_2_5_7B_INSTRUCT_AWQ_INT4="Qwen/Qwen2.5-7B-Instruct-AWQ"
QWEN_2_5_7B_INSTRUCT_GPTQ_INT4="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"





# =========================
# Calibration Data Experiments
# =========================
LLAMA_3_8B_WANDA_50_REASONING="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Llama-3.1-8B-Instruct-wanda-50-arc-gsm8k-math"
LLAMA_3_8B_WANDA_2_OUT_OF_4_REASONING="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Llama-3.1-8B-Instruct-wanda-2-out-of-4-arc-gsm8k-math"
LLAMA_3_8B_SPARSEGPT_50_REASONING="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Llama-3.1-8B-Instruct-sparsegpt-50-arc-gsm8k-math"
LLAMA_3_8B_SPARSEGPT_50_REASONING_WIKI="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Llama-3.1-8B-Instruct-sparsegpt-50-arc-gsm8k-wiki"
LLAMA_3_8B_AWQ_REASONING="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/quantized/Llama-3.1-8B-Instruct-awq--arc-gsm8k-math"

QWEN_2_5_7B_SPARSEGPT_50_REASONING="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Qwen2.5-7B-Instruct-sparsegpt-50-arc-gsm8k-math"
QWEN_2_5_7B_AWQ_REASONING="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/quantized/Qwen2.5-7B-Instruct-awq--arc-gsm8k-math"

# ====================
# LLAMA 3.2 3B models
# ====================
LLAMA_3_3B_INSTRUCT="meta-llama/Llama-3.2-3B-Instruct"
LLAMA_3_3B_INSTRUCT_SPARSEGPT_50="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Llama-3.2-3B-Instruct-sparsegpt-0.5"
LLAMA_3_3B_INSTRUCT_WANDA_50="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Llama-3.2-3B-Instruct-wanda-0.5"
LLAMA_3_3B_INSTRUCT_GPTQ_INT4="clowman/Llama-3.2-3B-Instruct-GPTQ-Int4"
LLAMA_3_3B_INSTRUCT_AWQ_INT4="clowman/Llama-3.2-3B-Instruct-AWQ-Int4" # needs transformers <= 4.54.0 bc of AutoAWQ compatiability
BOOMERANG_1_9B="JitaiHao/LRC-1.5B-SFT"


# ====================
# Qwen 2.5 3B models
# ====================
QWEN_2_5_3B_INSTRUCT="Qwen/Qwen2.5-3B-Instruct"
QWEN_2_5_3B_INSTRUCT_GPTQ_INT4="Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"
QWEN_2_5_3B_INSTRUCT_AWQ_INT4="Qwen/Qwen2.5-3B-Instruct-AWQ" # needs transformers <= 4.54.0 bc of AutoAWQ compatiability
QWEN_2_5_3B_INSTRUCT_SPARSEGPT_50="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/qwen-2.5-3b-it-sparsegpt-0.5"
QWEN_2_5_3B_INSTRUCT_WANDA_50="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/qwen-2.5-3b-it-wanda-0.5"
LRC_1_7B_INSTRUCT="JitaiHao/LRC-1.7B-SFT"

# =========================
# Quantized models
# =========================
LLAMA_3_8B_AWQ="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
LLAMA_3_8B_GPTQ="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4" 
LLAMA_3_8B_SMOOTHQUANT="/home/geiger/gwb082/Jonathans_Thesis/LLMCBench/llm-compressor/examples/quantization_w8a8_int8/Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token"

LLAMA_3_8B_INT8="/home/geiger/gwb082/Jonathans_Thesis/LLMCBench/llm-compressor/examples/quantization_w8a8_fp8/Llama-3.1-8B-Instruct-FP8-Dynamic"
QWEN_2_5_7B_INSTRUCT_AWQ_INT4="Qwen/Qwen2.5-7B-Instruct-AWQ"
QWEN_2_5_7B_INSTRUCT_GPTQ_INT4="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
QWEN_2_5_7B_INSTRUCT_SMOOTHQUANT="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/quantized/Qwen2.5-7B-Instruct-W8A8-Dynamic-Per-Token"

QWEN_2_5_7B_INSTRUCT="Qwen/Qwen2.5-7B-Instruct"

export CURR_MODEL=$QWEN_2_5_7B_INSTRUCT_SMOOTHQUANT

echo "######################### Current Model ########################"
echo "Current model: $CURR_MODEL"
echo "################################################################"


# conda activate vllm
# python llm-compressor/examples/quantization_w8a8_int8/llama3_example.py

###########################################
# PERFORMANCE
###########################################

# -------- Knowledge Benchmarks ---------
conda activate thesis
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_NO_USAGE_STATS=1
export VLLM_USE_V1=0

lm_eval --model vllm \
    --model_args "pretrained=$CURR_MODEL,enforce_eager=True" \
    --tasks mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande \
    --batch_size auto 

# mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande \
# for gptq model add. ,gptqmodel=True

# -------- Reasoning Benchmarks ---------
conda activate light
export VLLM_USE_V1=0
export LIGHTEVAL_CONFIG="model_name=$CURR_MODEL,max_model_length=8192,max_num_batched_tokens=8192,generation_parameters={max_new_tokens:4096}"

# For LRC models use and change vllm to accelerate:
# conda activate light
#export LIGHTEVAL_CONFIG="model_name=/home/geiger/gwb082/Jonathans_Thesis/LLMCBench/llm-compressor/examples/quantization_w8a8_int8/Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token"
# # # #lighteval eval "$LIGHTEVAL_CONFIG" "gsm8k|4" 

lighteval vllm "$LIGHTEVAL_CONFIG" "gsm8k|4" 
lighteval vllm "$LIGHTEVAL_CONFIG" "math_500|4" 
lighteval vllm "$LIGHTEVAL_CONFIG" "gpqa:diamond|5" 

# for qwen2.5 models add
# ,max_model_length=8192,max_num_batched_tokens=8192

# ---------- Instruction Following ----------
conda activate light
export VLLM_USE_V1=0
export LIGHTEVAL_CONFIG="model_name=$CURR_MODEL,max_model_length=8192,max_num_batched_tokens=8192"
lighteval vllm "$LIGHTEVAL_CONFIG" "ifbench_test" --remove-reasoning-tags
#,generation_parameters={max_new_tokens:512}
# For LRC models use:
# conda activate light
# export LIGHTEVAL_CONFIG="model_name=$CURR_MODEL"
lighteval accelerate "$LIGHTEVAL_CONFIG" "ifbench_test" 

## For LRC models add: --model-impl transformers

# -------- Multilingual Capabilities ----------
## for wanda 2:4 model add max_new_tokens:16,temperature:0

GLOBAL_1="global_mmlu_en,global_mmlu_de,global_mmlu_fr,global_mmlu_es,global_mmlu_it"
GLOBAL_2="global_mmlu_ar,global_mmlu_hi,global_mmlu_ja,global_mmlu_zh,global_mmlu_pt"
GLOBAL_3="global_mmlu_sw,global_mmlu_yo,global_mmlu_bn,global_mmlu_id,global_mmlu_ko"
conda activate thesis
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_NO_USAGE_STATS=1
export VLLM_USE_V1=0

# ------ GLOBAL MMLU ---------

lm_eval --model vllm \
	--model_args "pretrained=$CURR_MODEL,enforce_eager=True" \
	--tasks $GLOBAL_3 \
	--batch_size auto 

# ------ BBQ ---------

lm_eval --model vllm \
    --model_args "pretrained=$CURR_MODEL,enforce_eager=True" \
    --tasks bbq \
    --batch_size auto \
	--apply_chat_template





###########################################
# EFFICIENCY
###########################################

# --------- Hardware Acceleration ---------
# conda activate light
#vllm bench throughput   --model "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"    --dataset-name random   --input-len 1024 --output-len 16 
#vllm bench latency   --model "/home/geiger/gwb082/Jonathans_Thesis/compressed-models/distilled/LRC-4B-SFT" --model-impl transformers
# ## for LRC-4B-SFT use --model-impl transformers

# # --------- Inference Consumption ---------
# conda activate thesis
# python -u $CODE_DIR/Track_4/evaluate_flops.py --path $QWEN_2_5_7B_INSTRUCT_AWQ_INT4




###########################################
# RELIABILITY
###########################################

# conda activate trustllm
# python -u $CODE_DIR/TrustLLM/evaluate.py --model_name "$CURR_MODEL"

###########################################
# Calibration Data Experiments
###########################################

# python ./quantization/compress_calib.py --model "allenai/OLMoE-1B-7B-0924-Instruct" --compression_method "awq"


###########################################
# COMPRESSION
###########################################

# -------------- SmoothQuant ---------------
# CALIB_DATASET=/home/geiger/gwb082/Jonathans_Thesis/LLMCBench/quantization/smoothquant

# Collect Activations

# python examples/generate_act_scales.py \
#   --model-name meta-llama/Llama-2-7b-hf \
#   --output-path act_scales/llama-2-7b.pt \
#   --num-samples 512 \
#   --seq-len 512 \
#   --dataset-path /path/to/calibration_dataset



## Compress Model with Sparse 2 of 4, so hardware acceleration possible needs fp8 (for vllm < 0.13.0)
#python $CODE_DIR/llm-compressor/examples/sparse_2of4_quantization_fp8/llama3_8b_2of4.py --fp8 


####################### EVALUATION ########################
# srun --tasks=1  --cpus-per-task=8 --nodes=1        --partition=h100-ferranti  --time=0-03:35     --gres=gpu:1    --mem=80G  --pty bash
# srun python -u $CODE_DIR/evaluate_math.py	

# srun python $CODE_DIR/finetune_wiki2.py

# # # Run WIKI evaluator
# conda activate thesis

# srun python -u ./evaluate_wiki2.py \
#      --batch_size 1 \
#      --max_len 4096  \
# 	 --path  "mistralai/Mixtral-8x7B-Instruct-v0.1" 

# python investigate_layer_importance.py \
#   --dtype bfloat16 \
#   --device cuda


# srun python -u $CODE_DIR/Track_6/evaluate_tQA.py \
#      --path  $CHECKPOINT 

# python -u $CODE_DIR/Track_6/evaluate_advglue.py --path  $CHECKPOINT --ntrain 0

# ############### GTPQ QUANTIZATION ########################

# python ./quantization/gptq.py \
#   --model_path "/home/geiger/gwb082/Jonathans_Thesis/compressed-models/full/Llama-3.1-8B" \
#   --output_dir "/home/geiger/gwb082/Jonathans_Thesis/compressed-models/quantized/qwen3_8b_gptq8bit" \
#   --bits 8 

# python $CODE_DIR/quantization/quantize_awq.py

# ######################## PRUNING #########################
export WANDB_MODE=disabled

# srun python -u  ./pruning/wanda/main.py \
#   --model "meta-llama/Llama-3.2-3B-Instruct" \
#   --prune_method sparsegpt \
#   --sparsity_ratio 0.5 \
#   --sparsity_type 2:4 \
#   --save /home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Llama-3.2-3B-Instruct-sparsegpt-2-of-4/logs \
#   --save_model /home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Llama-3.2-3B-Instruct-sparsegpt-2-of-4 \
#   --wandb_run_name "Llama-3.2-3B-Instruct-sparsegpt-0.5_test_gpu??" \
#   --calib_dataset c4


# ######################## LM_EVAL_HARNESS #########################
# lm_eval --model hf \
#     --model_args "pretrained=$CHECKPOINT,device_map=auto,dtype=bfloat16" \
#     --tasks mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande \
#     --batch_size auto

# lm_eval --model hf \
#     --model_args "pretrained=$CHECKPOINT,device_map=auto" \
#     --tasks gsm8k_cot \
#     --num_fewshot 8 \
#     --batch_size auto

# lm_eval --model hf \
#     --model_args "pretrained=$CHECKPOINT,device_map=auto" \
#     --tasks hendrycks_math \
#     --num_fewshot 4 \
#     --batch_size auto

# lm_eval \
#   --model hf \
#   --model_args "pretrained=$QWEN_3_32B,dtype=bfloat16,device_map=auto" \
#   --tasks gsm8k_cot \
#   --num_fewshot 4 \
#   --batch_size auto \
#   --fewshot_as_multiturn \
#   --apply_chat_template



  
