#!/bin/bash
#SBATCH -J distill_llama2_3b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1           # ein Task (Prozess) pro GPU
#SBATCH --cpus-per-task=8             # pro Prozess
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1                  # vier GPUs
#SBATCH --mem=80G                   # Gesamt-RAM (optional anpassen)
#SBATCH --time=0-4:00:00             # z.B. 2 Tage
#SBATCH --output=logs/distill.%j.out
#SBATCH --error=logs/distill.%j.err
#SBATCH --mail-user=jonathan.sakouhi@gmail.com
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
LLAMA_3_8B_INSTRUCT_SPARSEGPT_50="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Llama-3.1-8B-Instruct-sparsegpt-0.5"
LLAMA_3_8B_INSTRUCT_SPARSEGPT_2_OUT_OF_4="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Llama-3.1-8B-Instruct-sparsegpt-2-out-of-4"
LLAMA_3_8B_INSTRUCT_WANDA_50="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Llama-3.1-8B-Instruct-wanda-0.5"
LLAMA_3_8B_INSTRUCT_WANDA_2_OUT_OF_4="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Llama-3.1-8B-Instruct-wanda-2-out-of-4"
LLAMA_3_3B_INSTRUCT_SPARSEGPT_50="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Llama-3.2-3B-Instruct-sparsegpt-0.5"
LLAMA_3_3B_INSTRUCT_WANDA_50="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/Llama-3.2-3B-Instruct-wanda-0.5"

QWEN_2_5_7B_INSTRUCT_SPARSEGPT_50="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/qwen-2.5-7b-it-sparsegpt-0.5"
QWEN_2_5_7B_INSTRUCT_SPARSEGPT_2_OUT_OF_4="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/qwen-2.5-7b-it-sparsegpt-2-out-of-4"
QWEN_2_5_7B_INSTRUCT_WANDA_50="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/qwen-2.5-7b-it-wanda-0.5"
QWEN_2_5_7B_INSTRUCT_WANDA_2_OUT_OF_4="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/qwen-2.5-7b-it-wanda-2-out-of-4"
QWEN_2_5_3B_INSTRUCT_SPARSEGPT_50="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/qwen-2.5-3b-it-sparsegpt-0.5"
QWEN_2_5_3B_INSTRUCT_WANDA_50="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/qwen-2.5-3b-it-wanda-0.5"

# =========================
# Quantized models
# =========================
LLAMA_3_8B_AWQ="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
LLAMA_3_8B_GPTQ="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4" 
QWEN_2_5_7B_INSTRUCT_AWQ_INT4="Qwen/Qwen2.5-7B-Instruct-AWQ"
QWEN_2_5_7B_INSTRUCT_GPTQ_INT4="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"





#CURR_MODEL=$QWEN_2_5_7B_INSTRUCT_GPTQ_INT4
echo "######################### Current Model ########################"
echo "Current model: $CURR_MODEL"
echo "################################################################"
#/home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/qwen-3-8b-SparseGPT-0.5

###########################################
# EFFICIENCY
###########################################


conda activate vllm
vllm bench throughput   --model "Qwen/Qwen2.5-7B-Instruct-AWQ"    --dataset-name random   --input-len 32 --output-len 128
vllm bench latency   --model "$CURR_MODEL" 


###########################################
# RELIABILITY
###########################################

# conda activate trustllm
# python -u $CODE_DIR/TrustLLM/evaluate.py --model_name "$CURR_MODEL"

### lighteval setup ###
# conda activate light
# export VLLM_USE_V1=0

# lighteval vllm \
#     "model_name=$CURR_MODEL" \
#     "ifbench_test" 

# lighteval vllm \
#     "model_name=$CURR_MODEL" \
#     "gsm8k|4" 

# lighteval vllm \
#     "model_name=$CURR_MODEL" \
#     "math_500|4" 

# lighteval vllm \
#     "model_name=$CURR_MODEL" \
#     "gpqa:diamond|5" 



####################### EVALUATION ########################
# srun --tasks=1  --cpus-per-task=8 --nodes=1        --partition=h100-ferranti  --time=0-03:35     --gres=gpu:1    --mem=80G  --pty bash
# srun python -u $CODE_DIR/evaluate_math.py	

# srun python $CODE_DIR/finetune_wiki2.py

# # # Run WIKI evaluator
# srun python -u $CODE_DIR/evaluate_wiki2.py \
#      --path  "$CHECKPOINT" \
#      --batch_size 1 \
#      --max_len 4096  

# python investigate_layer_importance.py \
#   --dtype bfloat16 \
#   --device cuda


# python -u $CODE_DIR/Track_4/evaluate_flops.py --path $CHECKPOINT

# srun python -u $CODE_DIR/Track_6/evaluate_tQA.py \
#      --path  $CHECKPOINT 

# python -u $CODE_DIR/Track_6/evaluate_advglue.py --path  $CHECKPOINT --ntrain 0

# ############### GTPQ QUANTIZATION ########################

# python $CODE_DIR/quantization/gptq.py \
#   --model_path $QWEN_3_8B_BASE \
#   --output_dir "/home/geiger/gwb082/Jonathans_Thesis/compressed-models/quantized/qwen3_8b_gptq4bit" \
#   --bits 4 \
#   --wandb_project      llama-distillation \
#   --wandb_entity       jonathan-von-rad \
#   --wandb_run_name     "qwen_gptq_memory_test" 

# python $CODE_DIR/quantization/quantize_awq.py

# ######################## PRUNING #########################
export WANDB_MODE=disabled

# srun python -u  $CODE_DIR/pruning/wanda/main.py \
#   --model "$LLAMA_3_8B" \
#   --prune_method sparsegpt \
#   --sparsity_ratio 0.5 \
#   --sparsity_type unstructured \
#   --save /home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/llama-3-8b-sparsegpt-0.5_mixed/logs \
#   --save_model /home/geiger/gwb082/Jonathans_Thesis/compressed-models/pruned/llama-3-8b-sparsegpt-0.5_mixed \
#   --wandb_run_name "llama-3-8b-it-sparsegpt-0.5_test_gpu??" \
#   --calib_dataset mixed_reasoning


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



  




# ##################### AWQ QUANTIZATION ########################

# QUANTIZE_MODEL="$QWEN_3_8B_BASE"  # Change this to the model you want to quantize
# WBIT=4
# QGROUP_SIZE=128
# SAVE_PATH="quant_cache/qwen-3-8b-base-w$WBIT-g$QGROUP_SIZE-awq"


# cd $CODE_DIR/quantization/llm-awq/awq
# python -m awq.entry --model_path "$QUANTIZE_MODEL" \
#     --w_bit "$WBIT" --q_group_size "$QGROUP_SIZE" \
#     --run_awq --dump_awq "awq_cache/qwen-3-8b-base-w$WBIT-g$QGROUP_SIZE.pt"

# echo -e "\n AWQ quantization completed for $QUANTIZE_MODEL model with $WBIT bits and $QGROUP_SIZE group size.\n"

# mkdir quant_cache
# python -m awq.entry --model_path "$QUANTIZE_MODEL" \
#     --w_bit "$WBIT" --q_group_size "$QGROUP_SIZE" \
#     --load_awq "awq_cache/qwen-3-8b-base-w$WBIT-g$QGROUP_SIZE.pt" \
#     --q_backend real --dump_quant "$SAVE_PATH"

# echo -e "\n AWQ quantization cache created and model_state_dict saved at $SAVE_PATH. \n"

# python -m awq.entry --model_path "$QUANTIZE_MODEL" \
#     --tasks gsm8k \
#     --w_bit "$WBIT" --q_group_size "$QGROUP_SIZE" \
#     --load_quant "$SAVE_PATH"  

# echo -e "\n AWQ quantization tasks completed \n"

# rom 2
  






#################### Teacher Correction ######################

# python $CODE_DIR/distillation/teacher_correction.py \
#   --checkpoint $LLAMA_3_8B \
#   --out_dir    ./llama_3_1_8b_pile_finetuned_1B \
#   --batch_size 2 \
#   --acc_steps  8


############### Distillation #######################


# ─── Edit only these three ────────────────────────────────────────────

LR=5e-5
ALPHA=1
SEQ_LEN=1024
LAYER_RATIO=0.5
WIKI_PCT=25
# # # # # # ─────────────────────────────────────────────────────────────────────



# # # # # # # Der Rest passt sich automatisch an
# RUN_NAME="llama-3-8b-distill-on-wiki${WIKI_PCT}pct_exponential_alpha${ALPHA}"
# OUTPUT_DIR="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/distilled/${RUN_NAME}"

# echo "=== Starte Distillation von LLAMA-3-8B ==="
# echo "  Teacher:   LLAMA-3-8B"
# echo "  Output:    ${OUTPUT_DIR}"
# echo "  Run name:  ${RUN_NAME}"
# echo "=================================================="
# echo

# srun torchrun \
#   --nnodes=1 \
#   --nproc_per_node=1 \
#   --master_port=29501 \
#   /home/geiger/gwb082/Jonathans_Thesis/LLMCBench/distillation/distill_llama_student.py \
#     --teacher_path       "$LLAMA_3_8B" \
#     --output_dir         "${OUTPUT_DIR}" \
#     --layer_ratio       "${LAYER_RATIO}" \
#     --dataset_name       wikipedia \
#     --dataset_config     20220301.en \
#     --wiki_pct           "${WIKI_PCT}" \
#     --warmup_steps       40 \
#     --max_seq_length     "${SEQ_LEN}" \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8\
#     --learning_rate      "${LR}" \
#     --num_train_epochs   3 \
#     --temperature        2.0 \
#     --alpha_distill      "${ALPHA}" \
#     --wandb_project      llama-distillation \
#     --wandb_entity       jonathan-von-rad \
#     --wandb_run_name     "${RUN_NAME}" 





########


# torchrun --nproc_per_node=2 finetune.py \
#   --base-path        $BASE_PATH \
#   --model-path       /home/geiger/gwb082/LLMs/llama-3/llama-3.2-1b-hf \
#   --teacher-model-path /home/geiger/gwb082/LLMs/llama-3/llama-3.1-8b-hf \
#   --ckpt-name        llama3-1B \
#   --teacher-ckpt-name llama3-8B \
#   --model-type       llama \
#   --teacher-model-fp16 \
#   --model-parallel --model-parallel-size 4 \
#   --gradient-checkpointing \
#   --data-dir         $BASE_PATH/processed_data/dolly/full/llama3 \
#   --batch-size       1 \
#   --eval-batch-size  8 \
#   --gradient-accumulation-steps 16 \
#   --lr               1e-5 \
#   --epochs           3 \
#   --max-length       512 \
#   --max-prompt-length 256 \
#   --save             $BASE_PATH/results/llama3/train/minillm \
#   --log-interval     10 \
#   --eval-interval    200 \
#   --deepspeed        --deepspeed_config $BASE_PATH/configs/deepspeed/ds_config_zero2_fp16.json
