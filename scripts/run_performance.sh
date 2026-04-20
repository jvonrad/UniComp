#!/bin/bash
#SBATCH -J unicomp_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-ferranti        # <-- change to your partition
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=0-10:00:00
#SBATCH --output=logs/eval.%j.out
#SBATCH --error=logs/eval.%j.err
#SBATCH --mail-user=${SLURM_MAIL}
#SBATCH --mail-type=END

set -euo pipefail
echo "Starting Job $SLURM_JOB_ID at $(date)"

# ── Load user config ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# ── Conda setup ───────────────────────────────────────────────────────────────
export PATH="$CONDA_ROOT/bin:$PATH"
source "$CONDA_ROOT/etc/profile.d/conda.sh"

# ── Model registry ────────────────────────────────────────────────────────────
# Public HuggingFace models (downloaded automatically)
LLAMA_3_8B_INSTRUCT="meta-llama/Llama-3.1-8B-Instruct"
QWEN_2_5_7B_INSTRUCT="Qwen/Qwen2.5-7B-Instruct"
LLAMA_3_3B_INSTRUCT="meta-llama/Llama-3.2-3B-Instruct"
QWEN_2_5_3B_INSTRUCT="Qwen/Qwen2.5-3B-Instruct"
LLAMA_3_8B_AWQ="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
LLAMA_3_8B_GPTQ="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
QWEN_2_5_7B_INSTRUCT_AWQ_INT4="Qwen/Qwen2.5-7B-Instruct-AWQ"
QWEN_2_5_7B_INSTRUCT_GPTQ_INT4="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
QWEN_2_5_3B_INSTRUCT_GPTQ_INT4="Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"
QWEN_2_5_3B_INSTRUCT_AWQ_INT4="Qwen/Qwen2.5-3B-Instruct-AWQ"
LLAMA_3_3B_INSTRUCT_GPTQ_INT4="clowman/Llama-3.2-3B-Instruct-GPTQ-Int4"
LLAMA_3_3B_INSTRUCT_AWQ_INT4="clowman/Llama-3.2-3B-Instruct-AWQ-Int4"
BOOMERANG_1_9B="JitaiHao/LRC-1.5B-SFT"
LRC_1_7B_INSTRUCT="JitaiHao/LRC-1.7B-SFT"

# Locally compressed models (must be generated via scripts in compress/)
LLAMA_3_8B_WANDA_50="$MODEL_DIR/pruned/Llama-3.1-8B-Instruct-wanda-0.5"
LLAMA_3_8B_WANDA_2OF4="$MODEL_DIR/pruned/Llama-3.1-8B-Instruct-wanda-2-out-of-4"
LLAMA_3_8B_SPARSEGPT_50="$MODEL_DIR/pruned/Llama-3.1-8B-Instruct-sparsegpt-0.5"
LLAMA_3_8B_SPARSEGPT_2OF4="$MODEL_DIR/pruned/Llama-3.1-8B-Instruct-sparsegpt-2-out-of-4"
QWEN_2_5_7B_WANDA_50="$MODEL_DIR/pruned/qwen-2.5-7b-it-wanda-0.5"
QWEN_2_5_7B_SPARSEGPT_50="$MODEL_DIR/pruned/qwen-2.5-7b-it-sparsegpt-0.5"
# ... etc, all relative to $MODEL_DIR

# ── Select model ──────────────────────────────────────────────────────────────
# Pass model as argument: sbatch run_benchmark.sh llama_8b_wanda
# Or set CURR_MODEL directly
MODEL_KEY="${1:-llama_8b_instruct}"

case "$MODEL_KEY" in
  llama_8b_instruct)        CURR_MODEL="$LLAMA_3_8B_INSTRUCT" ;;
  llama_8b_wanda_50)        CURR_MODEL="$LLAMA_3_8B_WANDA_50" ;;
  llama_8b_sparsegpt_50)    CURR_MODEL="$LLAMA_3_8B_SPARSEGPT_50" ;;
  llama_8b_awq)             CURR_MODEL="$LLAMA_3_8B_AWQ" ;;
  qwen_7b_instruct)         CURR_MODEL="$QWEN_2_5_7B_INSTRUCT" ;;
  qwen_7b_wanda_50)         CURR_MODEL="$QWEN_2_5_7B_WANDA_50" ;;
  # add more as needed
  *)
    echo "Unknown model key: $MODEL_KEY"
    echo "Available: llama_8b_instruct, llama_8b_wanda_50, ..."
    exit 1 ;;
esac

echo "Model: $CURR_MODEL"

# ── Select benchmark ──────────────────────────────────────────────────────────
BENCH_KEY="${2:-knowledge}"

case "$BENCH_KEY" in
  knowledge)
    conda activate thesis
    lm_eval --model hf \
      --model_args "pretrained=$CURR_MODEL,device_map=auto,dtype=bfloat16" \
      --tasks mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande \
      --batch_size auto
    ;;

  reasoning)
    conda activate light
    export VLLM_USE_V1=0
    export LIGHTEVAL_CONFIG="model_name=$CURR_MODEL,max_model_length=8192,max_num_batched_tokens=8192"
    lighteval vllm "$LIGHTEVAL_CONFIG" "gsm8k|4"
    lighteval vllm "$LIGHTEVAL_CONFIG" "math_500|4"
    lighteval vllm "$LIGHTEVAL_CONFIG" "gpqa:diamond|5"
    ;;

  instruction)
    conda activate light
    export VLLM_USE_V1=0
    export LIGHTEVAL_CONFIG="model_name=$CURR_MODEL,max_model_length=8192,max_num_batched_tokens=8192"
    lighteval vllm "$LIGHTEVAL_CONFIG" "ifbench_test" --remove-reasoning-tags
    ;;

  multilingual)
    conda activate thesis
    TASKS="global_mmlu_full_en,global_mmlu_full_de,global_mmlu_full_fr,global_mmlu_full_es"
    TASKS+=",global_mmlu_full_ru,global_mmlu_full_zh,global_mmlu_full_ja,global_mmlu_full_ar"
    TASKS+=",global_mmlu_full_sw,global_mmlu_full_bn,global_mmlu_full_te,global_mmlu_full_pt"
    lm_eval --model hf \
      --model_args "pretrained=$CURR_MODEL,device_map=auto,dtype=bfloat16" \
      --tasks "$TASKS" \
      --batch_size auto
    ;;

  *)
    echo "Unknown benchmark: $BENCH_KEY"
    echo "Available: knowledge, reasoning, instruction, multilingual, reliability, efficiency"
    exit 1 ;;
esac

echo "Done at $(date)"