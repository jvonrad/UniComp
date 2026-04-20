#!/bin/bash
# sweep.sh — submit all jobs from the paper
# Usage: bash sweep.sh

MODELS=(
  llama_8b_instruct llama_8b_wanda_50 llama_8b_sparsegpt_50 llama_8b_awq
  qwen_7b_instruct qwen_7b_wanda_50 qwen_7b_sparsegpt_50
  # ...
)
BENCHMARKS=(knowledge reasoning instruction multilingual)

for model in "${MODELS[@]}"; do
  for bench in "${BENCHMARKS[@]}"; do
    sbatch run_benchmark.sh "$model" "$bench"
  done
done