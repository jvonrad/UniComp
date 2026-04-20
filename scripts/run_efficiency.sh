conda activate light
    vllm bench throughput \
      --model "$CURR_MODEL" \
      --dataset-name random \
      --input-len 1024 --output-len 16
    ;;
    vllm bench latency \
      --model "$CURR_MODEL" \
      --dataset-name random \
      --input-len 1024 --output-len 16
    ;;