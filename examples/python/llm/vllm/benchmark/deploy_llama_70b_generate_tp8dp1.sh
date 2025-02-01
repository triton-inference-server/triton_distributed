#!/bin/bash
set -e
set -x
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_TORCH_PORT=36183
export VLLM_CONTEXT_WORKERS=4
export VLLM_CONTEXT_TP_SIZE=2
export VLLM_GENERATE_WORKERS=1
export VLLM_GENERATE_TP_SIZE=8
export VLLM_LOGGING_LEVEL=INFO
export VLLM_DATA_PLANE_BACKEND=nccl
export PYTHONUNBUFFERED=1



if [ "$1" != "--head-url" ] || [ -z "$2" ]; then
    echo "Usage: $0 --head-url <head url>"
    exit 1
fi
head_url=$2

export DEFAULT_REQUESTS_HOST="$head_url"
export DEFAULT_REQUESTS_URI="nats://${DEFAULT_REQUESTS_HOST}:${DEFAULT_REQUESTS_PORT}"
export VLLM_TORCH_HOST="$head_url"
export API_SERVER_HOST="$head_url"


# Empty --log-dir will dump logs to stdout
echo "Starting vLLM generate workers..."

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
  VLLM_WORKER_ID=${VLLM_CONTEXT_WORKERS} \
  python3 -m llm.vllm.deploy \
  --generate-worker-count 1 \
  --context-tp-size ${VLLM_CONTEXT_TP_SIZE} \
  --generate-tp-size ${VLLM_GENERATE_TP_SIZE} \
  --request-plane-uri ${DEFAULT_REQUESTS_URI} \
  --model-name neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
  --worker-name llama \
  --kv-cache-dtype fp8 \
  --dtype auto \
  --disable-async-output-proc \
  --disable-log-stats \
  --max-model-len 3500 \
  --max-batch-size 10000 \
  --gpu-memory-utilization 0.9 &
