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

export NATS_STORE="$(mktemp -d)"
export API_SERVER_PORT=8005


if [ "$1" != "--head-url" ] || [ -z "$2" ]; then
    echo "Usage: $0 --head-url <head url>"
    exit 1
fi
head_url=$2

export DEFAULT_REQUESTS_HOST="$head_url"
export DEFAULT_REQUESTS_URI="nats://${DEFAULT_REQUESTS_HOST}:${DEFAULT_REQUESTS_PORT}"
export VLLM_TORCH_HOST="$head_url"
export API_SERVER_HOST="$head_url"


# Start NATS Server
echo "Flushing NATS store: ${NATS_STORE}..."
rm -r "${NATS_STORE}"

echo "Starting NATS Server..."
nats-server -p ${DEFAULT_REQUESTS_PORT} --addr ${DEFAULT_REQUESTS_HOST} --jetstream --store_dir "${NATS_STORE}" &


# Start API Server
echo "Starting LLM API Server..."
python3 -m llm.api_server \
  --tokenizer neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
  --request-plane-uri ${DEFAULT_REQUESTS_URI} \
  --api-server-host ${API_SERVER_HOST} \
  --model-name "llama" \
  --api-server-port ${API_SERVER_PORT} &


# Empty --log-dir will dump logs to stdout
echo "Starting vLLM baseline workers..."

gpu_configs=(
  "0,1"
  "2,3"
  "4,5"
  "6,7"
)

for i in "${!gpu_configs[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu_configs[$i]}" \
    VLLM_WORKER_ID=$i \
    python3 -m llm.vllm.deploy \
    --context-worker-count 1 \
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
    --gpu-memory-utilization 0.5 &
done
