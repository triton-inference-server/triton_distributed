#!/bin/bash

export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_TORCH_HOST=""
export VLLM_TORCH_PORT=36183
export VLLM_BASELINE_WORKERS=0
export VLLM_CONTEXT_WORKERS=1
export VLLM_GENERATE_WORKERS=1
export VLLM_BASELINE_TP_SIZE=1
export VLLM_CONTEXT_TP_SIZE=1
export VLLM_GENERATE_TP_SIZE=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_DATA_PLANE_BACKEND=nccl
export PYTHONUNBUFFERED=1

export NATS_HOST=""
export NATS_PORT=4223
export NATS_STORE="$(mktemp -d)"
export API_SERVER_HOST=""
export API_SERVER_PORT=8005

start_nats_server() {
    local head_url=$1
    export NATS_HOST="$head_url"
    echo "Flushing NATS store: ${NATS_STORE}..."
    rm -r "${NATS_STORE}"
    echo "Starting NATS Server..."
    nats-server -p ${NATS_PORT} --jetstream --store_dir "${NATS_STORE}" &
}

start_api_server() {
    local head_url=$1
    export VLLM_TORCH_HOST="$head_url"
    echo "Starting LLM API Server..."
    python3 -m llm.api_server \
      --tokenizer neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
      --request-plane-uri ${head_url}:${NATS_PORT} \
      --api-server-host ${API_SERVER_HOST} \
      --model-name llama \
      --api-server-port ${API_SERVER_PORT} &
}


start_context_worker() {
    local head_url=$1
    export VLLM_TORCH_HOST="$head_url"
    echo "Starting vLLM context workers..."
    CUDA_VISIBLE_DEVICES=0 \
    VLLM_WORKER_ID=0 \
    python3 -m llm.vllm.deploy \
      --context-worker-count ${VLLM_CONTEXT_WORKERS} \
      --request-plane-uri ${head_url}:${NATS_PORT} \
      --model-name neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
      --kv-cache-dtype fp8 \
      --dtype auto \
      --worker-name llama \
      --disable-async-output-proc \
      --disable-log-stats \
      --max-model-len 1000 \
      --max-batch-size 10000 \
      --gpu-memory-utilization 0.9 \
      --context-tp-size ${VLLM_CONTEXT_TP_SIZE} \
      --generate-tp-size ${VLLM_GENERATE_TP_SIZE} \
      --log-dir "/tmp/vllm_logs" &
}

start_generate_worker() {
    local head_url=$1
    export VLLM_TORCH_HOST="$head_url"
    echo "Starting vLLM generate workers..."
    CUDA_VISIBLE_DEVICES=1 \
    VLLM_WORKER_ID=1 \
    python3 -m llm.vllm.deploy \
      --generate-worker-count ${VLLM_GENERATE_WORKERS} \
      --request-plane-uri ${head_url}:${NATS_PORT} \
      --model-name neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
      --kv-cache-dtype fp8 \
      --dtype auto \
      --worker-name llama \
      --disable-async-output-proc \
      --disable-log-stats \
      --max-model-len 1000 \
      --max-batch-size 10000 \
      --gpu-memory-utilization 0.9 \
      --context-tp-size ${VLLM_CONTEXT_TP_SIZE} \
      --generate-tp-size ${VLLM_GENERATE_TP_SIZE} \
      --log-dir "/tmp/vllm_logs" &
}

case "$1" in
    context)
        if [ "$2" != "--head-url" ] || [ -z "$3" ]; then
            echo "Usage: $0 context --head-url <head url>"
            exit 1
        fi
        head_url=$3
        export API_SERVER_HOST="$head_url"
        start_nats_server
        start_api_server "$head_url"
        start_context_worker "$head_url"
        ;;
    generate)
        if [ "$2" != "--head-url" ] || [ -z "$3" ]; then
            echo "Usage: $0 generate --head-url <head url>"
            exit 1
        fi
        head_url=$3
        export API_SERVER_HOST="$head_url"
        start_generate_worker "$head_url"
        ;;
    *)
        echo "Usage: $0 {context|generate} --head-url <head url>"
        exit 1
        ;;
esac

echo "Waiting for deployment to finish startup..."
echo "Once you see all ranks connected to the server, it should be ready..."
echo "Example output:"
echo "\tRank 0 connected to the server"
echo "\t..."
echo "\tRank 1 connected to the server"
sleep 120

echo "Sending chat completions request..."
curl ${API_SERVER_HOST}:${API_SERVER_PORT}/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "llama",
  "messages": [
    {"role": "system", "content": "What is the capital of France?"}
  ],
  "temperature": 0,
  "top_p": 0.95,
  "max_tokens": 25,
  "stream": true,
  "n": 1,
  "frequency_penalty": 0.0,
  "stop": []
}'
