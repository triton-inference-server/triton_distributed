#!/bin/bash

export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_TORCH_HOST=localhost
export VLLM_TORCH_PORT=36183
export VLLM_LOGGING_LEVEL=INFO
export VLLM_DATA_PLANE_BACKEND=nccl
export PYTHONUNBUFFERED=1

export NATS_HOST=localhost
export NATS_PORT=4223
export NATS_STORE="$(mktemp -d)"
export API_SERVER_HOST=localhost
export API_SERVER_PORT=8005


# Start NATS Server
echo "Flushing NATS store: ${NATS_STORE}..."
rm -r "${NATS_STORE}"

echo "Starting NATS Server..."
nats-server -p ${NATS_PORT} --jetstream --store_dir "${NATS_STORE}" &

# Start API Server
echo "Starting LLM API Server..."
python3 -m llm.api_server \
  --tokenizer neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
  --request-plane-uri ${NATS_HOST}:${NATS_PORT} \
  --api-server-host ${API_SERVER_HOST} \
  --model-name "dummy" \
  --api-server-port ${API_SERVER_PORT} &


CUDA_VISIBLE_DEVICES=0 \
VLLM_WORKER_ID=0 \
echo "Starting dummy workers..."
python3 -m llm.vllm.deploy \
  --dummy-worker-count 1 \
  --request-plane-uri ${NATS_HOST}:${NATS_PORT} \
  --model-name neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
  --kv-cache-dtype fp8 \
  --dtype auto \
  --disable-async-output-proc \
  --disable-log-stats \
  --max-model-len 1000 \
  --max-batch-size 10000 \
  --gpu-memory-utilization 0.9

# Give deployment a minute to spin up
echo "Waiting for deployment to finish startup..."
sleep 60

# Make a Chat Completion Request
echo "Sending chat completions request..."
curl ${API_SERVER_HOST}:${API_SERVER_PORT}/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "dummy",
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
