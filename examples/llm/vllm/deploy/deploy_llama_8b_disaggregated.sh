#! /bin/bash

export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_TORCH_HOST=localhost
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



# Start NATS Server
#nats-server -p 4223 --jetstream &

# Start API Server
#python3 -m llm.api_server \
#  --tokenizer neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
#  --request-plane-uri localhost:4223 \
#  --api-server-host localhost \
#  --model-name llama \
#  --api-server-port 8005 &




CUDA_VISIBLE_DEVICES=0 \
VLLM_WORKER_ID=0 \
python3 -m llm.vllm.deploy \
  --context-worker-count 1 \
  --request-plane-uri localhost:4223 \
  --model-name neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
  --kv-cache-dtype fp8 \
  --dtype auto \
  --worker-name llama \
  --disable-async-output-proc \
  --disable-log-stats \
  --max-model-len 1000 \
  --max-batch-size 10000 \
  --gpu-memory-utilization 0.9 \
  --context-tp-size 1 \
  --generate-tp-size 1 &

# Start VLLM Worker 1
CUDA_VISIBLE_DEVICES=1 \
VLLM_WORKER_ID=1 \
python3 -m llm.vllm.deploy \
  --generate-worker-count 1 \
  --request-plane-uri localhost:4223 \
  --model-name neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
  --kv-cache-dtype fp8 \
  --dtype auto \
  --worker-name llama \
  --disable-async-output-proc \
  --disable-log-stats \
  --max-model-len 1000 \
  --max-batch-size 10000 \
  --gpu-memory-utilization 0.9 \
  --context-tp-size 1 \
  --generate-tp-size 1 &

# Make a Chat Completion Request
curl localhost:8005/v1/chat/completions \
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
}' &
