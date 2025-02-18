#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# FIXME: Convert this script to README steps

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

export NATS_STORE="$(mktemp -d)"
export API_SERVER_HOST=localhost
export API_SERVER_PORT=8005


# Start NATS Server
echo "Flushing NATS store: ${NATS_STORE}..."
rm -r "${NATS_STORE}"

echo "Starting NATS Server..."
nats-server -p ${DEFAULT_REQUESTS_PORT} --addr ${DEFAULT_REQUESTS_HOST} --jetstream --store_dir "${NATS_STORE}" &


# Start API Server
echo "Starting LLM API Server..."
python3 -m llm.api_server \
  --tokenizer neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
  --request-plane-uri ${DEFAULT_REQUESTS_URI} \
  --api-server-host ${API_SERVER_HOST} \
  --model-name llama \
  --api-server-port ${API_SERVER_PORT} &


# Start VLLM Worker 0
echo "Starting vLLM context workers..."
CUDA_VISIBLE_DEVICES=0 \
VLLM_WORKER_ID=0 \
python3 -m llm.vllm.deploy \
  --context-worker-count ${VLLM_CONTEXT_WORKERS} \
  --request-plane-uri ${DEFAULT_REQUESTS_URI} \
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

# Start VLLM Worker 1
echo "Starting vLLM generate workers..."
CUDA_VISIBLE_DEVICES=1 \
VLLM_WORKER_ID=1 \
python3 -m llm.vllm.deploy \
  --generate-worker-count ${VLLM_GENERATE_WORKERS} \
  --request-plane-uri ${DEFAULT_REQUESTS_URI} \
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

# NOTE: It may take more than a minute for the vllm worker to start up
# if the model weights aren't cached and need to be downloaded.
echo "Waiting for deployment to finish startup..."
echo "Once you see all ranks connected to the server, it should be ready..."
echo "Example output:"
echo "\tRank 0 connected to the server"
echo "\t..."
echo "\tRank 1 connected to the server"
sleep 120

# Make a Chat Completion Request
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
