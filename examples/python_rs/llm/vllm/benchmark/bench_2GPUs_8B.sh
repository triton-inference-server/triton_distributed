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

ENDPOINT_HOST="localhost"
ENDPOINT_PORT="8080"
ENDPOINT_URL="http://$ENDPOINT_HOST:$ENDPOINT_PORT"

MEAN_INPUT_TOKENS=3000
MEAN_OUTPUT_TOKENS=150

MAX_MODEL_LEN=$((MEAN_INPUT_TOKENS + MEAN_OUTPUT_TOKENS + 100))

CHAT_MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Start NATS server

nats-server \
    -js \
    --trace \
    -p 4222 \
    -m 8222 &

sleep 15

# Start etcd server

etcd

sleep 15

# Start HTTP server endpoint



http --host $ENDPOINT_HOST --port $ENDPOINT_PORT &

sleep 15

# Add model to HTTP server

llmctl http add chat-models $CHAT_MODEL_NAME triton-init.vllm.generate&

sleep 15

# Activate Triton environment

source /opt/triton/venv/bin/activate
cd /workspace/examples/python_rs/llm/vllm


# Start prefill worker

VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0 python3 -m disaggregated.prefill_worker \
            --model $CHAT_MODEL_NAME \
            --max-model-len $MAX_MODEL_LEN \
            --gpu-memory-utilization 0.8 \
            --enforce-eager \
            --tensor-parallel-size 1 \
            --kv-transfer-config \
            '{"kv_connector":"TritonNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'

# Start decode worker

VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=1 python3 -m disaggregated.decode_worker \
        --model $CHAT_MODEL_NAME \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization 0.8 \
        --enforce-eager \
        --tensor-parallel-size 1 \
        --kv-transfer-config \
        '{"kv_connector":"TritonNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}'



# Run benchmark

CONFIG_PREFIX="prefill_tp1dp1_generate_t1d1

for p in {0..8}; do
    CONCURRENCY=$((2**p))
    CONCURRENCY_STR="$CONCURRENCY.0"
    # If request count should be 10 times concurrency, but not less than 100 and
    # not more than 2 times concurrency if it is more than 256

    # Step 1: Set request_count to 10 times concurrency
    REQUEST_COUNT=$(( 10 * CONCURRENCY ))

    # Step 2: Ensure request_count is at least 100
    if [ "$REQUEST_COUNT" -lt 100 ]; then
        REQUEST_COUNT=100
    fi

    # Step 3: If concurrency > 256, request_count must not exceed 2 * concurrency
    if [ "$CONCURRENCY" -gt 256 ]; then
        MAX_COUNT=$(( 2 * CONCURRENCY ))
        if [ "$REQUEST_COUNT" -gt "$MAX_COUNT" ]; then
            REQUEST_COUNT=$MAX_COUNT
        fi
    fi
    DATE=$(date +%Y%m%d_%H%M%S)
    ARTIFACT_DIR="./artifacts/$CONFIG_PREFIX/concurrency_$CONCURRENCY_STR"_$DATE
    python3 /workspace/examples/python_rs/llm/vllm/benchmark/run_benchmark.py \
        --isl-cached 0 \
        --isl-uncached $MEAN_INPUT_TOKENS \
        --osl $MEAN_OUTPUT_TOKENS \
        --model $CHAT_MODEL_NAME \
        --tokenizer $CHAT_MODEL_NAME \
        --url localhost:8080 \
        --artifact-dir $ARTIFACT_DIR \
        --request-count $REQUEST_COUNT \
        --load-type concurrency \
        --load-value $CONCURRENCY
done
