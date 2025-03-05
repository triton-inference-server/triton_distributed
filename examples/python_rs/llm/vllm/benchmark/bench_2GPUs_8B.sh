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

set -e
set -x

ENDPOINT_HOST="localhost"
ENDPOINT_PORT="8080"
ENDPOINT_URL="http://$ENDPOINT_HOST:$ENDPOINT_PORT"

MEAN_INPUT_TOKENS=3000
MEAN_OUTPUT_TOKENS=150
IO_PREFIX="isl_cached_0_isl_uncached_${MEAN_INPUT_TOKENS}_osl_${MEAN_OUTPUT_TOKENS}"

MAX_MODEL_LEN=$((MEAN_INPUT_TOKENS + MEAN_OUTPUT_TOKENS + 100))

CHAT_MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

nats-server \
    -js \
    -p 4222 \
    -m 8222 &

echo "Waiting for NATS server to start..."
sleep 5

echo "Starting etcd server..."
etcd &

echo "Waiting for etcd server to start..."
sleep 5

echo "Starting HTTP server endpoint..."
http --host $ENDPOINT_HOST --port $ENDPOINT_PORT &

echo "Waiting for HTTP server to start..."
sleep 5

echo "Adding model to HTTP server..."
llmctl http add chat-models $CHAT_MODEL_NAME triton-init.vllm.generate

echo "Waiting for model to be added..."
sleep 15

echo "Activating Triton environment..."

source /opt/triton/venv/bin/activate
cd /workspace/examples/python_rs/llm/vllm


echo "Starting prefill worker..."

VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0 python3 -m disaggregated.prefill_worker \
            --model $CHAT_MODEL_NAME \
            --max-model-len $MAX_MODEL_LEN \
            --gpu-memory-utilization 0.8 \
            --tensor-parallel-size 1 \
            --kv-transfer-config \
            '{"kv_connector":"TritonNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}' &

echo "Starting decode worker..."

VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=1 python3 -m disaggregated.decode_worker \
        --model $CHAT_MODEL_NAME \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization 0.8 \
        --tensor-parallel-size 1 \
        --kv-transfer-config \
        '{"kv_connector":"TritonNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}' &


echo "Running benchmark..."

CONFIG_PREFIX="prefill_tp1dp1_generate_t1d1"

ARTIFACT_DIR_PREFIX="./artifacts/$IO_PREFIX/$CONFIG_PREFIX"

mkdir -p $ARTIFACT_DIR_PREFIX

for p in {0..8}; do
    CONCURRENCY=$((2**p))
    echo "Running benchmark for concurrency $CONCURRENCY..."
    python3 /workspace/examples/python_rs/llm/vllm/benchmark/run_benchmark.py \
        --isl-cached 0 \
        --isl-uncached $MEAN_INPUT_TOKENS \
        --osl $MEAN_OUTPUT_TOKENS \
        --model $CHAT_MODEL_NAME \
        --tokenizer $CHAT_MODEL_NAME \
        --url $ENDPOINT_URL \
        --artifact-dir $ARTIFACT_DIR_PREFIX \
        --load-type concurrency \
        --load-value $CONCURRENCY
done


pkill -f nats-server   || true
pkill -f etcd          || true
pkill -f "http --host $ENDPOINT_HOST --port $ENDPOINT_PORT" || true
pkill -f python3 || true

# Start vllm serve baseline using 2 GPUs

VLLM_CONFIGURE_LOGGING=0 vllm serve \
    $CHAT_MODEL_NAME \
    --tensor-parallel-size 2 \
    --port $ENDPOINT_PORT \
    --host $ENDPOINT_HOST &

sleep 15

echo "Running vllm serve baseline benchmark..."

CONFIG_PREFIX="baseline_tp2dp1"

ARTIFACT_DIR_PREFIX="./artifacts/$IO_PREFIX/$CONFIG_PREFIX"

mkdir -p $ARTIFACT_DIR_PREFIX

for p in {0..8}; do
    CONCURRENCY=$((2**p))
    echo "Running benchmark for concurrency $CONCURRENCY..."
    python3 /workspace/examples/python_rs/llm/vllm/benchmark/run_benchmark.py \
        --isl-cached 0 \
        --isl-uncached $MEAN_INPUT_TOKENS \
        --osl $MEAN_OUTPUT_TOKENS \
        --model $CHAT_MODEL_NAME \
        --tokenizer $CHAT_MODEL_NAME \
        --url $ENDPOINT_URL \
        --artifact-dir $ARTIFACT_DIR_PREFIX \
        --load-type concurrency \
        --load-value $CONCURRENCY
done

# Kill all python3 processes from vllm serve

pkill -f python3 || true

echo "Generating plots..."

# Seaborn and matplotlib are not installed in the Triton environment
deactivate
python3 /workspace/examples/python_rs/llm/vllm/benchmark/process_gap_results.py ./artifacts/

echo "Done!"