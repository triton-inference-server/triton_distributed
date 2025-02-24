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

http &

sleep 15

# Add model to HTTP server

llmctl http add chat-models deepseek-ai/DeepSeek-R1-Distill-Llama-8B triton-init.vllm.generate&

sleep 15

# Activate Triton environment

source /opt/triton/venv/bin/activate
cd /workspace/examples/python_rs/llm/vllm

# Start prefill workers

VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0,1 python3 -m disaggregated.prefill_worker \
            --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
            --max-model-len 100 \
            --gpu-memory-utilization 0.8 \
            --enforce-eager \
            --tensor-parallel-size 2 \
            --kv-transfer-config \
            '{"kv_connector":"TritonNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'

VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=2,3 python3 -m disaggregated.prefill_worker \
            --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
            --max-model-len 100 \
            --gpu-memory-utilization 0.8 \
            --enforce-eager \
            --tensor-parallel-size 2 \
            --kv-transfer-config \
            '{"kv_connector":"TritonNcclConnector","kv_role":"kv_producer","kv_rank":1,"kv_parallel_size":3}'



VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m disaggregated.decode_worker \
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --max-model-len 100 \
        --gpu-memory-utilization 0.8 \
        --enforce-eager \
        --tensor-parallel-size 4 \
        --kv-transfer-config \
        '{"kv_connector":"TritonNcclConnector","kv_role":"kv_consumer","kv_rank":2,"kv_parallel_size":3}'

