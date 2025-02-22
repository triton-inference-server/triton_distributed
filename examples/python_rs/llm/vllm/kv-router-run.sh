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

#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Usage: $0 <number_of_workers> <routing_strategy> [model_name]"
    echo "Error: Must specify at least number of workers and routing strategy"
    exit 1
fi

NUM_WORKERS=$1
ROUTING_STRATEGY=$2
MODEL_NAME=${3:-"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}
VALID_STRATEGIES=("prefix" "round_robin" "random")

if [[ ! " ${VALID_STRATEGIES[@]} " =~ " ${ROUTING_STRATEGY} " ]]; then
    echo "Error: Invalid routing strategy. Must be one of: ${VALID_STRATEGIES[*]}"
    exit 1
fi

SESSION_NAME="v"
WORKDIR="/workspace/examples/python_rs/llm/vllm"
INIT_CMD="source /opt/triton/venv/bin/activate && cd $WORKDIR"

ROUTER_CMD="RUST_LOG=info python3 -m kv_router.router \
    --routing-strategy $ROUTING_STRATEGY \
    --min-workers $NUM_WORKERS "

tmux new-session -d -s "$SESSION_NAME-router"

tmux send-keys -t "$SESSION_NAME-router" "$INIT_CMD && $ROUTER_CMD" C-m

WORKER_CMD="RUST_LOG=info python3 -m kv_router.worker \
    --model $MODEL_NAME \
    --tokenizer $MODEL_NAME \
    --enable-prefix-caching \
    --block-size 64 \
    --max-model-len 16384 "

for i in $(seq 1 $NUM_WORKERS); do
        tmux new-session -d -s "$SESSION_NAME-$i"
done

for i in $(seq 1 $NUM_WORKERS); do
        tmux send-keys -t "$SESSION_NAME-$i" "$INIT_CMD && CUDA_VISIBLE_DEVICES=$((i-1)) $WORKER_CMD" C-m
done
