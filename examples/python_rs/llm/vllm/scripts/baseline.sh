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

# LIMITATIONS:
# - Must use a single GPU for workers as CUDA_VISIBLE_DEVICES is set to a fixed value
# - Must use a single node

if [ $# -lt 2 ]; then
    echo "Usage: $0 <number_of_workers> <routing_strategy> <log_dir_name> [model_name] [model_args] [chat_endpoint_name] [completions_endpoint_name]"
    echo "Error: Must specify at least number of workers and routing strategy and log_dir_name"
    echo "Optional: model_name (default: deepseek-ai/DeepSeek-R1-Distill-Llama-8B)"
    echo "Optional: model_args (quoted string with model arguments)"
    echo "Optional: chat_endpoint_name (default: triton-init.vllm.chat/completions)"
    echo "Optional: completions_endpoint_name (default: triton-init.vllm.completions)"
    exit 1
fi

# If using Cache, can set this to not check HF
# export HF_HUB_OFFLINE=1

# Required for Qwen2.5 R1 Distilled in order to set --block-size 64 and --kv-cache-dtype fp8
# uv pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.5/
# export VLLM_ATTENTION_BACKEND=FLASHINFER

NUM_WORKERS=$1
ROUTING_STRATEGY=$2
LOG_DIR_NAME=$3
MODEL_NAME=${4:-"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}
CUSTOM_MODEL_ARGS=$5
CHAT_ENDPOINT_NAME=${6:-"triton-init.vllm.chat/completions"}
COMPLETIONS_ENDPOINT_NAME=${7:-"triton-init.vllm.completions"}
VALID_STRATEGIES=("random")
SESSION_NAME="v"
WORKDIR="/workspace/examples/python_rs/llm/vllm"
INIT_CMD="source /opt/triton/venv/bin/activate && cd $WORKDIR"
# Default model args
DEFAULT_MODEL_ARGS="--model $MODEL_NAME \
    --tokenizer $MODEL_NAME \
    --enable-prefix-caching \
    --block-size 64"

# Use custom model args if provided, otherwise use default
if [ -n "$CUSTOM_MODEL_ARGS" ]; then
    MODEL_ARGS="$CUSTOM_MODEL_ARGS"
    echo "Using custom model arguments"
else
    MODEL_ARGS="$DEFAULT_MODEL_ARGS"
    echo "Using default model arguments"
fi

# Create logs directory if it doesn't exist
LOGS_DIR="/workspace/logs/$LOG_DIR_NAME"
mkdir -p $LOGS_DIR
chmod -R 775 $LOGS_DIR

if [[ ! " ${VALID_STRATEGIES[@]} " =~ " ${ROUTING_STRATEGY} " ]]; then
    echo "Error: Invalid routing strategy. Must be one of: ${VALID_STRATEGIES[*]}"
    exit 1
fi
########################################################
# HTTP Server
########################################################
HTTP_CMD="TRD_LOG=DEBUG http |& tee $LOGS_DIR/http.log"
tmux new-session -d -s "$SESSION_NAME-http"
tmux send-keys -t "$SESSION_NAME-http" "$INIT_CMD && $HTTP_CMD" C-m

########################################################
# LLMCTL
########################################################
LLMCTL_CMD="sleep 5 && \
    llmctl http remove chat $MODEL_NAME && \
    llmctl http remove completions $MODEL_NAME && \
    llmctl http add chat $MODEL_NAME $CHAT_ENDPOINT_NAME && \
    llmctl http add completions $MODEL_NAME $COMPLETIONS_ENDPOINT_NAME && \
    llmctl http list |& tee $LOGS_DIR/llmctl.log"
tmux new-session -d -s "$SESSION_NAME-llmctl"
tmux send-keys -t "$SESSION_NAME-llmctl" "$INIT_CMD && $LLMCTL_CMD" C-m

########################################################
# Workers
########################################################
WORKER_CMD="RUST_LOG=info python3 -m monolith.worker $MODEL_ARGS"

for i in $(seq 1 $NUM_WORKERS); do
        tmux new-session -d -s "$SESSION_NAME-$i"
done

for i in $(seq 1 $NUM_WORKERS); do
        tmux send-keys -t "$SESSION_NAME-$i" "$INIT_CMD && CUDA_VISIBLE_DEVICES=$((i-1)) $WORKER_CMD |& tee $LOGS_DIR/worker-$i.log" C-m
done
