#!/bin/bash -e
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
# - Only uses a single worker for simple sanity test
# - Must use a single node

if [ $# -gt 2 ]; then
    echo "Usage: $0 [model_name] [endpoint_name]"
    echo "Optional: model_name (default: deepseek-ai/DeepSeek-R1-Distill-Llama-8B)"
    echo "Optional: endpoint_name (default: triton-init.vllm.generate)"
    exit 1
fi

MODEL_NAME=${1:-"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}
ENDPOINT_NAME=${2:-"triton-init.vllm.generate"}
SESSION_NAME="vllm_disagg"
WORKDIR="$(dirname $0)/.."
INIT_CMD="cd $WORKDIR"

########################################################
# TMUX SESSION SETUP
########################################################

# Start new session
tmux new-session -d -s "$SESSION_NAME"

# Split into 4 equal panes
tmux split-window -h
tmux split-window -v
tmux select-pane -t 0
tmux split-window -v

########################################################
# HTTP Server
########################################################
HTTP_HOST="localhost"
HTTP_PORT=8080
HTTP_CMD="TRD_LOG=DEBUG http --host ${HTTP_HOST} --port ${HTTP_PORT}"
tmux select-pane -t 0
tmux send-keys "$INIT_CMD && $HTTP_CMD" C-m

########################################################
# LLMCTL
########################################################
LLMCTL_CMD="sleep 5 && llmctl http remove chat-model $MODEL_NAME && \
    llmctl http add chat-model $MODEL_NAME $ENDPOINT_NAME && \
    llmctl http list chat-model"
tmux select-pane -t 1
tmux send-keys "$INIT_CMD && $LLMCTL_CMD" C-m

CLIENT_CMD="python3 -m common.client \
    --prompt \"In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden.\" \
    --component preprocess \
    --max-tokens 10 \
    --temperature 0.5"
# Prepare a client command for a quick test, but don't execute it since the server
# needs to spin up first.
tmux send-keys "$CLIENT_CMD"

########################################################
# Router
########################################################
ROUTER_CMD="RUST_LOG=info python3 -m kv_router.router \
    --routing-strategy prefix"

tmux select-pane -t 2
tmux send-keys "$INIT_CMD && $ROUTER_CMD" C-m

########################################################
# Worker
########################################################

# FIXME: Only run 1 worker for simplicity in sanity test, but can be 2+ workers
# for an actual functional test to see the routing behavior between each worker.
WORKER_CMD="RUST_LOG=info python3 -m kv_router.worker \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --enable-prefix-caching \
    --block-size 64 \
    --max-model-len 16384"

tmux select-pane -t 3
tmux send-keys "$INIT_CMD && $WORKER_CMD" C-m
tmux attach-session -t "$SESSION_NAME"
