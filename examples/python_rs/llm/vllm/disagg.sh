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
# - Must have at least 2 GPUs since CUDA_VISIBLE_DEVICES is hard-coded to 0 and 1
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
#WORKDIR="/workspace/examples/python_rs/llm/vllm"
WORKDIR="/workspace/examples/vllm"
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
HTTP_CMD="TRD_LOG=DEBUG http"
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

########################################################
# Processor
########################################################

# skip

########################################################
# Router
########################################################

# skip

########################################################
# Prefill
########################################################
PREFILL_CMD="VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0 \
    python3 -m disaggregated.prefill_worker \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --tensor-parallel-size 1 \
    --kv-transfer-config \
    '{\"kv_connector\":\"PyNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_rank\":0,\"kv_parallel_size\":2}'"

tmux select-pane -t 2
tmux send-keys "$INIT_CMD && $PREFILL_CMD" C-m

########################################################
# Decode
########################################################
DECODE_CMD="VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=1 \
    python3 -m disaggregated.decode_worker \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --tensor-parallel-size 1 \
    --kv-transfer-config \
    '{\"kv_connector\":\"PyNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_rank\":0,\"kv_parallel_size\":2}'"

tmux select-pane -t 3
tmux send-keys "$INIT_CMD && $DECODE_CMD" C-m

