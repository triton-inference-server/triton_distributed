<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# vLLM Integration with Triton Distributed

This example demonstrates how to use Triton Distributed to serve large language models with the vLLM engine, enabling efficient model serving with both monolithic and disaggregated deployment options.

## Prerequisites

Start required services (etcd and NATS):

   Option A: Using [Docker Compose](/runtime/rust/docker-compose.yml) (Recommended)
   ```bash
   docker-compose up -d
   ```

   Option B: Manual Setup

    - [NATS.io](https://docs.nats.io/running-a-nats-service/introduction/installation) server with [Jetstream](https://docs.nats.io/nats-concepts/jetstream)
        - example: `nats-server -js --trace`
    - [etcd](https://etcd.io) server
        - follow instructions in [etcd installation](https://etcd.io/docs/v3.5/install/) to start an `etcd-server` locally


## Building the Environment

The example is designed to run in a containerized environment using Triton Distributed, vLLM, and associated dependencies. To build the container:

```bash
# Build image
./container/build.sh --framework VLLM
```

## Launching the Environment
```
# Run image interactively
./container/run.sh --framework VLLM -it
```

## Deployment

#### 1. HTTP Server

Run the server logging (with debug level logging):
```bash
TRD_LOG=DEBUG http
```
By default the server will run on port 9992.

Add model to the server:
```bash
llmctl http add chat-models deepseek-ai/DeepSeek-R1-Distill-Llama-8B triton-init.vllm.generate
```

### 2. Workers

#### 2.1. Monolithic Deployment

In a separate terminal run the vllm worker:

```bash
# Activate virtual environment
source /opt/triton/venv/bin/activate

# Launch worker
cd /workspace/examples/python_rs/llm/vllm
python3 -m monolith.worker \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --max-model-len 100 \
    --enforce-eager
```

#### 2.2. Disaggregated Deployment

This deployment option splits the model serving across prefill and decode workers, enabling more efficient resource utilization.

**Terminal 1 - Prefill Worker:**
```bash
# Activate virtual environment
source /opt/triton/venv/bin/activate

# Launch prefill worker
cd /workspace/examples/python_rs/llm/vllm
VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0 python3 -m disaggregated.prefill_worker \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --tensor-parallel-size 1 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'
```

**Terminal 2 - Decode Worker:**
```bash
# Activate virtual environment
source /opt/triton/venv/bin/activate

# Launch decode worker
cd /workspace/examples/python_rs/llm/vllm
VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=1,2 python3 -m disaggregated.decode_worker \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --tensor-parallel-size 2 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}'
```

The disaggregated deployment utilizes separate GPUs for prefill and decode operations, allowing for optimized resource allocation and improved performance. For more details on the disaggregated deployment, please refer to the [vLLM documentation](https://docs.vllm.ai/en/latest/features/disagg_prefill.html).


### 3. Client

```bash
curl localhost:9992/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

Expected output:
```json
{
    "id": "5b04e7b0-0dcd-4c45-baa0-1d03d924010c",
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "The capital of France is Paris. Paris is a major city known for iconic landmarks like the Eiffel Tower and the Louvre Museum."
        },
        "index": 0,
        "finish_reason": "stop"
    }],
    "created": 1739548787,
    "model": "vllm",
    "object": "chat.completion",
    "usage": null,
    "system_fingerprint": null
}
```

### 4. Multi-Node Deployment

The vLLM workers can be deployed across multiple nodes by configuring the NATS and etcd connection endpoints through environment variables. This enables distributed inference across a cluster.

Set the following environment variables on each node before running the workers:

```bash
export NATS_SERVER="nats://<nats-server-host>:<nats-server-port>"
export ETCD_ENDPOINTS="http://<etcd-server-host1>:<etcd-server-port>,http://<etcd-server-host2>:<etcd-server-port>",...
```

For disaggregated deployment, you will also need to pass the `kv_ip` and `kv_port` to the workers in the `kv_transfer_config` argument:

```bash
...
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":<rank>,"kv_parallel_size":2,"kv_ip":<master_node_ip>,"kv_port":<kv_port>}'
```

### 5. Known Issues and Limitations

- vLLM is not working well with the `fork` method for multiprocessing and TP > 1. This is a known issue and a workaround is to use the `spawn` method instead. See [vLLM issue](https://github.com/vllm-project/vllm/issues/6152).
- `kv_rank` of `kv_producer` must be smaller than of `kv_consumer`.
- Instances with the same `kv_role` must have the same `--tensor-parallel-size`.
- Currently only `--pipeline-parallel-size 1` is supported for XpYd disaggregated deployment.


