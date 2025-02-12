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
./container/build.sh
```

## Launching the Environment
```
# Run image interactively
./container/run.sh -it

# Add vllm into the python virtual environment
source /opt/triton/venv/bin/activate
uv pip install vllm==0.7.2
```

## Deployment Options

### 1. Monolithic Deployment

Run the server and client components in separate terminal sessions:

**Terminal 1 - Server:**
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

**Terminal 2 - Client:**
```bash
# Activate virtual environment
source /opt/triton/venv/bin/activate

# Run client
cd /workspace/examples/python_rs/llm/vllm
python3 -m common.client \
    --prompt "what is the capital of france?" \
    --max-tokens 10 \
    --temperature 0.5
```

The output should look similar to:
```
Annotated(data=' Well', event=None, comment=[], id=None)
Annotated(data=' Well,', event=None, comment=[], id=None)
Annotated(data=' Well, France', event=None, comment=[], id=None)
Annotated(data=' Well, France is', event=None, comment=[], id=None)
Annotated(data=' Well, France is a', event=None, comment=[], id=None)
Annotated(data=' Well, France is a country', event=None, comment=[], id=None)
Annotated(data=' Well, France is a country located', event=None, comment=[], id=None)
Annotated(data=' Well, France is a country located in', event=None, comment=[], id=None)
Annotated(data=' Well, France is a country located in Western', event=None, comment=[], id=None)
Annotated(data=' Well, France is a country located in Western Europe', event=None, comment=[], id=None)
```


### 2. Disaggregated Deployment

This deployment option splits the model serving across prefill and decode workers, enabling more efficient resource utilization.

**Terminal 1 - Prefill Worker:**
```bash
# Activate virtual environment
source /opt/triton/venv/bin/activate

# Launch prefill worker
cd /workspace/examples/python_rs/llm/vllm
VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0,1 python3 -m disaggregated.prefill_worker \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --tensor-parallel-size 1 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":4}'
```


VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=2,3 python3 -m disaggregated.prefill_worker \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --tensor-parallel-size 1 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":1,"kv_parallel_size":4}'

**Terminal 2 - Decode Worker:**
```bash
# Activate virtual environment
source /opt/triton/venv/bin/activate

# Launch decode worker
cd /workspace/examples/python_rs/llm/vllm
VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=4,5 python3 -m disaggregated.decode_worker \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --tensor-parallel-size 2 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":2,"kv_parallel_size":4}'
```



VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=6,7 python3 -m disaggregated.decode_worker \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --tensor-parallel-size 2 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":3,"kv_parallel_size":4}'

**Terminal 3 - Client:**
```bash
# Activate virtual environment
source /opt/triton/venv/bin/activate

# Run client
cd /workspace/examples/python_rs/llm/vllm
python3 -m common.client \
    --prompt "what is the capital of france?" \
    --max-tokens 10 \
    --temperature 0.5
```

The disaggregated deployment utilizes separate GPUs for prefill and decode operations, allowing for optimized resource allocation and improved performance. For more details on the disaggregated deployment, please refer to the [vLLM documentation](https://docs.vllm.ai/en/latest/features/disagg_prefill.html).



### 3. Multi-Node Deployment

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




