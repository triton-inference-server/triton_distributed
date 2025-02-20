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

# TensorRT-LLM Integration with Triton Distributed

This example demonstrates how to use Triton Distributed to serve large language models with the tensorrt_llm engine, enabling efficient model serving with monolithic option.

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

*Note*: This example is work in progress.

## Building the Environment [WIP]

The example is designed to run in a containerized environment using Triton Distributed, tensorrt_llm, and associated dependencies. To build the container:

REVISIT: Currently using special container from Kris to build the image with tensorrt_llm supporting pytorch workflow in LLM API
TODO: Work on better instructions for building the container with latest tensorrt_llm.

```bash
# Build image
./container/build.sh --framework TENSORRTLLM --base-image gitlab-master.nvidia.com:5005/dl/dgx/tritonserver/tensorrt-llm/amd64 --base-image-tag krish-multinode-test
```

## Launching the Environment
```
# Run image interactively
./container/run.sh --framework TENSORRTLLM -it
```

## Deployment Options

### 1. Monolithic Deployment

Run the server and client components in separate terminal sessions:

**Server:**

REVISIT: I had to install some dependencies manually in the container to get this to work.
TODO: Move these extra dependencies to the container build step.

```bash
# Install dependencies
pip3 install flash_attn

# Launch worker in background
cd /workspace/examples/python_rs/llm/tensorrt_llm
python3 -m monolith.worker --engine_args model.json &
```

Upon successful launch, the output should look similar to:

```
[TensorRT-LLM][INFO] KV cache block reuse is disabled
[TensorRT-LLM][INFO] Max KV cache pages per sequence: 2048
[TensorRT-LLM][INFO] Number of tokens per block: 64.
[TensorRT-LLM][INFO] [MemUsageChange] Allocated 26.91 GiB for max tokens in paged KV cache (220480).
[02/14/2025-09:38:53] [TRT-LLM] [I] max_seq_len=131072, max_num_requests=2048, max_num_tokens=8192
[02/14/2025-09:38:53] [TRT-LLM] [I] Engine loaded and ready to serve...
```

**Client:**

```bash

# Run client
python3 -m common.client \
    --prompt "Describe the capital of France" \
    --max-tokens 10 \
    --temperature 0.5
```

The output should look similar to:
```
Annotated(data=',', event=None, comment=[], id=None)
Annotated(data=', Paris', event=None, comment=[], id=None)
Annotated(data=', Paris,', event=None, comment=[], id=None)
Annotated(data=', Paris, in', event=None, comment=[], id=None)
Annotated(data=', Paris, in terms', event=None, comment=[], id=None)
Annotated(data=', Paris, in terms of', event=None, comment=[], id=None)
Annotated(data=', Paris, in terms of its', event=None, comment=[], id=None)
Annotated(data=', Paris, in terms of its history', event=None, comment=[], id=None)
Annotated(data=', Paris, in terms of its history,', event=None, comment=[], id=None)
Annotated(data=', Paris, in terms of its history, culture', event=None, comment=[], id=None)
```

Next steps:
- Building container with latest tensorrt_llm wheel.
- Support and test TP>1: single-node , multi-GPU
- Support and test TP>1: multi-node, multi-GPU
- Support and test dissagregated serving: single-node
- Support and test dissagregated serving: multi-node

NOTE: For multi-node deployment we need to handle the MPI_WORLD setup.


### 2. Disaggregated Deployment

**Environment**
This is the latest image with tensorrt_llm supporting distributed serving with pytorch workflow in LLM API.
TODO: can use TRT-LLM main in dockerfile now.


Run the container interactively with the following command:
```bash
docker run -it --network host --shm-size=64G --ulimit memlock=-1 --ulimit stack=67108864 -e HF_HOME=/path/to/hf_cache --gpus=all -v .:/workspace IMAGE
```

**TRTLLM LLMAPI Disaggregated config file**
Define disaggregated config file as shown below. The only important sections are the model, context_servers and generation_servers.

```yaml
model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
##################
# The following fields are unused in this example.
# They are only needed for the TRT-LLM when using TRT-LLM's disaggregated server which we do not use here.
hostname: localhost
port: 8000
backend: "pytorch"
##################
context_servers: # servers is a list of servers
  num_instances: 1 # The number of servers to launch
  gpu_fraction: 0.25
  tp_size: 1 # each server will be a TP=1 instance
  pp_size: 1 # each server will be a PP=1 instance
  urls:
      - "localhost:8001" # The URL of the server.
generation_servers:
  num_instances: 1
  gpu_fraction: 0.25
  tp_size: 1
  pp_size: 1
  urls:
      - "localhost:8002"
```

Consider the following example for 1 TP2 context server and 2 TP1 generation servers.

```yaml
model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
...
context_servers:
  num_instances: 1
  gpu_fraction: 0.25
  tp_size: 2
  pp_size: 1
  urls:
      - "localhost:8001"
generation_servers:
  num_instances: 2
  gpu_fraction: 0.25
  tp_size: 1
  pp_size: 1
  urls:
      - "localhost:8002"
      - "localhost:8003"
```

**Launch the servers**

Launch context and generation servers.\
WORLD_SIZE is the total number of workers covering all the servers described in disaggregated configuration.\
Reason: 2 TP2 generation servers are 2 servers but 4 workers/mpi executor.

```bash
cd /workspace/examples/python_rs/llm/tensorrt_llm/
mpirun --allow-run-as-root -n WORLD_SIZE python3 -m disagg.worker --engine_args model.json &
```

**Launch the router**

```bash
cd /workspace/examples/python_rs/llm/tensorrt_llm/
python3 -m disagg.router --disagg-config disagg/disagg_config.yaml &
```

**Send Requests**

```bash
cd /workspace/examples/python_rs/llm/tensorrt_llm/
python3 -m common.client \
    --prompt "Describe the capital of France" \
    --max-tokens 10 \
    --temperature 0.5 \
    --component router
```

For more details on the disaggregated deployment, please refer to the [TRT-LLM example](#TODO).


### 3. Multi-Node Deployment

#### 3.1 Monolithic Deployment

#### 3.2 Disaggregated Deployment

To run the disaggregated deployment across multiple nodes, we need to launch the servers using MPI, pass the correct NATS and etcd endpoints to each server and update the LLMAPI disaggregated config file to use the correct endpoints.

1. Allocate nodes
The following command allocates nodes for the job and returns the allocated nodes.
```bash
salloc -A ACCOUNT -N NUM_NODES -p batch -J JOB_NAME -t HH:MM:SS
```

You can use `squeue -u $USER` to check the URLs of the allocated nodes. These URLs should be added to the TRTLLM LLMAPI disaggregated config file as shown below.
```yaml
model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
...
context_servers:
  num_instances: 2
  gpu_fraction: 0.25
  tp_size: 2
  pp_size: 1
  urls:
      - "node1:8001"
      - "node2:8002"
generation_servers:
  num_instances: 2
  gpu_fraction: 0.25
  tp_size: 2
  pp_size: 1
  urls:
      - "node2:8003"
      - "node2:8004"
```

2. Start the NATS and ETCD endpoints

Use the following commands. These commands will require downloading [NATS.io](https://docs.nats.io/running-a-nats-service/introduction/installation) and [ETCD](https://etcd.io/docs/v3.5/install/):
```bash
./nats-server -js --trace
./etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379
```

Export the correct NATS and etcd endpoints.
```bash
export NATS_SERVER="nats://node1:4222"
export ETCD_ENDPOINTS="http://node1:2379,http://node2:2379"
```

3. Launch the workers from node1 or login node. WORLD_SIZE is similar to single node deployment. Update the `model.json` to point to the new disagg config file.
```bash
srun --mpi pmix -N NUM_NODES --ntasks WORLD_SIZE --ntasks-per-node=WORLD_SIZE --no-container-mount-home --overlap --container-image IMAGE --output batch_%x_%j.log --err batch_%x_%j.err --container-mounts PATH_TO_TRITON_DISTRIBUTED:/workspace --container-env=NATS_SERVER,ETCD_ENDPOINTS bash -c 'cd /workspace/examples/python_rs/llm/tensorrt_llm && python3 -m disagg.worker --engine_args model.json' &
```

Once the workers are launched, you should see the output similar to the following in the worker logs.
```
[TensorRT-LLM][INFO] [MemUsageChange] Allocated 18.88 GiB for max tokens in paged KV cache (1800032).
[02/20/2025-07:10:33] [TRT-LLM] [I] max_seq_len=2048, max_num_requests=2048, max_num_tokens=8192
[02/20/2025-07:10:33] [TRT-LLM] [I] Engine loaded and ready to serve...
[02/20/2025-07:10:33] [TRT-LLM] [I] max_seq_len=2048, max_num_requests=2048, max_num_tokens=8192
[TensorRT-LLM][INFO] Number of tokens per block: 32.
[TensorRT-LLM][INFO] [MemUsageChange] Allocated 18.88 GiB for max tokens in paged KV cache (1800032).
[02/20/2025-07:10:33] [TRT-LLM] [I] max_seq_len=2048, max_num_requests=2048, max_num_tokens=8192
[02/20/2025-07:10:33] [TRT-LLM] [I] Engine loaded and ready to serve...
```

4. Launch the router from node1 or login node.
```bash
srun --mpi pmix -N 1 --ntasks 1 --ntasks-per-node=1 --overlap --container-image IMAGE --output batch_router_%x_%j.log --err batch_router_%x_%j.err --container-mounts PATH_TO_TRITON_DISTRIBUTED:/workspace  --container-env=NATS_SERVER,ETCD_ENDPOINTS bash -c 'cd /workspace/examples/python_rs/llm/tensorrt_llm && python3 -m disagg.router --disagg-config disagg/disagg_config_multi_node.yaml' &
```

5. Send requests to the router.
```bash
srun --mpi pmix -N 1 --ntasks 1 --ntasks-per-node=1 --overlap --container-image IMAGE --output batch_client_%x_%j.log --err batch_client_%x_%j.err --container-mounts PATH_TO_TRITON_DISTRIBUTED:/workspace  --container-env=NATS_SERVER,ETCD_ENDPOINTS bash -c 'cd /workspace/examples/python_rs/llm/tensorrt_llm && python3 -m common.client --prompt "Describe the capital of France" --max-tokens 10 --temperature 0.5 --component router' &
```

Finally, you should see the output similar to the following in the client logs.

```
Annotated(data='and', event=None, comment=[], id=None)
Annotated(data='and its', event=None, comment=[], id=None)
Annotated(data='and its significance', event=None, comment=[], id=None)
Annotated(data='and its significance in', event=None, comment=[], id=None)
Annotated(data='and its significance in the', event=None, comment=[], id=None)
Annotated(data='and its significance in the country', event=None, comment=[], id=None)
Annotated(data="and its significance in the country'", event=None, comment=[], id=None)
Annotated(data="and its significance in the country's", event=None, comment=[], id=None)
Annotated(data="and its significance in the country's history", event=None, comment=[], id=None)
Annotated(data="and its significance in the country's history.", event=None, comment=[], id=None)
```
