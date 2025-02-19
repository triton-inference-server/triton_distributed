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

# TensorRT-LLM Integration with Triton Distributed [WIP]

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

#### Option 1: Single-node, single-GPU

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

#### Option 2: Single-node, multi-GPU

Change the `tensor_parallel_size` in the `model.json` file to the number of GPUs you want to use on the node.

**Server:**

```bash
# Run worker
mpirun -n 1 --oversubscribe --allow-run-as-root python3 -m monolith.worker --engine_args model.json &
```

This should load the model on specified many GPUs.

#### Option 3: Multi-node, multi-GPU

We will use a wrapper script to launch the server on multiple nodes within the same mpi comm world on slurm cluster.
Change the `tensor_parallel_size` in the `model.json` file to 16. Each node in EOS has 8 GPUs. Setting the number of
tasks to 16 will use all the GPUs on both the nodes.

**Server [SLURM]:**

```bash
# Allocate 2 nodes
salloc -A coreai_tritoninference_triton3 -N2 -p batch -J coreai_tritoninference_triton3-test:test -t 04:00:00
# Run worker
srun --mpi pmix -N 2 --ntasks 16 --ntasks-per-node=8 --container-image gitlab-master.nvidia.com#tanmayv/triton-3:trtllm-18-2 --container-mounts "/lustre/fsw/coreai_tritoninference_triton3/tanmayv/triton_distributed:/workspace"  bash -c 'cd /workspace/examples/python_rs/llm/tensorrt_llm && ./trtllm_worker_launch.sh python3 -m monolith.worker --engine_args model.json' &

```

Used EOS to test the multi-node, multi-GPU deployment. After the worker is launched, the output should look similar to:

```
tanmayv@eos0302:/lustre/fsw/coreai_tritoninference_triton3/tanmayv/triton_distributed/examples/python_rs/llm/tensorrt_llm$ pyxis: imported docker image: gitlab-master.nvidia.com#tanmayv/triton-3:trtllm-18-2
pyxis: imported docker image: gitlab-master.nvidia.com#tanmayv/triton-3:trtllm-18-2
mpi_rank: 5
5 launch worker ...
mpi_rank: 7
7 launch worker ...
mpi_rank: 2
2 launch worker ...
mpi_rank: 6
6 launch worker ...
mpi_rank: 1
1 launch worker ...
mpi_rank: 0
0 run python3 -m monolith.worker --engine_args model.json ...
mpi_rank: 3
mpi_rank: 4
3 launch worker ...
4 launch worker ...
mpi_rank: 11
11 launch worker ...
mpi_rank: 13
13 launch worker ...
mpi_rank: 8
8 launch worker ...
mpi_rank: 10
10 launch worker ...
mpi_rank: 9
9 launch worker ...
mpi_rank: 14
14 launch worker ...
mpi_rank: 12
12 launch worker ...
mpi_rank: 15
15 launch worker ...
```

NOTE: These instructions just allow us to launch the worker on multiple nodes and load the model on TP16.
WIP: Launching the client, etcd, nats server on a separate node.

**Client:**

NOTE: The client is not yet implemented for multi-node, multi-GPU deployment.

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
- Support and test TP>1: single-node , multi-GPU - Done
- Support and test TP>1: multi-node, multi-GPU
- Support and test dissagregated serving: single-node
- Support and test dissagregated serving: multi-node

NOTE: For multi-node deployment we need to handle the MPI_WORLD setup.

