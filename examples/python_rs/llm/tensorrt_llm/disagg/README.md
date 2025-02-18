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

Use the following image with tritonserver, triton distributed and tensorrt_llm.

```bash
nvcr.io/pfteb4cqjzrs/playground/tritondistributed:trtllm_disagg-shreyas
```

## Launching the Environment
```
# Run image interactively
docker run -it --network host --shm-size=64G --ulimit memlock=-1 --ulimit stack=67108864 -e HF_HOME=/home/scratch.shreyasm_gpu/hf_cache --gpus=all -v .:/workspace nvcr.io/pfteb4cqjzrs/playground/tritondistributed:trtllm_disagg-shreyas
```

## Deployment Options

### 1. Disaggregated Deployment

NOTE: This will start both context and generation servers.
```bash
mpirun --allow-run-as-root -n 2 python3 -m disagg.worker --engine_args model.json &
```

Run router. This is still a WIP. It needs information about which endpoints are ctx/gen.
```bash
python3 -m disagg.router
```

Run client
```bash
python3 -m common.client \
    --prompt "Describe the capital of France" \
    --max-tokens 10 \
    --temperature 0.5
```