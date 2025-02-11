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

# vllm tests for testing concurrencty

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
# Run image interactively - mount local workspace
./container/run.sh -it --mount-workspace --name "test"
source /opt/triton/venv/bin/activate
```

## Testing Rust Based Bindings

### Launch Worker

```
# Launch worker
cd /workspace/examples/python_rs/llm/vllm
python3 -m monolith.worker \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --max-model-len 100 \
    --enforce-eager
```

### Launch Client

```
# exec into same container
docker exec -it test bash
source /opt/triton/venv/bin/activate
# Run client
cd /workspace/examples/python_rs/llm/vllm
python3 -m common.client \
    --prompt "what is the capital of france?" \
    --max-tokens 10 \
    --temperature 0.5 \
	--request-count 100
```

## Testing Python Native

### Launch Environment

Exit out of the container and launch again.

```
# Run image interactively - mount local workspace
./container/run.sh -it --mount-workspace --name "test"
source /opt/triton/venv/bin/activate
```

### Launch Server

```
# Launch worker
cd /workspace/examples/python_rs/llm/vllm_python_native
python3 -m monolith.worker \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --max-model-len 100 \
    --enforce-eager
```

### Launch Client

```
# exec into same container
docker exec -it test bash
source /opt/triton/venv/bin/activate
cd /workspace/examples/python_rs/llm/vllm_python_native
python3 -m common.client \
    --prompt "what is the capital of france?" \
    --max-tokens 10 \
    --temperature 0.5 \
	--request-count 100
```


### Launch Client With ZMQ Response Path

```
# exec into same container
docker exec -it test bash
source /opt/triton/venv/bin/activate
cd /workspace/examples/python_rs/llm/vllm_python_native
python3 -m common.client \
    --prompt "what is the capital of france?" \
    --max-tokens 10 \
    --temperature 0.5 \
	--request-count 100 \
	--use-zmq-response-path
```
