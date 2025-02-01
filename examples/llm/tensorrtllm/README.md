<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Disaggregated Serving with TensorRT-LLM

This example demonstrates **disaggregated serving** [^1] using Triton Distributed together with TensorRT-LLM engines. Disaggregated serving decouples the prefill (prompt encoding) and the decode (token generation) stages of large language model (LLM) inference into separate processes. This separation allows you to independently scale, optimize, and distribute resources for each stage.

In this example, you will deploy

- An **OpenAI-compatible API server** (which receives requests and streams responses).
- One or more **prefill workers** (for encoding the prompt).
- One or more **decode workers** (for generating tokens based on the encoded prompt).

## 1. Prerequisites

1. **GPU Availability**
   This setup requires at least two GPUs:
   - One GPU is typically used by the **prefill** process.
   - Another GPU is used by the **decode** process.
   In production systems with heavier loads, you will typically allocate more GPUs across multiple prefill and decode workers.

2. **NATS or Another Coordination Service**
   Triton Distributed uses NATS by default for coordination and message passing. Make sure your environment has a running NATS service accessible via a valid `nats://<address>:<port>` endpoint. By default, examples assume `nats://localhost:4223`.

3. **HuggingFace**
   - You need a HuggingFace account to download the model and set HF_TOKEN environment variable.

---

## 2. Building the Environment

The example is designed to run in a containerized environment using Triton Distributed, TensorRT-LLM, and associated dependencies. To build the container:

```bash
./container/build.sh --framework tensorrtllm
```

---

## 3. Starting the Deployment

Below is a minimal example of how to start each component of a disaggregated serving setup. The typical sequence is:

1. **Download and build model directories**
2. **Start the Context Worker(s) and Request Plane**
3. **Start the Generate Worker(s)**
1. **Start the API Server** (handles incoming requests and coordinates workers)

All components must be able to connect to the same request plane to coordinate.

### 3.1 Launch Interactive Environment

```bash
./container/run.sh --framework tensorrtllm -it
```

Note: all subsequent commands will be run in the same container for simplicity

Note: by default this command makes all gpu devices visible. Use flag

```
--gpus
```

to selectively make gpu devices visible.

### 3.2: Build model directories

```bash
export HF_TOKEN=<YOUR TOKEN>
python3 /workspace/examples/llm/tensorrtllm/scripts/prepare_models.py --tp-size 1 --model llama-3.1-8b-instruct --max-num-tokens 8192
```


After this you should see the following in `/workspace/examples/llm/tensorrtllm/operators`

```bash
|-- hf_downloads
|   `-- llama-3.1-8b-instruct
|       |-- config.json
|       |-- generation_config.json
|       |-- model-00001-of-00004.safetensors
|       |-- model-00002-of-00004.safetensors
|       |-- model-00003-of-00004.safetensors
|       |-- model-00004-of-00004.safetensors
|       |-- model.safetensors.index.json
|       |-- original
|       |   `-- params.json
|       |-- special_tokens_map.json
|       |-- tokenizer.json
|       `-- tokenizer_config.json
|-- tensorrtllm_checkpoints
|   `-- llama-3.1-8b-instruct
|       `-- NVIDIA_H100_NVL
|           `-- TP_1
|               |-- config.json
|               `-- rank0.safetensors
|-- tensorrtllm_engines
|   `-- llama-3.1-8b-instruct
|       `-- NVIDIA_H100_NVL
|           `-- TP_1
|               |-- config.json
|               `-- rank0.engine
|-- tensorrtllm_models
|   `-- llama-3.1-8b-instruct
|       `-- NVIDIA_H100_NVL
|           `-- TP_1
|               |-- context
|               |   |-- 1
|               |   |   `-- model.py
|               |   `-- config.pbtxt
|               |-- generate
|               |   |-- 1
|               |   |   `-- model.py
|               |   `-- config.pbtxt
|               |-- llama-3.1-8b-instruct
|               |   |-- 1
|               |   `-- config.pbtxt
|               |-- postprocessing
|               |   |-- 1
|               |   |   `-- model.py
|               |   `-- config.pbtxt
|               |-- preprocessing
|               |   |-- 1
|               |   |   `-- model.py
|               |   `-- config.pbtxt
|               `-- tensorrt_llm
|                   |-- 1
|                   |   `-- model.py
|                   `-- config.pbtxt
`-- triton_core_models
    |-- mock
    |   |-- 1
    |   |   `-- model.py
    |   `-- config.pbtxt
    |-- simple_postprocessing
    |   |-- 1
    |   |   `-- model.py
    |   `-- config.pbtxt
    `-- simple_preprocessing
        |-- 1
        |   `-- model.py
        `-- config.pbtxt
```


### 3.3: Deployment Example

To start a basic deployment with 1 prefill and 1 decode worker:

```bash
export MODEL_NAME="llama-3.1-8b-instruct"
python3 /workspace/examples/llm/tensorrtllm/deploy/launch_workers.py \
  --context-worker-count 1 \
  --generate-worker-count 1 \
  --model ${MODEL_NAME} \
  --initialize-request-plane \
  --disaggregated-serving \
  --request-plane-uri ${HOSTNAME}:4222 &
```

Then start the OpenAI compatible API server

```bash
python3 -m llm.api_server \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --request-plane-uri ${HOSTNAME}:4222 \
  --api-server-host ${HOSTNAME} \
  --model-name ${MODEL_NAME} &
```

### 3.4: Sending Requests

Once the API server is running (by default on `localhost:8000`), you can send OpenAI-compatible requests. For example:

```bash
curl ${HOSTNAME}:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b-instruct",
    "messages": [
      {"role": "user", "content": "Why is Roger Federer the greatest tennis player of all time?"}
    ],
    "temperature": 0,
    "top_p": 0.95,
    "max_tokens": 25,
    "stream": true,
    "n": 1,
    "frequency_penalty": 0.0,
    "stop": []
  }'
```

The above request will return a streamed response with the model’s answer.

## 4. Teardown

To tear down a deployment during local development, you can either kill the
container or the kill the relevant processes involved in the deployment.

To kill the processes being run inside the container, you can run:
```bash
pkill -SIGINT -f python3
pkill -SIGINT -f nats-server
```

You will generally want to make sure you have a clean slate between
deployments to avoid any unexpected errors.

NOTE: If you have other unrelated processes in the environment with `python3`
in the name, the `pkill` command above will terminate them as well. In this
scenario, you could select specific process IDs and use the following command
instead for each process ID replacing `<pid>` below:
```
kill -9 <pid>
```

## Known Issues & Limitations

1. **Tensor Parallelism Constraints**
   - Currently limited to TP=1 for both prefill and decode workers

## References

[^1]: Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao
Zhang. Distserve: Disaggregating prefill and decoding for goodput-optimized large language
model serving. *arXiv:2401.09670v3 [cs.DC]*, 2024.

For more details on Triton Distributed, see the [Hello World example](../../hello_world/) and [Triton Inference Server documentation](https://github.com/triton-inference-server/server).
