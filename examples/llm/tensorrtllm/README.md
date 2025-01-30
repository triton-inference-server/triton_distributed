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

4. **Supported GPUs**
   - For FP8 usage, GPUs with **Compute Capability >= 8.9** are required.
   - If you have older GPUs, consider BF16/FP16 precision variants instead of `FP8`. (See [below](#model-precision-variants).)

5. **HuggingFace**
   - You need a HuggingFace account to download the model and set HF_TOKEN environment variable.

---

## 2. Building the Environment

The example is designed to run in a containerized environment using Triton Distributed, TensorRT-LLM, and associated dependencies. To build the container:

```bash
./container/build.sh --framework TENSORRTLLM
```

---

## 3. Starting the Deployment

Below is a minimal example of how to start each component of a disaggregated serving setup. The typical sequence is:

1. **Download and build model directories**
<!-- 2. **Start the Context Worker(s) and Request Plane**
3. **Start the Generate Worker(s)**
1. **Start the API Server** (handles incoming requests and coordinates workers) -->

All components must be able to connect to the same request plane to coordinate.

### 3.1 HuggingFace Token

```bash
export HF_TOKEN=<YOUR TOKEN>
```

### 3.2 Launch Interactive Environment

```bash
./container/run.sh --framework TENSORRTLLM -it
```

Note: all subsequent commands will be run in the same container for simplicity

Note: by default this command makes all gpu devices visible. Use flag

```
--gpus
```

to selectively make gpu devices visible.

# 3.3: Build model directories

TODO: swap to neural magic fp8 so no key required.
```
cd /workspace/examples/llm/tensorrtllm/scripts
HF_TOKEN=<> python3 prepare_models.py --tp-size 1 --model llama-3.1-8b-instruct --max-num-tokens 8192
```

After this you should see the following in `/workspace/examples/llm/tensorrtllm`
```

```

python3 -m llm.tensorrtllm.deploy --initialize-request-plane

mv hf_downloads tensorrtllm_checkpoints tensorrtllm_engines tensorrtllm_models /workspace/examples/llm/tensorrtllm/operators/

CUDA_VISIBLE_DEVICES=0 \
python3 -m llm.tensorrtllm.deploy \
  --context-worker-count 1 \
  --worker-name llama \
  --initialize-request-plane \
  --request-plane-uri ${HOSTNAME}:4223 &


CUDA_VISIBLE_DEVICES=1 python3 -m llm.tensorrtllm.deploy   --generate-worker-count 1   --worker-name generate  --request-plane-uri ${HOSTNAME}:4223 &

HF_TOKEN= python3 -m llm.api_server --tokenizer meta-llama/Llama-3.1-8B --request-plane-uri ${HOSTNAME}:4223 --api-server-host ${HOSTNAME} --model-name llama


## X. References

[^1]: Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao
Zhang. Distserve: Disaggregating prefill and decoding for goodput-optimized large language
model serving. *arXiv:2401.09670v3 [cs.DC]*, 2024.