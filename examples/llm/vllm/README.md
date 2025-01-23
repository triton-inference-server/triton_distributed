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

# vLLM Disaggregated Serving Example

This example demonstrates **disaggregated serving** [^1] using Triton Distributed together with vLLM engines. Disaggregated serving decouples the prefill (prompt encoding) and the decode (token generation) stages of large language model (LLM) inference into separate processes. This separation allows you to independently scale, optimize, and distribute resources for each stage.

In this example, you will deploy:

- An **OpenAI-compatible API server** (which receives requests and streams responses).
- One or more **prefill workers** (for encoding the prompt).
- One or more **decode workers** (for generating tokens based on the encoded prompt).

![Overview of disaggregated serving deployment architecture](assets/vllm_disagg_architecture_overview.jpg)

For more details on the basics of Triton Distributed, please see the [Hello World example](../../hello_world/).

---

## 1. Prerequisites

1. **GPU Availability**
   This setup requires at least two GPUs:
   - One GPU is typically used by the **prefill** process.
   - Another GPU is used by the **decode** process.
   In production systems with heavier loads, you will typically allocate more GPUs across multiple prefill and decode workers.

2. **NATS or Another Coordination Service**
   Triton Distributed uses NATS by default for coordination and message passing. Make sure your environment has a running NATS service accessible via a valid `nats://<address>:<port>` endpoint. By default, examples assume `nats://localhost:4223`.

3. **vLLM Patch**
   This example requires some features that are not yet in the main vLLM release. A patch is automatically applied inside the provided container. Details of the patch can be found [here](../../../container/deps/vllm/). The current patch is compatible with **vLLM 0.6.3post1**.

4. **Supported GPUs**
   - For FP8 usage, GPUs with **Compute Capability >= 8.9** are required.
   - If you have older GPUs, consider BF16/FP16 precision variants instead of `FP8`. (See [below](#model-precision-variants).)

---

## 2. Building the Environment

The example is designed to run in a containerized environment using Triton Distributed, vLLM, and associated dependencies. To build the container:

```bash
./container/build.sh --framework VLLM
```

This command pulls necessary dependencies and patches vLLM in the container image.

---

## 3. Starting the Deployment

Below is a minimal example of how to start each component of a disaggregated serving setup. The typical sequence is:

1. **Start the API Server** (handles incoming requests and coordinates workers)
2. **Start the Prefill Worker(s)**
3. **Start the Decode Worker(s)**

All components must be able to connect to the same NATS server to coordinate.

### 3.1 API Server

The API server in a vLLM-disaggregated setup listens for OpenAI-compatible requests on a chosen port (default 8005). Below is an example command:

```bash
python3 -m examples.api_server \
  --nats-url nats://localhost:4223 \
  --log-level INFO \
  --port 8005
```

### 3.2 Prefill Worker

The prefill stage encodes incoming prompts. By default, vLLM uses GPU resources to tokenize and prepare the model’s key-value (KV) caches. Run the prefill worker:

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_WORKER_ID=0 \
python3 -m examples.vllm.deploy \
  --context-worker-count 1 \
  --nats-url nats://localhost:4223 \
  --model-name neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
  --kv-cache-dtype fp8 \
  --dtype auto \
  --log-level INFO \
  --worker-name llama \
  --disable-async-output-proc \
  --disable-log-stats \
  --max-model-len 32768 \
  --max-batch-size 10000 \
  --gpu-memory-utilization 0.9 \
  --context-tp-size 1 \
  --generate-tp-size 1
```

**Key flags**:
- `--context-worker-count`: Launches only context (prefill) workers.
- `--kv-cache-dtype fp8`: Using FP8 for caching (requires CC >= 8.9).
- `CUDA_VISIBLE_DEVICES=0`: Binds worker to GPU `0`.

### 3.3 Decode Worker

The decode stage consumes the KV cache produced in the prefill step and generates output tokens. Run the decode worker:

```bash
CUDA_VISIBLE_DEVICES=1 \
VLLM_WORKER_ID=1 \
python3 -m examples.vllm.deploy \
  --generate-worker-count 1 \
  --nats-url nats://localhost:4223 \
  --model-name neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
  --kv-cache-dtype fp8 \
  --dtype auto \
  --log-level INFO \
  --worker-name llama \
  --disable-async-output-proc \
  --disable-log-stats \
  --max-model-len 32768 \
  --max-batch-size 10000 \
  --gpu-memory-utilization 0.9 \
  --context-tp-size 1 \
  --generate-tp-size 1
```

**Key flags**:
- `--generate-worker-count`: Launches decode worker(s).
- `CUDA_VISIBLE_DEVICES=1`: Binds worker to GPU `1`.

> [!NOTE]
> - You can run multiple prefill and decode workers for higher throughput.
> - For large models, ensure you have enough GPU memory (or GPUs).


## 4. Sending Requests

Once the API server is running (by default on `localhost:8005`), you can send OpenAI-compatible requests. For example:

```bash
curl localhost:8005/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [
      {"role": "system", "content": "What is the capital of France?"}
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


## 5. Benchmarking

You can benchmark this setup using [**GenAI-Perf**](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/README.md), which supports OpenAI endpoints for chat or completion requests.

```bash
genai-perf profile \
  -m llama \
  --url <API_SERVER_HOST>:8005 \
  --endpoint-type chat \
  --streaming \
  --num-dataset-entries 1000 \
  --service-kind openai \
  --endpoint v1/chat/completions \
  --warmup-request-count 10 \
  --random-seed 123 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-stddev 0 \
  --tokenizer neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
  --synthetic-input-tokens-mean 3000 \
  --output-tokens-mean 150 \
  --extra-inputs seed:100 \
  --extra-inputs min_tokens:150 \
  --extra-inputs max_tokens:150 \
  --profile-export-file my_profile_export.json \
  --artifact-dir artifacts/ \
  --concurrency 32 \
  --request-count 320 \
  -- -v \
  --async
```

**Key Parameters**:
- **`-m llama`**: Your model name (must match the name used in your server).
- **`--url <API_SERVER_HOST>:8005`**: The location of your API server.
- **`--endpoint v1/chat/completions`**: Using the OpenAI chat endpoint.
- **`--streaming`**: Ensures tokens are streamed back for chat-like usage.


## 6. Model Precision Variants

In the commands above, we used the FP8 variant `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8` because it significantly reduces KV cache size, which helps with network transfer and memory usage. However, if your GPU is older or does not support FP8, try using the standard BF16/FP16 precision variant, for example:

```bash
--model-name meta-llama/Meta-Llama-3.1-8B-Instruct
--kv-cache-dtype bf16
```


## 7. Known Issues & Limitations

1. **Fixed Worker Count**
   Currently, the number of prefill and decode workers must be fixed at the start of deployment. Dynamically adding or removing workers is not yet supported.

2. **KV Transfer OOM**
   During heavy loads, KV cache transfers between prefill and decode processes may cause out-of-memory errors if there is insufficient GPU memory.

3. **KV Cache Preemption**
   Cache preemption (evicting old prompts to free memory) is not supported in the current patch.

4. **Experimental Patch**
   The required vLLM patch is experimental and not yet merged into upstream vLLM. Future releases may remove the need for a custom patch.


## 8. References

[^1]: Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao
Zhang. Distserve: Disaggregating prefill and decoding for goodput-optimized large language
model serving. *arXiv:2401.09670v3 [cs.DC]*, 2024.

For more details on Triton Distributed and additional examples, please consult the official [Hello World example](../../hello_world/) and the [Triton Inference Server documentation](https://github.com/triton-inference-server/server).
