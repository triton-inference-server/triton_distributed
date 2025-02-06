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

1. Follow the setup instructions in the Python bindings [README](/runtime/rust/python-wheel/README.md) to prepare your environment
2. Install vLLM:
```bash
uv pip install vllm==0.7.0
```

## Deployment Options

### 1. Monolithic Deployment

Run the server and client components in separate terminal sessions:

**Terminal 1 - Server:**
```bash
python3 -m monolith.worker \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --max-model-len 100 \
    --enforce-eager
```

**Terminal 2 - Client:**
```bash
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
CUDA_VISIBLE_DEVICES=0 python3 -m disaggregated.prefill_worker \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'
```

**Terminal 2 - Decode Worker:**
```bash
CUDA_VISIBLE_DEVICES=1 python3 -m disaggregated.decode_worker \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}'
```

**Terminal 3 - Client:**
```bash
python3 -m common.client \
    --prompt "what is the capital of france?" \
    --max-tokens 10 \
    --temperature 0.5
```

The disaggregated deployment utilizes separate GPUs for prefill and decode operations, allowing for optimized resource allocation and improved performance. For more details on the disaggregated deployment, please refer to the [vLLM documentation](https://docs.vllm.ai/en/latest/features/disagg_prefill.html).
