# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import asyncio

import random
import uvloop
from triton_distributed_rs import DistributedRuntime, triton_worker

from .protocol import Request, GpuUsageRequest
import json


@triton_worker()
async def worker(
    runtime: DistributedRuntime, prompts: list[str], max_tokens: int, temperature: float
):
    """
    Instantiate a `backend` client and call the `generate` endpoint with load balancing
    """
    # get endpoint
    generate_endpoint = runtime.namespace("triton-init").component("vllm").endpoint("generate")
    gpu_stats_endpoint = runtime.namespace("triton-init").component("vllm").endpoint("gpu_stats")
    
    # create client
    generate_client = await generate_endpoint.client()
    gpu_usage_client = await gpu_stats_endpoint.client()
    
    endpoint_ids = generate_client.endpoint_ids()

    async def process_request(prompt: str):
        await asyncio.sleep(10 * random.random())
        
        # Get GPU stats for all endpoints
        gpu_stats = {}
        for endpoint_id in endpoint_ids:
            response = await gpu_usage_client.direct(GpuUsageRequest(prompt='').model_dump_json(), endpoint_id)
            async for chunk in response:
                pass
            gpu_stats[endpoint_id] = chunk.data()
        
        # Find endpoint with minimum GPU usage
        endpoint_id = min(gpu_stats.items(), key=lambda x: float(x[1]))[0]
        print(gpu_stats)
        
        try:
            request = Request(
                prompt=prompt,
                sampling_params={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "ignore_eos": True, 
                },
            ).model_dump_json()
            
            async for resp in await generate_client.direct(request, endpoint_id):
                pass
        finally:
            pass

    # Create tasks for all prompts
    tasks = [process_request(prompt) for prompt in prompts]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    uvloop.install()

    # Use default prompt if none provided
    prompts = ["Write me a short story about a capybara playing the violin."]
    prompts = prompts * 100
    
    asyncio.run(worker(prompts, 1024, 0.5))