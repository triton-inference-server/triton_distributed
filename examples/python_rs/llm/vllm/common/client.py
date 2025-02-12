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

import uvloop
from triton_distributed_rs import DistributedRuntime, triton_worker

from .protocol import Request


@triton_worker()
async def worker(
    runtime: DistributedRuntime, prompt: str, max_tokens: int, temperature: float
):
    """
    Instantiate a `backend` client and call the `generate` endpoint
    """
    # get endpoint
    endpoint = runtime.namespace("triton-init").component("vllm").endpoint("generate")

    # create client
    client = await endpoint.client()

    # list the endpoints
    print(client.endpoint_ids())

    # issue request
    tasks = []
    for _ in range(1):
        tasks.append(
            client.generate(
                Request(
                    prompt=prompt,
                    sampling_params={
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                ).model_dump_json()
            )
        )
    streams = await asyncio.gather(*tasks)

    # process response
    for stream in streams:
        async for resp in stream:
            print(resp)


if __name__ == "__main__":
    uvloop.install()

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="what is the capital of france?")
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.5)

    args = parser.parse_args()

    asyncio.run(worker(args.prompt, args.max_tokens, args.temperature))
