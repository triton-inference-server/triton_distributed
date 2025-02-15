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


import asyncio
import time

import uvloop
from tqdm.asyncio import tqdm
from triton_distributed_rs import DistributedRuntime, triton_worker
from vllm.utils import FlexibleArgumentParser

from .protocol import Request


async def do_one(client, prompt, max_tokens, temperature):
    stream = await client.generate(
        Request(
            prompt=prompt,
            sampling_params={
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        ).model_dump_json()
    )

    # process response
    async for resp in stream:
        pass  # print(resp)


@triton_worker()
async def worker(
    runtime: DistributedRuntime,
    prompt: str,
    max_tokens: int,
    temperature: float,
    request_count: int,
):
    """
    Instantiate a `backend` client and call the `generate` endpoint
    """
    start_time = time.time()

    # get endpoint
    endpoint = runtime.namespace("triton-init").component("vllm").endpoint("generate")

    # create client
    client = await endpoint.client()

    # list the endpoints
    print(client.endpoint_ids())

    tasks = []
    for i in range(request_count):
        tasks.append(
            asyncio.create_task(do_one(client, prompt, max_tokens, temperature))
        )

    await tqdm.gather(*tasks)

    # ensure all tasks are done and without errors
    error_count = 0
    for task in tasks:
        if task.exception():
            error_count += 1

    assert error_count == 0, f"expected 0 errors, got {error_count}"

    print(f"time: {time.time()-start_time}")


if __name__ == "__main__":
    uvloop.install()

    parser = FlexibleArgumentParser()
    parser.add_argument("--prompt", type=str, default="what is the capital of france?")
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--request-count", type=int, default=10)

    args = parser.parse_args()

    asyncio.run(
        worker(args.prompt, args.max_tokens, args.temperature, args.request_count)
    )
