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
import msgspec
import uuid
import uvloop
from vllm.sampling_params import SamplingParams

from triton_distributed_rs import DistributedRuntime, triton_worker

from protocol import Request


@triton_worker()
async def worker(
    runtime: DistributedRuntime,
    prompt: str,
    remote_prefill: bool,
):
    """
    Instantiate a `backend` client and call the `generate` endpoint
    """
    # get endpoint
    endpoint = (
        runtime.namespace("test-nixl").component("vllm").endpoint("generate")
    )

    # create client
    client = await endpoint.client()

    # wait for an endpoint to be ready
    await client.wait_for_endpoints()

    request = Request(
        request_id=str(uuid.uuid4()),
        prompt=prompt,
        sampling_params=SamplingParams(
            temperature=0.0,
            max_tokens=10,
        ),
        do_remote_prefill=remote_prefill,
    )

    json_str = msgspec.json.encode(request).decode('utf-8')
    print(f"Sending request: {json_str}", type(json_str))
    async for resp in await client.generate(json_str):
        print(resp)


if __name__ == "__main__":
    uvloop.install()

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="what is the capital of france?")
    parser.add_argument("--remote-prefill", action="store_true")
    args = parser.parse_args()

    asyncio.run(worker(args.prompt, args.remote_prefill))
