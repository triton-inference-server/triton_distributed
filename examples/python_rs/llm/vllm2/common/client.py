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

from tqdm import tqdm

# import uvloop
from vllm.utils import FlexibleArgumentParser

from triton_distributed.icp import NatsRequestPlane, UcpDataPlane

# from triton_distributed import DistributedRuntime, triton_worker
from triton_distributed.runtime import RemoteOperator as RemoteFunction

from .protocol import Request, Response


async def main(prompt: str, max_tokens: int, temperature: float):
    """
    Instantiate a `backend` client and call the `generate` endpoint
    """
    request_plane = NatsRequestPlane()
    await request_plane.connect()

    data_plane = UcpDataPlane()
    data_plane.connect()

    client = RemoteFunction("vllm_generate", request_plane, data_plane)

    request_count = 50

    with tqdm(total=request_count, desc="Sending Requests", unit="request") as pbar:
        for index in range(request_count):
            # issue request
            stream = client.call(
                Request(
                    prompt=prompt,
                    sampling_params={
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                ),
                return_type=Response,
            )

            # process response
            async for resp in stream:
                print(resp)

            pbar.update(1)


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--prompt", type=str, default="what is the capital of france?")
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.5)

    args = parser.parse_args()

    asyncio.run(main(args.prompt, args.max_tokens, args.temperature))
