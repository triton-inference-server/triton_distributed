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

import uvloop

from triton_distributed.runtime import DistributedRuntime, triton_worker


class RequestHandler:
    """
    Request handler for the generate function
    """

    async def generate(self, request):
        print(f"Received request: {request}")
        for char in request:
            if char == "w":
                raise ValueError("w is not allowed")
            yield char


@triton_worker()
async def worker(runtime: DistributedRuntime):
    await init(runtime, "triton-init")


async def init(runtime: DistributedRuntime, ns: str):
    """
    Instantiate a `backend` component and serve the `generate` function
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace(ns).component("backend")
    await component.create_service()

    function = component.function("generate")
    print("Started server instance")
    await function.serve_endpoint(RequestHandler().generate)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
