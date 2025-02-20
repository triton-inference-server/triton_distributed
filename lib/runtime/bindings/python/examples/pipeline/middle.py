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
from triton_distributed_rs import DistributedRuntime, triton_worker

uvloop.install()


class RequestHandler:
    def __init__(self, backend):
        self.backend = backend

    async def generate(self, request):
        request = f"{request}-mid"
        async for output in await self.backend.random(request):
            yield output.get("data")


@triton_worker()
async def worker(runtime: DistributedRuntime):
    # client to backend
    backend = (
        await runtime.namespace("examples/pipeline")
        .component("backend")
        .endpoint("generate")
        .client()
    )

    # create endpoint service for middle component
    component = runtime.namespace("examples/pipeline").component("middle")
    await component.create_service()

    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(RequestHandler(backend).generate)


asyncio.run(worker())
