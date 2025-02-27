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
from enum import Enum


from compoundai import async_onstart, nova_endpoint, service, tdist_context

WorkerId = str

import bentoml
with bentoml.importing():
    from triton_distributed_rs import KvRouter

from common.protocol import Tokens


class RoutingStrategy(Enum):
    PREFIX = "prefix"
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"


@service(
    nova={
        "enabled": True,
        "namespace": "triton-init",
    },
)
class Router:
    """
    Request handler for the generate endpoint
    """
    def __init__(self):
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.routing_strategy = RoutingStrategy.PREFIX
        self.runtime = tdist_context["runtime"]
        self.min_workers = 1
        self.router = None

    @async_onstart
    async def init_engine(self):
        print("ROUTER init_engine")
        workers_client = (
            await self.runtime.namespace("triton-init")
            .component("VllmEngine")
            .endpoint("generate")
            .client()
        )
        wait_task = workers_client.wait_for_endpoints()
        await asyncio.sleep(1)

        while not wait_task.done():
            print("Waiting for workers to be ready...")
            await asyncio.sleep(5)

        print("ROUTER init_engine wait_task.result()")
        wait_task.result()

        print(
            "ROUTER init_engine while len(workers_client.endpoint_ids()) < self.min_workers"
        )
        while len(workers_client.endpoint_ids()) < self.min_workers:
            print(
                f"Waiting for more workers... Current: {len(workers_client.endpoint_ids())}, Required: {self.min_workers}"
            )
            await asyncio.sleep(5)

        kv_listener = self.runtime.namespace("router").component(self.model_name)
        await kv_listener.create_service()
        self.router = KvRouter(self.runtime, kv_listener)

    @nova_endpoint()
    async def generate(self, request: Tokens):
        lora_id = 0
        worker_id = ""
        if self.routing_strategy == RoutingStrategy.PREFIX:
            try:
                worker_id = await self.router.schedule(request.tokens, lora_id)
                print(f"ROUTER.generate WORKER ID: {worker_id}")
            except Exception as e:
                print(f"ROUTER error: {e}")
                if "No worker found" in str(e):
                    worker_id = ""
                else:
                    print(f"Error during worker selection: {e}")
            print(f"Scheduling to worker_id: {worker_id}")
            yield worker_id
        else:
            # TODO: Do we implement round_robin and random here?
            # or just skip this router and directly enable in preprocess?
            raise NotImplementedError(
                f"Routing strategy {self.routing_strategy} not implemented"
            )
