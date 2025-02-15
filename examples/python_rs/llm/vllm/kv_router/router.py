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
import uuid
from argparse import Namespace
from enum import Enum

import uvloop
from common.protocol import Response, TokenizedRequest
from triton_distributed_rs import (
    DistributedRuntime,
    KvRouter,
    triton_endpoint,
    triton_worker,
)
from vllm.logger import logger as vllm_logger


class RoutingStrategy(Enum):
    PREFIX = "prefix"
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"


class Router:
    """
    Request handler for the generate endpoint
    """

    def __init__(
        self,
        router,
        workers_client,
        routing_strategy: RoutingStrategy = RoutingStrategy.PREFIX,
    ):
        vllm_logger.info(
            f"Initializing KV Router with strategy: {routing_strategy.value}"
        )
        self.router = router
        self.workers_client = workers_client
        self.routing_strategy = routing_strategy

    @triton_endpoint(TokenizedRequest, Response)
    async def generate(self, request):
        lora_id = 0
        worker_id = ""
        if self.routing_strategy == RoutingStrategy.PREFIX:
            try:
                worker_id = await self.router.schedule(request.tokens, lora_id)
            except Exception as e:
                vllm_logger.info(f"{e}")
                if "No worker found" in str(e):
                    worker_id = ""
                else:
                    vllm_logger.exception(f"Error during worker selection: {e}")

            vllm_logger.info(f"Scheduling to worker_id: {worker_id}")

        if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            engine_generator = await self.workers_client.round_robin(
                request.model_dump_json()
            )
        elif self.routing_strategy == RoutingStrategy.RANDOM or worker_id == "":
            engine_generator = await self.workers_client.random(
                request.model_dump_json()
            )
        else:
            # extract back lease_id
            engine_generator = await self.workers_client.direct(
                request.model_dump_json(), uuid.UUID(worker_id).int
            )

        async for resp in engine_generator:
            resp = resp.data() if hasattr(resp, "data") else resp
            yield resp


@triton_worker()
async def worker(runtime: DistributedRuntime, args: Namespace):
    workers_client = (
        await runtime.namespace("triton-init")
        .component("vllm")
        .endpoint("generate")
        .client()
    )
    vllm_logger.info("Waiting for workers to be ready")
    await workers_client.wait_for_endpoints()
    vllm_logger.info("A worker is ready")

    # TODO Router is a fixed namespace separate from the others
    kv_listener = runtime.namespace("router").component(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    )
    await kv_listener.create_service()

    router_component = runtime.namespace("triton-init").component("router")
    await router_component.create_service()

    router = None
    if args.routing_strategy == RoutingStrategy.PREFIX:
        router = KvRouter(runtime, kv_listener)

    endpoint = router_component.endpoint("generate")
    await endpoint.serve_endpoint(
        Router(router, workers_client, args.routing_strategy).generate
    )


if __name__ == "__main__":
    uvloop.install()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--routing-strategy",
        type=RoutingStrategy,
        default=RoutingStrategy.PREFIX,
        choices=list(RoutingStrategy),
        help="Routing strategy to use",
    )
    args = parser.parse_args()

    asyncio.run(worker(args))
