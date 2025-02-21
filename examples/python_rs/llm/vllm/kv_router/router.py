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
        worker_id = None
        if self.routing_strategy == RoutingStrategy.PREFIX:
            try:
                worker_id = await self.router.schedule(request.tokens, lora_id)
            # [NOTE][TODO] Now that the scheduler may return more error messages,
            # now we are catching all exceptions and logging them. Should have
            # catch specific router exceptions once we have dedicated types.
            except Exception as e:
                vllm_logger.info(f"{e}")
                worker_id = None
                vllm_logger.exception(f"Error during worker selection: {e}")

            vllm_logger.info(f"Scheduling to worker_id: {worker_id}")

        if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            engine_generator = await self.workers_client.round_robin(
                request.model_dump_json()
            )
        elif self.routing_strategy == RoutingStrategy.RANDOM or worker_id is None:
            engine_generator = await self.workers_client.random(
                request.model_dump_json()
            )
        else:
            # extract back lease_id
            engine_generator = await self.workers_client.direct(
                request.model_dump_json(), worker_id
            )

        async for resp in engine_generator:
            resp = resp.data() if hasattr(resp, "data") else resp
            yield resp


@triton_worker()
async def worker(runtime: DistributedRuntime, args: Namespace):
    workers_client = (
        await runtime.namespace("triton-init")
        .component("vllm")
        .endpoint("generate_from_tokens")
        .client()
    )
    vllm_logger.info("Waiting for workers to be ready")
    await workers_client.wait_for_endpoints()

    while len(workers_client.endpoint_ids()) < args.min_workers:
        vllm_logger.info(
            f"Waiting for more workers... Current: {len(workers_client.endpoint_ids())}, Required: {args.min_workers}"
        )
        await asyncio.sleep(5)

    vllm_logger.info(
        f"Required number of workers ({args.min_workers}) are ready:\n"
        + "\n".join(f"id: {id}" for id in workers_client.endpoint_ids())
    )

    kv_listener = runtime.namespace("triton-init").component("vllm")
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
    parser.add_argument(
        "--min-workers",
        type=int,
        default=1,
        help="Minimum number of workers required before proceeding",
    )
    args = parser.parse_args()

    asyncio.run(worker(args))
