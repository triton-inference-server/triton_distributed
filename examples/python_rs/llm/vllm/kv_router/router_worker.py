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

import uvloop
import vllm
from common.parser import parse_vllm_args
from common.protocol import TokenizeResponse, Response
from triton_distributed_rs import DistributedRuntime, triton_endpoint, triton_worker, KvRouter
# from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import logger as vllm_logger


class Router:
    """
    Request handler for the generate endpoint
    """

    def __init__(self, router):
        vllm_logger.info("Router init")
        # self.workers = workers
        self.router = router

    @triton_endpoint(TokenizeResponse, Response)
    async def generate(self, start_ids):
        vllm_logger.info(f"Received start_ids: {start_ids}")

        lora_id = 0
        try:
            worker_subject = await self.router.schedule(start_ids.tokens, lora_id)
        except Exception as e:
            vllm_logger.info(f"{e}")
            if "No worker found" in str(e):
                worker_subject = ""
            else:
                vllm_logger.exception(f"Error during worker selection: {e}")

        vllm_logger.info(f"Scheduling to worker subject: {worker_subject}")
        yield worker_subject


# @triton_worker()
# async def worker(runtime: DistributedRuntime):
#     vllm_logger.info("Starting router worker")
    
#     # Create router component
#     router_component = runtime.namespace("triton-init").component("router")
#     await router_component.create_service()
#     vllm_logger.info("Created router service")

#     # Get VLLM component and create endpoint
#     vllm_component = runtime.namespace("triton-init").component("vllm")
#     generate_endpoint = vllm_component.endpoint("generate")
    
#     # Get client and wait for it to be ready
#     generate_client = await generate_endpoint.client()
#     try:
#         await asyncio.wait_for(generate_client.wait_for_endpoints(), timeout=30.0)
#         vllm_logger.info("VLLM endpoints are available")
#     except asyncio.TimeoutError:
#         vllm_logger.error("Timeout waiting for VLLM endpoints")
#         raise Exception("No VLLM workers available after 30 seconds")

#     # Create router
#     router = KvRouter(runtime, vllm_component)
#     vllm_logger.info("Created KvRouter")

    
#     # # List available endpoints
#     # endpoints = await vllm_component.list_endpoints()
#     # vllm_logger.info(f"Available endpoints: {endpoints}")
    
#     # # Create router
#     # router = KvRouter(runtime, vllm_component)
#     # vllm_logger.info("Created KvRouter")
    
#     endpoint = router_component.endpoint("generate")
#     vllm_logger.info("Created router endpoint")
    
#     await endpoint.serve_endpoint(Router(router).generate)
#     vllm_logger.info("Router endpoint is now serving")


@triton_worker()
async def worker(runtime: DistributedRuntime):
    # create endpoint service for frontend component
    vllm_logger.info(f"=========== Hi ===========")

    router_component = runtime.namespace("triton-init").component("router")
    await router_component.create_service()

    router = KvRouter(runtime, router_component)

    endpoint = router_component.endpoint("generate")
    await endpoint.serve_endpoint(Router(router).generate)
    # vllm_logger.info(f"router_id {router_id}")


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
