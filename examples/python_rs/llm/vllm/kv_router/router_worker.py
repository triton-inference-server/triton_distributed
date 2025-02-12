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

    def __init__(self, workers, router):
        vllm_logger.info("Router init")
        self.workers = workers
        self.router = router

    @triton_endpoint(TokenizeResponse, Response)
    async def generate(self, start_ids):
        vllm_logger.info(f"Received start_ids: {start_ids}")

        lora_id = 0
        worker_subject = await self.router.schedule((start_ids, lora_id))
        vllm_logger.info(f"Scheduling to worker subject: {worker_subject}")
        yield worker_subject



@triton_worker()
async def worker(runtime: DistributedRuntime, workers):
    # create endpoint service for frontend component
    router_component = runtime.namespace("triton-init").component("router")
    await router_component.create_service()

    endpoint = router_component.endpoint("generate")

    await endpoint.serve_endpoint(Router(workers, KvRouter(runtime, router_component)).generate)

if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()
    asyncio.run(worker(engine_args))
