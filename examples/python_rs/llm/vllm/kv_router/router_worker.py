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
import logging

import uvloop
import vllm
from common.parser import parse_vllm_args
from common.protocol import TokenizeResponse, Response
from vllm.logger import logger as vllm_logger

# # need to set before rust import to get logs from rust
vllm_logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)
from triton_distributed_rs import DistributedRuntime, triton_endpoint, triton_worker, KvRouter
from triton_distributed_rs._core import log_test
# from vllm.engine.arg_utils import AsyncEngineArgs


class Router:
    """
    Request handler for the generate endpoint
    """

    def __init__(self, router):
        vllm_logger.info("Router Init")
        self.router = router

    @triton_endpoint(TokenizeResponse, Response)
    async def generate(self, start_ids):

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



@triton_worker()
async def worker(runtime: DistributedRuntime):

    # TODO Router is a fixed namespace seperate from the others
    kv_listener = runtime.namespace("router").component("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    await kv_listener.create_service()

    router = KvRouter(runtime, kv_listener)

    router_component = runtime.namespace("triton-init").component("router")
    await router_component.create_service()

    endpoint = router_component.endpoint("generate")
    await endpoint.serve_endpoint(Router(router).generate)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
