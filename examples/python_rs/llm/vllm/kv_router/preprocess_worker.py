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
from common.protocol import TokenizeRequest, TokenizeResponse, Request, Response
from triton_distributed_rs import DistributedRuntime, triton_endpoint, triton_worker
# , Client
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import logger as vllm_logger


class VllmPreprocessEngine:
    """
    Request handler for the generate endpoint
    """

    def __init__(self, tokenize_workers, workers, router):
        # assert (
        #     engine_args.kv_transfer_config.is_kv_consumer
        # ), "Decode worker must be a KV consumer"
        vllm_logger.info("Preprocess init")
        self.tokenize_workers = tokenize_workers
        self.workers = workers
        self.router = router

    # @triton_endpoint(Request, Response)
    # async def generate(self, request):
    #     vllm_logger.info(f"Received request: {request}")

    #     start_ids = await self.tokenize_workers.generate(request)

    #     # process response
    #     async for resp in start_ids:
    #         print(resp)

    #     vllm_logger.info(f"Received start ids: {start_ids}")

    #     worker_subject = await self.router.random(start_ids)

    #     # TODO: recognize that worker_subject might be None

    #     async for response in self.workers.direct(request, worker_subject):
    #         vllm_logger.info(f"Generated response: {response}")
    #         # yield response
    #         yield response.outputs[0].text

    @triton_endpoint(Request, Response)
    async def generate(self, request):
        vllm_logger.info(f"Received request: {request}")

        tokenize_generator = await self.tokenize_workers.generate(
            TokenizeRequest(
                prompt=request.prompt,
            ).model_dump_json()
        )
        tokenizer_response = [resp async for resp in tokenize_generator]
        vllm_logger.info(f"Tokenize response: {tokenizer_response}")


        worker_subject = await self.router.generate(
            TokenizeResponse(
                tokens=tokenizer_response
            ).model_dump_json()
        )
        vllm_logger.info(f"Router choice: {worker_subject}")

        # engine_generator = await self.workers.generate(request)
        # engine_generator = await self.workers.generate(
        #     Request(
        #     prompt="hello",
        #     sampling_params={"temperature": 0.5, "max_tokens": 100},
        # ).model_dump_json())

        async for response in tokenize_generator:
            vllm_logger.info(f"Generated response: {response}")
            # yield response
            yield response.outputs[0].text


@triton_worker()
async def worker(runtime: DistributedRuntime):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    vllm_workers = runtime.namespace("triton-init").component("vllm")

    # Get clients and wait for endpoints to be available
    tokenize_client = await vllm_workers.endpoint("tokenize").client()
    workers_client = await vllm_workers.endpoint("generate").client()
    router_client = await runtime.namespace("triton-init").component("router").endpoint("generate").client()

    await asyncio.gather(
        tokenize_client.wait_for_endpoints(),
        workers_client.wait_for_endpoints(),
        router_client.wait_for_endpoints()
    )

    # Process component
    component = runtime.namespace("triton-init").component("preprocess")
    await component.create_service()
    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(VllmPreprocessEngine(tokenize_client, workers_client, router_client).generate)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
