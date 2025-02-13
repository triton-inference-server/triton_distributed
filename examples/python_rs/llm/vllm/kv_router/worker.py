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
import os
import uuid

import uvloop
import vllm
from common.parser import parse_vllm_args
from common.protocol import Request, Response, TokenizeRequest, TokenizeResponse
from triton_distributed_rs import DistributedRuntime, triton_endpoint, triton_worker, KvRouter
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import logger as vllm_logger

import uuid


class VllmEngine:
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine_args: AsyncEngineArgs):
        vllm_logger.info("Worker init")
        self.engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)

    @triton_endpoint(Request, Response)
    async def generate(self, request):
        # TODO: take in tokenized input
        # tokens_prompt = TokensPrompt(prompt_token_ids=tokens)

        vllm_logger.info(f"Received request: {request}")
        sampling_params = vllm.SamplingParams(**request.sampling_params)
        request_id = str(uuid.uuid4())
        async for response in self.engine.generate(
            request.prompt, sampling_params, request_id
        ):
            vllm_logger.info(f"Generated response: {response}")
            yield response.outputs[0].text

    @triton_endpoint(TokenizeRequest, TokenizeResponse)
    async def tokenize(self, request):
        vllm_logger.info(f"Received request: {request}")
        tokenizer = await self.engine.get_tokenizer()

        # tokens = tokenizer.apply_chat_template(request.prompt, tokenize=True)
        tokens = tokenizer.encode(request.prompt)

        vllm_logger.info(f"Tokens: {tokens}")
        yield tokens


@triton_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("triton-init").component("vllm")
    await component.create_service()

    vllm_engine = VllmEngine(engine_args)

    vllm_logger.info(f"Event subject: {component.event_subject('kv_events')}")

    generate_endpoint = component.endpoint("generate")
    tokenize_endpoint = component.endpoint("tokenize")

    generate_id = generate_endpoint.lease_id()
    vllm_logger.info(f"Generate endpoint ID: {generate_id}")

    random_uuid = str(uuid.uuid4())
    # TODO: Fix this. Added a random UUID as KVPublisher only takes in UUIDs as the worker ID
    vllm_logger.info(f"Random UUID: {random_uuid}")
    os.environ["VLLM_WORKER_ID"] = str(random_uuid)
    vllm_logger.info(f"VLLM_WORKER_ID: {os.environ['VLLM_WORKER_ID']}")


    await asyncio.gather(
        generate_endpoint.serve_endpoint(vllm_engine.generate),
        tokenize_endpoint.serve_endpoint(vllm_engine.tokenize)
    )


if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()
    engine_args.dtype = 'float16'
    asyncio.run(worker(engine_args))
