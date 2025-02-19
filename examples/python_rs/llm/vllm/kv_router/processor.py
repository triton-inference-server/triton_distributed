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
import re
import uuid
from typing import Optional, AsyncIterator

import uvloop
import vllm
from common.parser import parse_vllm_args
from common.protocol import Request, Response, TokenizedRequest
from common.base_engine import BaseVllmEngine

from triton_distributed_rs import (
    DistributedRuntime,
    KvRouter,
    triton_endpoint,
    triton_worker,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.logger import logger as vllm_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.outputs import RequestOutput




from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
)


class Processor(BaseVllmEngine):
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine_args: AsyncEngineArgs, router: KvRouter):
        super().__init__(engine_args)
        self.router = router

    async def generate_responses(self, engine_generator) -> AsyncIterator[RequestOutput]:
        async for resp in engine_generator:
            # Assuming resp.data() returns a dictionary
            vllm_logger.info(f"resp: {resp.data()}")
            yield RequestOutput(**resp.data())
    
    @triton_endpoint(ChatCompletionRequest, ChatCompletionStreamResponse)
    async def generate(self, raw_request):
        vllm_logger.debug(f"Got raw request: {raw_request}")
        (
            request,
            conversation,
            prompt,
            engine_prompt,
            sampling_params,
        ) = await self._parse_raw_request(raw_request)

        vllm_logger.info(f"Engine prompt: {engine_prompt["prompt_token_ids"]}")

        vllm_logger.info(f"Sampling params: {sampling_params}")
        vllm_logger.info(f"Request: {prompt}")

        # TODO: properly handle sampling params
        engine_generator = await self.router.generate(
            TokenizedRequest(tokens=engine_prompt["prompt_token_ids"], prompt=prompt, sampling_params={"temperature":sampling_params.temperature}).model_dump_json()
        )

        output = self.generate_responses(engine_generator)


        # # connect with worker id
        request_id = str(uuid.uuid4())

        vllm_logger.info(f"output: {output}")

        vllm_logger.info(f"Got engine generator")

        async for response in await self._stream_response(
            request, output, request_id, conversation
        ):
            vllm_logger.info(f"Generated response: {response}")
            yield response


@triton_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """

    router_client = (
        await runtime.namespace("triton-init")
        .component("router")
        .endpoint("generate")
        .client()
    )

    preprocess_component = runtime.namespace("triton-init").component("preprocess")
    await preprocess_component.create_service()
    preprocess_endpoint = preprocess_component.endpoint("generate")

    processor = Processor(engine_args, router_client)
    await preprocess_endpoint.serve_endpoint(processor.generate)


if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()
    asyncio.run(worker(engine_args))
