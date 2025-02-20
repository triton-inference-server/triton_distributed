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
from typing import AsyncIterator

import uvloop
from common.parser import parse_vllm_args
from common.protocol import Tokens, MyRequestOutput, vLLMGenerateRequest
from common.base_engine import BaseVllmEngine

from kv_router.router import WorkerId

from triton_distributed_rs import (
    DistributedRuntime,
    triton_endpoint,
    triton_worker,
)
from triton_distributed_rs._core import Client

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import logger as vllm_logger
from vllm.outputs import RequestOutput


from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
)


class Processor(BaseVllmEngine):
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine_args: AsyncEngineArgs, router_client: Client, workers_client: Client):
        super().__init__(engine_args)
        self.router_client = router_client
        self.workers_client = workers_client

    async def generate_responses(self, engine_generator) -> AsyncIterator[RequestOutput]:
        async for resp in engine_generator:
            output = MyRequestOutput.model_validate_json(resp.data())
            
            yield RequestOutput(
                request_id=output.request_id,
                prompt=output.prompt,
                prompt_token_ids=output.prompt_token_ids,
                prompt_logprobs=output.prompt_logprobs,
                outputs=output.outputs,
                finished=output.finished,
                metrics=output.metrics,
            )
    
    @triton_endpoint(ChatCompletionRequest, ChatCompletionStreamResponse)
    async def generate(self, raw_request):
        request_id = str(uuid.uuid4())
        vllm_logger.debug(f"Got raw request: {raw_request}")
        (
            request,
            conversation,
            prompt,
            engine_prompt,
            sampling_params,
        ) = await self._parse_raw_request(raw_request)
        worker_id_generator: AsyncIterator[WorkerId] = await self.router_client.generate(
            Tokens(tokens=engine_prompt["prompt_token_ids"]).model_dump_json()
        )

        worker_id = await worker_id_generator.__anext__()
        worker_id = worker_id.data()
        vllm_logger.info(f"Worker ID: {worker_id}")

        if worker_id == "":
            engine_generator: AsyncIterator[MyRequestOutput] = await self.workers_client.random(
                vLLMGenerateRequest(
                    engine_prompt=engine_prompt,
                    sampling_params=sampling_params,
                    request_id=request_id
                ).model_dump_json()
            )
        else:
            engine_generator: AsyncIterator[MyRequestOutput] = await self.workers_client.direct(
                vLLMGenerateRequest(
                    engine_prompt=engine_prompt,
                    sampling_params=sampling_params,
                    request_id=request_id
                ).model_dump_json(),
                uuid.UUID(worker_id).int
            )

        output = self.generate_responses(engine_generator)

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
    workers_client = (
        await runtime.namespace("triton-init")
        .component("vllm")
        .endpoint("generate")
        .client()
    )

    router_client = (
        await runtime.namespace("triton-init")
        .component("router")
        .endpoint("generate")
        .client()
    )

    preprocess_component = runtime.namespace("triton-init").component("preprocess")
    await preprocess_component.create_service()
    preprocess_endpoint = preprocess_component.endpoint("generate")

    processor = Processor(engine_args, router_client, workers_client)
    await preprocess_endpoint.serve_endpoint(processor.generate)


if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()
    asyncio.run(worker(engine_args))
