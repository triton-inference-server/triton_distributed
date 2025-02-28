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
from typing import Any, Dict, List, Tuple

import uvloop
from common.base_engine import BaseTensorrtLLMEngine
from common.parser import parse_tensorrt_llm_args
from common.processor import merge_promises
from common.protocol import nvCompletionRequest
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import CompletionStreamResponse

from triton_distributed.runtime import (
    DistributedRuntime,
    triton_endpoint,
    triton_worker,
)

logger.set_level("debug")


class RequestType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"


class TensorrtLLMEngine(BaseTensorrtLLMEngine):
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine_args: Tuple[Dict[str, Any], Dict[str, Any]]):
        super().__init__(engine_args)

    @triton_endpoint(nvCompletionRequest, CompletionStreamResponse)
    async def generate_completion(self, request):
        if self._llm_engine is None:
            raise RuntimeError("Engine not initialized")

        self._ongoing_request_count += 1
        logger.debug(f"Received completion request: {request}")

        if isinstance(request.prompt, str) or (
            isinstance(request.prompt, list) and isinstance(request.prompt[0], int)
        ):
            prompts = [request.prompt]
        else:
            prompts = request.prompt

        promises: List[RequestOutput] = []
        sampling_params = request.to_sampling_params()
        disaggregated_params = request.to_llm_disaggregated_params()

        for prompt in prompts:
            promise = self._llm_engine.generate_async(
                prompt,
                sampling_params,
                streaming=request.stream,
                disaggregated_params=disaggregated_params,
            )
            promises.append(promise)

        generator = merge_promises(promises)
        num_choices = len(prompts) if request.n is None else len(prompts) * request.n
        if request.stream:
            response_generator = self.completions_processor.create_completion_generator(
                request, generator, num_choices
            )
            async for response in response_generator:
                yield response
        else:
            # TODO: why doesn't it read stream from input?
            # stream is always True

            raise NotImplementedError("Non-streaming is not supported")
            # response = await self.completions_processor.create_completion_response(
            #     request, generator, num_choices
            # )
            # yield response


@triton_worker()
async def worker(
    runtime: DistributedRuntime, engine_args: Tuple[Dict[str, Any], Dict[str, Any]]
):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("triton-init").component("tensorrt-llm")
    await component.create_service()

    completions_endpoint = component.endpoint("completions")

    engine = TensorrtLLMEngine(engine_args)

    await asyncio.gather(
        completions_endpoint.serve_endpoint(engine.generate_completion),
    )


if __name__ == "__main__":
    uvloop.install()
    _, engine_args = parse_tensorrt_llm_args()
    asyncio.run(worker(engine_args))
