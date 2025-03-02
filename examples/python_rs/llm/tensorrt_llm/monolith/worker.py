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
import json
from enum import Enum
from typing import List

import uvloop
from common.base_engine import BaseTensorrtLLMEngine
from common.parser import LLMAPIConfig, parse_tensorrt_llm_args
from common.processor import merge_promises
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import (
    CompletionRequest,
    CompletionStreamResponse,
)

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

    def __init__(self, engine_config: LLMAPIConfig):
        super().__init__(engine_config)

    @triton_endpoint(CompletionRequest, CompletionStreamResponse)
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
        response_generator = self.completions_processor.create_completion_generator(
            request, generator, num_choices
        )
        async for response in response_generator:
            yield json.loads(response)

        # TODO: non stream
        self._ongoing_request_count -= 1


@triton_worker()
async def worker(runtime: DistributedRuntime, engine_config: LLMAPIConfig):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("triton-init").component("tensorrt-llm")
    await component.create_service()

    completions_endpoint = component.endpoint("completions")

    engine = TensorrtLLMEngine(engine_config)

    await completions_endpoint.serve_endpoint(engine.generate_completion)


if __name__ == "__main__":
    uvloop.install()
    args, engine_config = parse_tensorrt_llm_args()
    asyncio.run(worker(engine_config))
