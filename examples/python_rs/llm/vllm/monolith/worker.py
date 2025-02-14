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
from common.chat_processor import ChatProcessor
from common.parser import parse_vllm_args
from triton_distributed_rs import DistributedRuntime, triton_worker
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import logger as vllm_logger


class VllmEngine:
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine_args: AsyncEngineArgs):
        self.model_config = engine_args.create_model_config()
        self.engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        self.chat_processor = ChatProcessor(self.engine, self.model_config)

    async def generate(self, raw_request):
        vllm_logger.debug(f"Received raw request: {raw_request}")
        request = self.chat_processor.parse_raw_request(raw_request)
        conversation, _, engine_prompt = await self.chat_processor.preprocess(
            raw_request
        )
        default_max_tokens = self.model_config.max_model_len - len(
            engine_prompt["prompt_token_ids"]
        )
        default_sampling_params = self.model_config.get_diff_sampling_param()
        sampling_params = request.to_sampling_params(
            default_max_tokens,
            self.model_config.logits_processor_pattern,
            default_sampling_params,
        )
        request_id = str(uuid.uuid4())

        vllm_logger.debug(
            f"Running generate with engine_prompt: {engine_prompt}, sampling_params: {sampling_params}, request_id: {request_id}"
        )
        generator = self.engine.generate(engine_prompt, sampling_params, request_id)

        async for response in self.chat_processor.stream_response(
            request,
            generator,
            request_id,
            conversation,
        ):
            vllm_logger.debug(f"Generated response: {response}")
            yield response


@triton_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("triton-init").component("vllm")
    await component.create_service()

    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(VllmEngine(engine_args).generate)


if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()
    asyncio.run(worker(engine_args))
