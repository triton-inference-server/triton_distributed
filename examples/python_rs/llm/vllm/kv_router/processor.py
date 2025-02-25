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
from common.chat_processor import ChatProcessor, ProcessMixIn
from common.parser import parse_vllm_args
from common.protocol import MyRequestOutput, Tokens, vLLMGenerateRequest
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
)
from vllm.logger import logger as vllm_logger
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import AnyTokenizer

from triton_distributed._core import Client
from triton_distributed.runtime import (
    DistributedRuntime,
    triton_endpoint,
    triton_worker,
)


class Processor(ProcessMixIn):
    """
    vLLM pre and post processing
    """

    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        router_client: Client,
        workers_client: Client,
    ):
        self.engine_args = engine_args
        self.model_config = self.engine_args.create_model_config()
        self.tokenizer = self._create_tokenizer(engine_args)
        self.chat_processor = ChatProcessor(self.tokenizer, self.model_config)
        self.router_client = router_client
        self.workers_client = workers_client

    def _create_tokenizer(self, engine_args: AsyncEngineArgs) -> AnyTokenizer:
        """Create a TokenizerGroup using engine arguments similar to VLLM's approach"""
        model_path = engine_args.model

        # Create the base tokenizer with VLLM's typical settings
        base_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
            truncation_side="left",
            use_fast=True,  # VLLM might use the fast tokenizer for efficiency
        )
        return base_tokenizer

    async def generate_responses(
        self, engine_generator
    ) -> AsyncIterator[RequestOutput]:
        async for resp in engine_generator:
            # Deserialize the response from the engine
            # Creates correct vLLM objects for each field
            output = MyRequestOutput.model_validate_json(resp.data())

            # OpenAIServingChat.chat_completion_stream_generator() method expects a RequestOutput object
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
        worker_id_generator: AsyncIterator = await self.router_client.generate(
            Tokens(tokens=engine_prompt["prompt_token_ids"]).model_dump_json()
        )

        worker_id = (
            await worker_id_generator.__anext__()
        )  # only one worker id is returned
        worker_id = worker_id.data()
        vllm_logger.info(f"Worker ID: {worker_id}")

        if worker_id == "":
            engine_generator = await self.workers_client.random(
                vLLMGenerateRequest(
                    engine_prompt=engine_prompt,
                    sampling_params=sampling_params,
                    request_id=request_id,
                ).model_dump_json()
            )
        else:
            engine_generator = await self.workers_client.direct(
                vLLMGenerateRequest(
                    engine_prompt=engine_prompt,
                    sampling_params=sampling_params,
                    request_id=request_id,
                ).model_dump_json(),
                uuid.UUID(worker_id).int,
            )

        output = self.generate_responses(engine_generator)

        async for response in await self._stream_response(
            request, output, request_id, conversation
        ):
            yield response


@triton_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    """
    Set up clients to the router and workers.
    Serve the triton-init.process.chat/completions function.
    """
    workers_client = (
        await runtime.namespace("triton-init")
        .component("vllm")
        .function("generate")
        .client()
    )

    router_client = (
        await runtime.namespace("triton-init")
        .component("router")
        .function("generate")
        .client()
    )

    preprocess_component = runtime.namespace("triton-init").component("process")
    await preprocess_component.create_service()
    preprocess_endpoint = preprocess_component.function("chat/completions")

    processor = Processor(engine_args, router_client, workers_client)
    assert isinstance(processor, ProcessMixIn)
    await preprocess_endpoint.serve_endpoint(processor.generate)


if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()
    asyncio.run(worker(engine_args))
