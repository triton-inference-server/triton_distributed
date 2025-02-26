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

import uuid
from typing import AsyncIterator

from common.chat_processor import ChatProcessor, ProcessMixIn
from common.protocol import MyRequestOutput, Tokens
from kv_router_cai.router import Router
from kv_router_cai.worker import VllmEngine
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import AnyTokenizer
from common.protocol import MyRequestOutput, Tokens, vLLMGenerateRequest
from compoundai import depends, nova_endpoint, service


@service(
    nova={
        "enabled": True,
        "namespace": "triton-init",
    },
)
class Processor(ProcessMixIn):
    """
    vLLM pre and post processing
    """

    engine = depends(VllmEngine)
    router = depends(Router)

    def __init__(self):
        model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.engine_args = AsyncEngineArgs(
            model=model,
            tokenizer=model,
            enable_prefix_caching=True,
            block_size=64,
            max_model_len=16384,
        )
        self.model_config = self.engine_args.create_model_config()
        self.tokenizer = self._create_tokenizer()
        self.chat_processor = ChatProcessor(self.tokenizer, self.model_config)

    def _create_tokenizer(self) -> AnyTokenizer:
        """Create a TokenizerGroup using engine arguments similar to VLLM's approach"""
        model_path = self.engine_args.model

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

    # @nova_endpoint()
    # async def generate(self, text: str):
    #     """Forward requests to backend."""
    #     text = f"{text}-mid"
    #     print(f"Middle received: {text}")
    #     async for response in self.router.generate(text):
    #         print(f"Middle received response: {response}")
    #         yield f"Middle: {response}"

    # @nova_endpoint()
    # async def generate(self, raw_request: ChatCompletionRequest):
    #     """Forward requests to backend."""
    #     print("RAW REQUEST: ", raw_request)
    #     text = raw_request.messages[0]["content"]
    #     print("TEXT: ", text)
    #     # print("Middle received: ", text)
    #     # text = f"{text}-mid"
    #     # print(f"Middle received: {text}")
    #     # async for response in self.router.generate(text):
    #     #     print(f"Middle received response: {response}")
    #     #     yield f"Middle: {response}"
    #     for token in text.split():
    #         yield f"Middle: {token}"

    @nova_endpoint()
    async def generate(self, raw_request: ChatCompletionRequest):
        print("RAW REQUEST: ", raw_request)
        print("RAW REQUEST TYPE: ", type(raw_request))
        request_id = str(uuid.uuid4())
        print(f"Got raw request: {raw_request}")
        (
            request,
            conversation,
            prompt,
            engine_prompt,
            sampling_params,
        ) = await self._parse_raw_request(raw_request)
        worker_id = None
        async for worker in self.router.generate(
            Tokens(tokens=engine_prompt["prompt_token_ids"]).model_dump_json()
        ):
            worker_id = worker
            print("worker_id: ", worker_id)
            break
        print(f"Worker ID: {worker_id}")

        for token in ["Hello", "World"]:
            yield f"Middle: {token}"

        if worker_id == "":
            print("PROCESSOR usinge mpty worker_id: ", worker_id)
            engine_generator = await self.workers.generate(
                vLLMGenerateRequest(
                    engine_prompt=engine_prompt,
                    sampling_params=sampling_params,
                    request_id=request_id,
                ).model_dump_json()
            )
        else:
            print("PROCESSOR using worker_id: ", worker_id)
            engine_generator = await self.workers.direct(
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
