# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import AsyncIterator

from engine.engine import LLMEngine
from llm.api_server.chat_tensorrtllm import ChatHandlerTensorrtLLM
from llm.api_server.chat_vllm import ChatHandlerVllm
from llm.api_server.remote_model_connector import RemoteModelConnector
from schemas.openai import (
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateCompletionRequest,
    CreateCompletionResponse,
    Model,
    ObjectType,
)


class TritonDistributedTensorrtLLMChatHandler(ChatHandlerTensorrtLLM):
    def __init__(
        self, triton_connector: RemoteModelConnector, model_name: str, tokenizer: str
    ):
        super().__init__(triton_connector, model_name, tokenizer)

    # Request / response format can vary between frontends, so allow override
    # of adaptor functions accordingly.
    def stream_response_adaptor(self, response_stream):
        async def adaptor_stream():
            async for response in response_stream():
                if isinstance(response, Exception):
                    raise response
                else:
                    # Already in SSE String format
                    yield response

        return adaptor_stream

    def response_adaptor(self, response):
        return response

    def exception_adaptor(self, exception):
        raise exception


class TritonDistributedChatHandler(ChatHandlerVllm):
    def __init__(
        self, triton_connector: RemoteModelConnector, model_name: str, tokenizer: str
    ):
        super().__init__(triton_connector, model_name, tokenizer)

    # Request / response format can vary between frontends, so allow override
    # of adaptor functions accordingly.
    def stream_response_adaptor(self, response_stream):
        async def adaptor_stream():
            async for response in response_stream():
                if isinstance(response, Exception):
                    raise response
                else:
                    # Already in SSE String format
                    yield response

        return adaptor_stream

    def response_adaptor(self, response):
        return response

    def exception_adaptor(self, exception):
        raise exception


class TritonDistributedEngine(LLMEngine):
    def __init__(
        self,
        nats_url: str,
        data_plane_host: str,
        data_plane_port: int,
        model_name: str,
        tokenizer: str,
        backend: str,
    ):
        self.triton_connector = RemoteModelConnector(
            nats_url=nats_url,
            data_plane_host=data_plane_host,
            data_plane_port=data_plane_port,
            model_name=model_name,
            keep_dataplane_endpoints_open=True,
        )

        if not backend or backend == "vllm":
            # FIXME: Consider supporting multiple or per-model tokenizers
            self.request_handler = TritonDistributedChatHandler(
                self.triton_connector, model_name, tokenizer
            )
        else:
            self.request_handler = TritonDistributedTensorrtLLMChatHandler(
                self.triton_connector, model_name, tokenizer
            )

    async def chat(
        self, request: CreateChatCompletionRequest
    ) -> CreateChatCompletionResponse | AsyncIterator[str]:
        """
        If request.stream is True, this returns an AsyncIterator (or Generator) that
        produces server-sent-event (SSE) strings in the following form:
            'data: {CreateChatCompletionStreamResponse}\n\n'
            ...
            'data: [DONE]\n\n'

        If request.stream is False, this returns a CreateChatCompletionResponse.
        """
        # FIXME: Unify call whether streaming or not
        if request.stream:
            response_generator = await self.request_handler.process_request(
                request, None
            )
            return response_generator()

        response = await self.request_handler.process_request(request, None)
        return response

    async def completion(
        self, request: CreateCompletionRequest
    ) -> CreateCompletionResponse | AsyncIterator[str]:
        """
        If request.stream is True, this returns an AsyncIterator (or Generator) that
        produces server-sent-event (SSE) strings in the following form:
            'data: {CreateCompletionResponse}\n\n'
            ...
            'data: [DONE]\n\n'

        If request.stream is False, this returns a CreateCompletionResponse.
        """
        raise NotImplementedError

    def ready(self) -> bool:
        """
        Returns True if the engine is ready to accept inference requests, or False otherwise.
        """
        # FIXME: Add more useful checks if available.
        return True

    def metrics(self) -> str:
        """
        Returns the engine's metrics in a Prometheus-compatible string format.
        """
        raise NotImplementedError

    def models(self) -> list[Model]:
        """
        Returns a List of OpenAI Model objects.
        """
        # FIXME: Support 'async def models()'
        model_names = asyncio.run(self.triton_connector.list_models())

        models = [
            Model(
                id=model_name,
                object=ObjectType.model,
                owned_by="Triton Distributed",
                # FIXME: Need to track creation times, so set to 0 for now.
                created=0,
            )
            for model_name in model_names
        ]

        return models
