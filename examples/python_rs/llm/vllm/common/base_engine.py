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

import abc

from common.chat_processor import ChatProcessor
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)


class BaseVllmEngine:
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine_args: AsyncEngineArgs):
        self.engine_args = engine_args
        self.model_config = self.engine_args.create_model_config()
        self.engine_client = None
        self.chat_processor = None
        self._engine_context = None

    async def initialize(self):
        """Initialize the engine client and related components."""
        print("Initializing engine client")
        self._engine_context = build_async_engine_client_from_engine_args(
            self.engine_args
        )
        self.engine_client = await self._engine_context.__aenter__()
        self.chat_processor = ChatProcessor(self.engine_client, self.model_config)

    async def cleanup(self):
        """Cleanup resources."""
        print("Cleaning up engine client")
        if self._engine_context is not None:
            await self._engine_context.__aexit__(None, None, None)
            self.engine_client = None
            self._engine_context = None

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.cleanup()

    async def _parse_raw_request(self, raw_request):
        assert self.engine_client is not None
        request = self.chat_processor.parse_raw_request(raw_request)
        (
            conversation,
            request_prompt,
            engine_prompt,
        ) = await self.chat_processor.preprocess(raw_request)
        default_max_tokens = self.model_config.max_model_len - len(
            engine_prompt["prompt_token_ids"]
        )
        default_sampling_params = self.model_config.get_diff_sampling_param()
        sampling_params = request.to_sampling_params(
            default_max_tokens,
            self.model_config.logits_processor_pattern,
            default_sampling_params,
        )
        return request, conversation, request_prompt, engine_prompt, sampling_params

    async def _stream_response(self, request, generator, request_id, conversation):
        assert self.engine_client is not None
        return self.chat_processor.stream_response(
            request,
            generator,
            request_id,
            conversation,
        )

    @abc.abstractmethod
    async def generate(self, raw_request):
        pass
