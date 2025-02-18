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

import vllm
from common.chat_processor import ChatProcessor
from vllm.engine.arg_utils import AsyncEngineArgs


class BaseVllmEngine:
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine_args: AsyncEngineArgs):
        self.model_config = engine_args.create_model_config()
        self.engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        self.chat_processor = ChatProcessor(self.engine, self.model_config)

    async def _parse_raw_request(self, raw_request):
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
        return self.chat_processor.stream_response(
            request,
            generator,
            request_id,
            conversation,
        )

    @abc.abstractmethod
    async def generate(self, raw_request):
        pass
