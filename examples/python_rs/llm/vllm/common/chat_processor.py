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

import json
from typing import AsyncIterator, List

from vllm import TokensPrompt
from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import ConversationMessage
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import RequestPrompt
from vllm.transformers_utils.tokenizer import AnyTokenizer


class ChatProcessor:
    def __init__(self, tokenizer: AnyTokenizer, model_config: ModelConfig):
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.openai_serving = OpenAIServingChat(
            engine_client=None,
            model_config=model_config,
            models=None,
            request_logger=None,
            response_role="assistant",
            chat_template=None,
            chat_template_content_format="auto",
        )

    def parse_raw_request(self, raw_request: dict) -> ChatCompletionRequest:
        return ChatCompletionRequest.parse_obj(raw_request)

    async def preprocess(
        self, raw_request: dict
    ) -> tuple[ConversationMessage, RequestPrompt, TokensPrompt]:
        request = self.parse_raw_request(raw_request)

        (
            conversation,
            request_prompts,
            engine_prompts,
        ) = await self.openai_serving._preprocess_chat(
            request,
            self.tokenizer,
            request.messages,
            chat_template=request.chat_template or self.tokenizer.chat_template,
            chat_template_content_format=self.openai_serving.chat_template_content_format,
            add_generation_prompt=request.add_generation_prompt,
            continue_final_message=request.continue_final_message,
            tool_dicts=None,
            documents=request.documents,
            chat_template_kwargs=request.chat_template_kwargs,
            tool_parser=self.openai_serving.tool_parser,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
            add_special_tokens=request.add_special_tokens,
        )

        return conversation[0], request_prompts[0], engine_prompts[0]

    async def stream_response(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator,
        request_id: str,
        conversation: List,
    ):
        request_metadata = RequestResponseMetadata(request_id=request_id)
        assert request.stream, "Only stream is supported"
        async for raw_response in self.openai_serving.chat_completion_stream_generator(
            request,
            result_generator,
            request_id,
            request.model,
            conversation,
            self.tokenizer,
            request_metadata,
        ):
            if raw_response.startswith("data: [DONE]"):
                break
            response = json.loads(raw_response.lstrip("data: "))
            yield response
