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

from pydantic import ConfigDict
from tensorrt_llm.llmapi import DisaggregatedParams
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
    CompletionRequest,
)


class nvChatCompletionRequest(ChatCompletionRequest):
    model_config = ConfigDict(extra="allow")


class DisaggChatCompletionRequest(nvChatCompletionRequest):
    disaggregated_params: dict = {}
    model_config = ConfigDict(extra="allow")


class DisaggChatStreamCompletionResponse(ChatCompletionStreamResponse):
    disaggregated_params: dict = {}
    model_config = ConfigDict(extra="allow")


class DisaggregatedResponse(ChatCompletionStreamResponse):
    text: str
    disaggregated_params: DisaggregatedParams = {}


class nvCompletionRequest(CompletionRequest):
    model_config = ConfigDict(extra="allow")


# class nvCompletionStreamResponse(BaseModel):
#     model: str
#     choices: List[CompletionResponseStreamChoice]
