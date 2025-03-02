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

import base64
import time
import uuid
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field
from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
    CompletionRequest,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DisaggregatedParams,
    UsageInfo,
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


class DisaggCompletionResponse(CompletionStreamResponse):
    model_config = ConfigDict(extra="allow")
    disaggregated_params: DisaggregatedParams = {}


class nvCompletionResponseStreamChoice(CompletionResponseStreamChoice):
    disaggregated_params: Optional[DisaggregatedParams] = Field(default=None)

    @staticmethod
    def to_disaggregated_params(
        tllm_disagg_params: LlmDisaggregatedParams,
    ) -> DisaggregatedParams:
        if tllm_disagg_params is None:
            return None
        else:
            encoded_opaque_state = (
                base64.b64encode(tllm_disagg_params.opaque_state).decode("utf-8")
                if tllm_disagg_params is not None
                else None
            )
            return DisaggregatedParams(
                request_type=tllm_disagg_params.request_type,
                first_gen_tokens=tllm_disagg_params.first_gen_tokens,
                ctx_request_id=tllm_disagg_params.ctx_request_id,
                encoded_opaque_state=encoded_opaque_state,
            )


class nvCompletionStreamResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default_factory=lambda: f"cmpl-{str(uuid.uuid4().hex)}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[nvCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)
