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


from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler, field_validator
from pydantic_core import core_schema
import json

from typing import Optional, List, Any
from typing_extensions import NotRequired
import msgspec

from vllm import CompletionOutput
from vllm.sequence import PromptLogprobs, RequestMetrics
from vllm import TokensPrompt, SamplingParams


class Request(BaseModel):
    prompt: str
    sampling_params: dict


class Tokens(BaseModel):
    tokens: list[int]

# Hack to override the type of multi_modal_data in TokensPrompt
# as pydantic doesn't understand generic typess
# TokensPrompt.__annotations__["multi_modal_data"] = Optional[Any]
class PatchedTokensPrompt(TokensPrompt):
    multi_modal_data: NotRequired[Optional[Any]]

# Monkey-patch the SamplingParams type to add a dummy core schema.
# def sampling_params_schema(cls, source: type[Any], handler: GetCoreSchemaHandler):
#     return core_schema.any_schema()

SamplingParams.__get_pydantic_core_schema__ = classmethod(
    lambda cls, source, handler: core_schema.any_schema()
)


class vLLMGenerateRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    engine_prompt: PatchedTokensPrompt
    sampling_params: SamplingParams
    request_id: str

    @field_validator("sampling_params", mode="before")
    @classmethod
    def parse_sampling_params(cls, v: any) -> SamplingParams:
        if isinstance(v, str):
            v = json.loads(v)
        if isinstance(v, dict):
            return SamplingParams(**v)
        return v

    model_config = ConfigDict(
        json_encoders={SamplingParams: lambda v: msgspec.json.encode(v)}
    )

class PrefillRequest(Request):
    request_id: str


class Response(BaseModel):
    text: str


class PrefillResponse(BaseModel):
    prefilled: bool


class MyRequestOutput(BaseModel):
    """The output data of a completion request to the LLM.
    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
                For encoder/decoder models, this is the
                decoder input prompt.
        prompt_token_ids: The token IDs of the prompt.
                          For encoder/decoder models, this is the
                          decoder input prompt token ids.
        prompt_logprobs: The log probabilities to return per prompt token.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
        metrics: Metrics associated with the request.
        lora_request: The LoRA request that was used to generate the output.
        encoder_prompt: The encoder prompt string of the request.
                        None if decoder-only.
        encoder_prompt_token_ids: The token IDs of the encoder prompt.
                                  None if decoder-only.
        num_cached_tokens: The number of tokens with prefix cache hit.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str
    prompt: Optional[str] = None
    prompt_token_ids: Optional[List[int]] = None
    prompt_logprobs: Optional[PromptLogprobs] = None
    outputs: List[CompletionOutput]
    finished: bool
    metrics: Optional[RequestMetrics] = None
    # lora_request: Optional[LoRARequest] = None
    # encoder_prompt: Optional[str] = None
    # encoder_prompt_token_ids: Optional[List[int]] = None
    # num_cached_tokens: Optional[int] = None
    # multi_modal_placeholders: Optional[MultiModalPlaceholderDict] = None