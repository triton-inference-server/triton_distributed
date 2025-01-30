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

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from llm.api_server.chat import ChatHandler, generate_sampling_params
from llm.api_server.connector import BaseTriton3Connector, InferenceResponse
from schemas.openai import CreateChatCompletionRequest

LOGGER = logging.getLogger(__name__)


# FIXME: Share request conversion logic where applicable
def generate_sampling_params_vllm(
    request: CreateChatCompletionRequest,
    non_supported_params: Optional[List[str]] = None,
) -> dict:
    """
    Generate sampling params for vLLM from the request.

    Args:
        request: CreateChatCompletionRequest object.

    Returns:
        dict: Sampling params for vLLM.
    """

    errors_message = ""

    if request.logprobs:
        errors_message += "The parameter 'logprobs' set to True is not supported. "
    if request.tools and request.tools.type != "text":
        errors_message += (
            f"The parameter 'tools' type {request.tools.type} is not supported. "
        )
    if errors_message:
        raise ValueError(errors_message)

    if non_supported_params is None:
        non_supported_params = [
            "logit_bias",
            "top_logprobs",
            "tool_choice",
            "user",
            "service_tier",
        ]

    sampling_params = generate_sampling_params(request, non_supported_params)

    # NOTE: vLLM parameters (ex: top_k) not supported until added to schema
    return sampling_params


class ChatHandlerVllm(ChatHandler):
    def __init__(
        self, triton_connector: BaseTriton3Connector, model_name: str, tokenizer: str
    ):
        super().__init__(triton_connector, tokenizer)
        self._model_name = model_name

    def translate_chat_inputs(
        self, request: CreateChatCompletionRequest, request_id: str, prompt: str
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Translate the chat completion request to inference request"""

        if self._model_name is not None and self._model_name != request.model:
            raise ValueError(
                f"Model name mismatch: {self._model_name} != {request.model}"
            )
        inputs: Dict[str, np.ndarray] = {}
        sampling_params = generate_sampling_params_vllm(request)
        parameters = {
            "sampling_params": sampling_params,
            "request_id": request_id,
            "prompt": prompt,
        }
        return inputs, parameters

    def translate_chat_outputs(
        self, response: InferenceResponse, model_name: str
    ) -> Dict[str, Any]:
        """Translate the inference outputs to chat completion response"""
        if "text" in response.parameters:
            return {"model_output": [response.parameters["text"]]}
        elif "text_output" in response.outputs:
            return {"model_output": response.outputs["text_output"][0]}
