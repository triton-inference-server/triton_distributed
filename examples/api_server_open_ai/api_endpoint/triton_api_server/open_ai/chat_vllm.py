# Copyright 2024-2025, NVIDIA CORPORATION. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from schemas.openai import CreateChatCompletionRequest
from triton_api_server.connector import BaseTriton3Connector, InferenceResponse
from triton_api_server.open_ai.chat import ChatHandler, generate_sampling_params

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
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], str, str]:
        """Translate the chat completion request to inference request"""

        if self._model_name is not None and self._model_name != request.model:
            raise ValueError(
                f"Model name mismatch: {self._model_name} != {request.model}"
            )
        inputs = {}
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
        return {"model_output": [response.parameters["text"]]}
