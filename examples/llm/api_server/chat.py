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
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from llm.api_server.connector import (
    BaseTriton3Connector,
    InferenceRequest,
    InferenceResponse,
    TritonInferenceError,
)

# FIXME: Integrate better with api_server library
from schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionFinishReason,
    ChatCompletionResponseMessage,
    ChatCompletionStreamingResponseChoice,
    ChatCompletionStreamResponseDelta,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    ObjectType,
)
from transformers import AutoTokenizer

LOGGER = logging.getLogger(__name__)


"""
Example request with curl
curl -X 'POST' \\
  'http://{host}:{port}/v1/chat/completions' \\
  -H 'accept: application/json' \\
  -H 'Content-Type: application/json' \\
  -d '{{
    "model": "{model}",
    "messages": [
      {{
        "role":"user",
        "content":"Hello! How are you?"
      }},
      {{
        "role":"assistant",
        "content":"Hi! I am quite well, how can I help you today?"
      }},
      {{
        "role":"user",
        "content":"Can you write me a song?"
      }}
    ],
    "top_p": 1,
    "n": 1,
    "max_tokens": 15,
    "stream": true,
    "frequency_penalty": 1.0,
    "stop": ["hello"]
  }}'
"""


def generate_sampling_params(
    request: CreateChatCompletionRequest,
    non_supported_params_none: Optional[List[str]] = None,
) -> Dict[str, Any]:
    errors_message = ""
    if not non_supported_params_none:
        non_supported_params_none = []
    for param in non_supported_params_none:
        if getattr(request, param, None) is not None:
            errors_message += f"The parameter '{param}' is not supported. "

    if errors_message:
        raise ValueError(errors_message)

    sampling_params = {}

    if request.temperature is not None:
        sampling_params["temperature"] = request.temperature
    if request.n is not None:
        sampling_params["n"] = request.n
    if request.top_p is not None:
        sampling_params["top_p"] = request.top_p
    if request.presence_penalty is not None:
        sampling_params["presence_penalty"] = request.presence_penalty
    if request.frequency_penalty is not None:
        sampling_params["frequency_penalty"] = request.frequency_penalty
    if request.max_tokens is not None:
        sampling_params["max_tokens"] = request.max_tokens
    if request.min_tokens is not None:
        sampling_params["min_tokens"] = request.min_tokens
    if request.stop is not None:
        sampling_params["stop"] = request.stop
    if request.seed is not None:
        sampling_params["seed"] = request.seed

    return sampling_params


def create_chat_response(
    request_id: str,
    model: str,
    model_output: Union[np.ndarray, List[str]],
    role: str,
    prompt: str,
) -> CreateChatCompletionResponse:
    """Create chunk responses from the detokenized outputs for non-streaming completions"""

    detokenized_outputs = model_output
    # Extract prompt from detokenized_outputs
    cleaned_outputs = []
    for detokenized_output in detokenized_outputs:
        # FIXME: Should this be handled by 'echo' param instead?
        if detokenized_output.startswith(prompt):
            cleaned_output = detokenized_output[len(prompt) :]
        else:
            cleaned_output = detokenized_output
        cleaned_outputs.append(cleaned_output)

    messages = [
        ChatCompletionResponseMessage(role=role, content=output_str)
        for output_str in cleaned_outputs
    ]
    choices = [
        ChatCompletionChoice(
            index=idx,
            message=message,
            finish_reason=ChatCompletionFinishReason.stop,
            logprobs=None,
        )
        for idx, message in enumerate(messages)
    ]

    chat_response = CreateChatCompletionResponse(
        id=request_id,
        choices=choices,
        created=int(time.time()),
        model=model,
        system_fingerprint=None,
        object=ObjectType.chat_completion,
    )
    return chat_response


def generate_delta(
    output_str: str, role: str, previous_output: Optional[str] = None
) -> ChatCompletionStreamResponseDelta:
    """Generate the delta from the output string

    Args:
        output_str (str): The output string from the model.
        role (str): The role of the AI generating the output.
        previous_output (Optional[str]): The previous output string. Defaults to None.

    Example:
        print(generate_delta("user: Hello!, assistant: Hi!", "assistant", "user: Hello!, assistant: "))
        # Output: Delta(role='assistant', content='Hi!')
    """
    if previous_output is None:
        return ChatCompletionStreamResponseDelta(role=role, content=output_str)
    else:
        # FIXME: Should we be manually finding the delta here? Or full text from last response?
        delta_start = output_str.find(previous_output)
        if delta_start == -1:
            LOGGER.warning(
                f"Previous output \n<START>\n{previous_output}\n<END>\n not found in the output string: \n<START>\n{output_str}\n<END>\n"
            )
            return ChatCompletionStreamResponseDelta(role=role, content=output_str)
        delta = output_str[delta_start + len(previous_output) :]
        return ChatCompletionStreamResponseDelta(role=role, content=delta)


def create_chunk_responses(
    request_id: str,
    model: str,
    model_output: List[str],
    role: str,
    previous_output: Optional[list[str]] = None,
    logprobs: Optional[np.ndarray] = None,
    finish_reason: Optional[str] = None,
) -> Tuple[CreateChatCompletionStreamResponse, List[str]]:
    """Create chunk responses from the detokenized outputs for streaming completions.

    Function extracts the delta from the output string and creates a chunk response. It also updates the previous output.

    Args:
        request_id (str): The unique identifier for the request.
        model (str): The model used for generating the response.
        model_output (List[str]): The list of output strings from the model.
        role (str): The role of the AI generating the output.
        previous_output (Optional[str]): The previous output string. Defaults to None.
        logprobs (Optional[np.ndarray]): The log probabilities of the output tokens. Defaults to None.
        finish_reason (Optional[str]): The reason for stopping the completion. Defaults to None.

    Returns:
        Tuple[CreateChatCompletionStreamResponse, List[str]]: A tuple containing the chunk response and the new previous output.

    Example:
        create_chunk_responses(
            request_id="chatcmpl-123",
            model="gpt-4o-mini",
            model_output=["I am fine, thank you!", "How can I help you?"],
            role="assistant",
            previous_output="user: Hello!, assistant: ",
            logprobs=np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
            finish_reason="stop"
        ))
    """

    detokenized_outputs = model_output

    new_previous_output = []

    deltas = []
    for idx, output_str in enumerate(detokenized_outputs):
        if previous_output is not None:
            previous_output_row = previous_output[idx]
        else:
            previous_output_row = None
        delta = generate_delta(
            output_str=output_str, role=role, previous_output=previous_output_row
        )
        deltas.append(delta)
        new_previous_output.append(output_str)

    choices = []
    for idx, delta in enumerate(deltas):
        choice_kwargs = {
            "index": idx,
            "delta": delta,
            # FIXME: Validate finish_reason behavior on first vs last responses
            "finish_reason": finish_reason,
            "logprobs": None,
        }
        if logprobs is not None:
            choice_kwargs["logprobs"] = logprobs[idx]
        chunk_choice = ChatCompletionStreamingResponseChoice(**choice_kwargs)
        choices.append(chunk_choice)
    chunk_response = CreateChatCompletionStreamResponse(
        id=request_id,
        object=ObjectType.chat_completion_chunk,
        created=int(time.time()),
        model=model,
        system_fingerprint=request_id,
        choices=choices,
    )
    return chunk_response, new_previous_output


class ChatHandler:
    def __init__(self, triton_connector: BaseTriton3Connector, tokenizer: str):
        self._triton_connector = triton_connector
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def translate_chat_inputs(
        self, request: CreateChatCompletionRequest, request_id: str, prompt: str
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        raise NotImplementedError("This method is not implemented yet")

    def translate_chat_outputs(
        self, response: InferenceResponse, model_name: str
    ) -> Dict[str, Any]:
        raise NotImplementedError("This method is not implemented yet")

    def stream_response_adaptor(self, response_stream):
        async def adaptor_stream():
            async for response in response_stream():
                if isinstance(response, Exception):
                    yield self.exception_adaptor(response).body
                else:
                    yield response.model_dump_json() + "\n"

        return StreamingResponse(adaptor_stream(), media_type="application/json")

    def response_adaptor(self, response):
        return response.model_dump_json()

    def exception_adaptor(self, exception):
        return JSONResponse(
            content={"error": str(exception), "code": 500}, status_code=500
        )

    async def process_request(self, request: Any, raw_request: Optional[Request]):
        request_id = str(uuid.uuid4())
        LOGGER.debug(f"{request=}")
        prompt, role = self._create_prompt(request)
        inputs, parameters = self.translate_chat_inputs(request, request_id, prompt)

        triton_request = InferenceRequest(inputs=inputs, parameters=parameters)

        # Streaming
        if request.stream:
            response_stream = self._stream_response_factory(
                request_id, request.model, triton_request, prompt, role
            )
            return self.stream_response_adaptor(response_stream)

        # Non-Streaming
        response_data = None
        try:
            chat_outputs = None
            async for response in self._triton_connector.inference(
                request.model, triton_request
            ):
                chat_outputs = self.translate_chat_outputs(response, request.model)
            kwargs = {
                "request_id": request_id,
                "model": request.model,
                "role": role,
                "prompt": prompt,
            }
            if chat_outputs is not None:
                kwargs.update(chat_outputs)
            response_data = create_chat_response(**kwargs)
        except TritonInferenceError as e:
            logging.error(f"Error processing chat completion request: {e}")
            return self.exception_adaptor(e)

        LOGGER.info(f"Chat completion response: {response_data}")
        return self.response_adaptor(response_data)

    def _stream_response_factory(self, request_id, model, triton_request, prompt, role):
        async def stream_response():
            try:
                previous_output = None
                async for response in self._triton_connector.inference(
                    model, triton_request
                ):
                    # FIXME: Detect stop in response
                    try:
                        chat_outputs = self.translate_chat_outputs(response, model)
                    except KeyError as e:
                        LOGGER.info(f"KeyError {e} in response: {response}")
                        break
                    model_output = chat_outputs["model_output"]
                    chunk_response, new_previous_output = create_chunk_responses(
                        request_id=request_id,
                        model=model,
                        model_output=model_output,
                        role=role,
                        previous_output=previous_output,
                    )
                    previous_output = new_previous_output
                    yield f"data: {chunk_response.model_dump_json(exclude_unset=True)}\n\n"
            except TritonInferenceError as e:
                logging.error(f"Error processing chat completion request: {e}")
                # FIXME: Does this need to conform to SSE standard for errors?
                yield JSONResponse(
                    content={"error": str(e), "code": 500}, status_code=500
                ).body
            finally:
                yield "data: [DONE]\n\n"

        return stream_response

    # FIXME: Use shared/common module for these functions between
    # TritonLLMEngine and TritonDistributedEngine implementations.
    def _get_first_response_role(
        self, conversation: List[Dict], add_generation_prompt: bool, default_role: str
    ) -> str:
        if add_generation_prompt:
            return default_role

        return conversation[-1]["role"]

    def _create_prompt(self, request: CreateChatCompletionRequest) -> Tuple[str, str]:
        """Create a prompt for vLLM model from the messages"""
        conversation = [
            message.model_dump(exclude_none=True) for message in request.messages
        ]
        add_generation_prompt = True

        prompt = self.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        LOGGER.debug(f"{prompt=}")

        default_role = "assistant"
        role = self._get_first_response_role(
            conversation, add_generation_prompt, default_role
        )

        return prompt.strip(), role
