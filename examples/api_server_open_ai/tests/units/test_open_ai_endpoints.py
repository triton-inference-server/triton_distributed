# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

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

import json
import logging
import typing
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from triton_api_server.connector import (
    BaseTriton3Connector,
    InferenceRequest,
    InferenceResponse,
)
from triton_api_server.open_ai.chat import (
    DEFAULT_ROLE,
    INSTRUCTION_FOR_CONCISE,
    PREFIX,
    ChatCompletionResponse,
    Choice,
    Delta,
    Message,
    create_chat_response,
    create_chunk_responses,
    create_prompt,
    generate_delta,
)
from triton_api_server.open_ai.chat_vllm import (
    detokenize_output,
    tokenize_prompt,
)
from triton_api_server.open_ai.server import (
    create_app,  # Adjust the import according to your module structure
)

LOGGER = logging.getLogger(__name__)


class MockConnector(BaseTriton3Connector):
    async def inference(self, model_name: str, request: InferenceRequest):
        if model_name == "get_license":
            text = np.char.encode('{"license_key": "mocked_license_key"}', "utf-8")
            yield InferenceResponse(outputs={"result": text})
        elif model_name == "get_metadata":
            text = np.char.encode('{"metadata_key": "mocked_metadata_key"}', "utf-8")
            yield InferenceResponse(outputs={"result": text})
        elif model_name == "chat_model":
            messages = [{"role": "assistant", "content": "mocked_response"}]
            text = np.char.encode(f'{{"content": "{messages[0]["content"]}"}}', "utf-8")
            yield InferenceResponse(outputs={"result": text})
        elif model_name == "completion_model":
            prompt = request.inputs["prompt"]
            response = {"content": prompt + " mocked_completion"}
            text = np.char.encode(f'{{"content": "{response["content"]}"}}', "utf-8")
            yield InferenceResponse(outputs={"result": text})
        else:
            raise ValueError("Unknown model name")

    async def list_models(self) -> typing.List[str]:
        """List models available in Triton 3 system.

        Returns:
            List of model names.
        """
        return ["gpt-3.5-turbo"]


@pytest.fixture()
def app() -> FastAPI:
    mock_connector = MockConnector()
    app = FastAPI()
    create_app(mock_connector, app)
    return app


@pytest.fixture()
def async_client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app), base_url="http://test")


@pytest.mark.asyncio()
async def test_show_license(async_client: AsyncClient):
    async with async_client as ac:
        response = await ac.get("/v1/license")
    assert response.status_code == 200
    assert response.json() == {"license": {"license_key": "mocked_license_key"}}


@pytest.mark.asyncio()
async def test_show_metadata(async_client: AsyncClient):
    async with async_client as ac:
        response = await ac.get("/v1/metadata")
    assert response.status_code == 200
    assert response.json() == {"metadata": {"metadata_key": "mocked_metadata_key"}}


@pytest.mark.asyncio()
async def test_show_version(async_client: AsyncClient):
    async with async_client as ac:
        response = await ac.get("/v1/version")
    assert response.status_code == 200
    assert response.json() == {"release": "1.0.0", "api": "1.0.0"}


@pytest.mark.asyncio()
async def test_health_ready(async_client: AsyncClient):
    async with async_client as ac:
        response = await ac.get("/v1/health/ready")
    assert response.status_code == 200
    assert response.json() == {"message": "Service is ready."}


@pytest.mark.asyncio()
async def test_health_live(async_client: AsyncClient):
    async with async_client as ac:
        response = await ac.get("/v1/health/live")
    assert response.status_code == 200
    assert response.json() == {"message": "Service is live."}


@pytest.mark.asyncio()
async def test_list_models(async_client: AsyncClient):
    async with async_client as ac:
        response = await ac.get("/v1/models")
    assert response.status_code == 200
    print(response.json())
    assert response.json() == {
        "object": "list",
        "data": [
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": 0,
                "owned_by": "triton",
            }
        ],
    }


@pytest.mark.asyncio()
@pytest.mark.xfail
async def test_completions_non_stream(async_client: AsyncClient):
    request_payload = {
        "model": "completion_model",
        "prompt": "Once upon a time,",
        "max_tokens": 15,
        "temperature": 0.5,
        "top_p": 1.0,
        "n": 1,
        "stream": False,
    }
    async with async_client as ac:
        response = await ac.post("/v1/completions", json=request_payload)
    assert response.status_code == 200
    response_json = response.json()
    assert "choices" in response_json
    assert (
        response_json["choices"][0]["content"] == "Once upon a time, mocked_completion"
    )


@pytest.mark.asyncio()
@pytest.mark.xfail
async def test_completions_stream(async_client: AsyncClient):
    request_payload = {
        "model": "completion_model",
        "prompt": "Once upon a time,",
        "max_tokens": 15,
        "temperature": 0.5,
        "top_p": 1.0,
        "n": 1,
        "stream": True,
    }
    async with async_client as ac:
        response = await ac.post("/v1/completions", json=request_payload)
    assert response.status_code == 200
    async for chunk in response.aiter_text():
        assert "mocked_completion" in chunk
        break


def mock_tokenize(prompt: str, model: str):
    # Simple tokenization mock, each character as a token (for demonstration)
    tokens = [ord(c) for c in prompt]
    return np.array(tokens).reshape(1, -1)  # reshape to simulate batch size of 1


def mock_detokenize(token_ids: np.ndarray, model: str):
    # Assumes token_ids is an array where each row is a sequence of token IDs
    detokenized = ["".join(chr(id) for id in ids) for ids in token_ids]
    return detokenized


class MockConnectorChat(BaseTriton3Connector):
    def __init__(self):
        self.responses = {}
        self.last_request = None

    def set_response_generator(self, model_name: str, generation_function):
        self.responses[model_name] = generation_function

    async def inference(self, model_name: str, request: InferenceRequest):
        self.last_request = request
        if model_name in self.responses:
            generator = self.responses[model_name]
            for response_content in generator(request.inputs["input_ids"], model_name):
                yield InferenceResponse(outputs=response_content)
        else:
            raise ValueError("Unknown model name")


@pytest.fixture(scope="function")
def mock_connector_chat():
    return MockConnectorChat()


@pytest.fixture()
def model_name():
    return "chat_model"


@pytest.fixture()
def app_chat(mock_connector_chat, model_name) -> FastAPI:
    app = FastAPI()
    create_app(mock_connector_chat, app, model_name)
    return app


@pytest.fixture()
def async_client_chat(mock_connector_chat, app_chat: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app_chat), base_url="http://test")


@pytest.mark.asyncio()
@patch(
    "triton_api_server.open_ai.chat_vllm.detokenize_output", side_effect=mock_detokenize
)
@patch("triton_api_server.open_ai.chat_vllm.tokenize_prompt", side_effect=mock_tokenize)
async def test_chat_completions_no_messages(
    mock_detokenize_output,
    mock_tokenize_prompt,
    mock_connector_chat,
    async_client_chat: AsyncClient,
    app_chat: FastAPI,
    model_name,
):
    request_payload = {
        "model": model_name,
        "messages": [],
        "n": 1,
        "max_tokens": 15,
        "stream": False,
        "frequency_penalty": 0.0,
        "stop": [],
    }

    def response_generator(input_ids, model_name):
        detokenized_inputs = mock_detokenize(input_ids, model_name)
        output_str_list = [
            detokenized_input + " mocked_response"
            for detokenized_input in detokenized_inputs
        ]
        output_tokens_list = [
            mock_tokenize(output_str, model_name) for output_str in output_str_list
        ]
        output_tokens = np.concatenate(output_tokens_list)
        yield {"token_ids": output_tokens}

    mock_connector_chat.set_response_generator(model_name, response_generator)

    async with async_client_chat as ac:
        response = await ac.post("/v1/chat/completions", json=request_payload)
    print(response.json())
    assert response.status_code == 200
    response_json = response.json()
    LOGGER.info(response_json)
    assert "choices" in response_json
    assert response_json["choices"][0]["message"]["content"] == " mocked_response"


@pytest.mark.asyncio()
@patch(
    "triton_api_server.open_ai.chat_vllm.detokenize_output", side_effect=mock_detokenize
)
@patch("triton_api_server.open_ai.chat_vllm.tokenize_prompt", side_effect=mock_tokenize)
async def test_chat_completions_single_message(
    mock_detokenize_output,
    mock_tokenize_prompt,
    mock_connector_chat,
    async_client_chat: AsyncClient,
    app_chat: FastAPI,
):
    request_payload = {
        "model": "chat_model",
        "messages": [{"role": "user", "content": "Hello! How are you?"}],
        "n": 1,
    }

    def response_generator(input_ids, model_name):
        detokenized_inputs = mock_detokenize(input_ids, model_name)
        output_str_list = [
            detokenized_input + " mocked_response"
            for detokenized_input in detokenized_inputs
        ]
        output_tokens_list = [
            mock_tokenize(output_str, model_name) for output_str in output_str_list
        ]
        output_tokens = np.concatenate(output_tokens_list)
        yield {"token_ids": output_tokens}

    mock_connector_chat.set_response_generator("chat_model", response_generator)

    async with async_client_chat as ac:
        response = await ac.post("/v1/chat/completions", json=request_payload)

    assert response.status_code == 200
    response_json = response.json()
    assert "choices" in response_json
    assert response_json["choices"][0]["message"]["content"] == " mocked_response"


@pytest.mark.asyncio()
@patch(
    "triton_api_server.open_ai.chat_vllm.detokenize_output", side_effect=mock_detokenize
)
@patch("triton_api_server.open_ai.chat_vllm.tokenize_prompt", side_effect=mock_tokenize)
async def test_chat_completions_stream(
    mock_detokenize_output,
    mock_tokenize_prompt,
    mock_connector_chat,
    async_client_chat: AsyncClient,
    app_chat: FastAPI,
):
    request_payload = {
        "model": "chat_model",
        "messages": [{"role": "user", "content": "Hello! How are you?"}],
        "n": 1,
        "max_tokens": 15,
        "stream": True,
        "frequency_penalty": 0.0,
        "stop": [],
    }

    def response_generator(input_ids, model_name):
        detokenized_inputs = mock_detokenize(input_ids, model_name)
        chunks = [" first chunk", " second chunk"]
        accumulated_chunks = ""
        for chunk in chunks:
            accumulated_chunks += chunk
            output_str_list = [
                detokenized_input + accumulated_chunks
                for detokenized_input in detokenized_inputs
            ]
            output_tokens_list = [
                mock_tokenize(output_str, model_name) for output_str in output_str_list
            ]
            output_tokens = np.concatenate(output_tokens_list)
            yield {"token_ids": output_tokens}

    mock_connector_chat.set_response_generator("chat_model", response_generator)

    async with async_client_chat as ac:
        response = await ac.post("/v1/chat/completions", json=request_payload)
        assert response.status_code == 200

        collected_chunks = []
        async for chunk in response.aiter_lines():
            if chunk:
                collected_chunks.append(chunk)
                chunk_data = json.loads(chunk)
                LOGGER.info(chunk_data)
                assert "choices" in chunk_data
                assert "delta" in chunk_data["choices"][0]
                assert "content" in chunk_data["choices"][0]["delta"]

        # Final checks after all chunks are collected
        assert collected_chunks, "No chunks were collected in the response stream."
        final_message = "".join(
            [
                json.loads(chunk)["choices"][0]["delta"]["content"]
                for chunk in collected_chunks
                if json.loads(chunk)["choices"][0]["delta"]["content"] is not None
            ]
        )
        assert (
            final_message
            == 'ou are friendly AI from Triton project. User provided only single message with for you to answer as assistant. Your task is to produce a concise response. \n Messages in JSON:\n[{"role": "user", "content": "Hello! How are you?"}] first chunk second chunk'
        )


@pytest.fixture(scope="function")
def mock_auto_tokenizer(mocker):
    mock_tokenizer = MagicMock()
    mocker.patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
    )
    return mock_tokenizer


def test_tokenizer_handling(mock_auto_tokenizer):
    mock_auto_tokenizer.encode.return_value = np.array([[1212, 318, 1332, 3275]])

    model = "gpt2"
    prompt = "This is test message"
    tokens = tokenize_prompt(prompt, model)

    mock_auto_tokenizer.encode.assert_called_once_with(prompt, return_tensors="np")
    np.testing.assert_array_equal(tokens, np.array([[1212, 318, 1332, 3275]]))

    mock_auto_tokenizer.batch_decode.return_value = ["This is test message"]

    model = "gpt2"
    output = np.array([[1212, 318, 1332, 3275]])
    text = detokenize_output(output, model)

    mock_auto_tokenizer.batch_decode.assert_called_once_with(
        output, skip_special_tokens=True
    )
    assert text == ["This is test message"]


@pytest.fixture
def message_single_fixture():
    return Message(role="user", content="Hello! How are you?")


@pytest.fixture
def messages_fixture(message_single_fixture):
    return [
        message_single_fixture,
        Message(
            role="assistant", content="Hi! I am quite well, how can I help you today?"
        ),
        Message(role="user", content="Can you write me a song?"),
    ]


def test_create_prompt_no_messages():
    response, role = create_prompt([])
    assert PREFIX.strip() in response
    assert INSTRUCTION_FOR_CONCISE.strip() in response
    assert "any messages".strip() in response
    assert role == DEFAULT_ROLE


def test_create_prompt_single_message(message_single_fixture):
    response, role = create_prompt([message_single_fixture])
    assert PREFIX.strip() in response
    assert INSTRUCTION_FOR_CONCISE.strip() in response
    assert "single message" in response
    assert (
        'Messages in JSON:\n[{"role": "user", "content": "Hello! How are you?"}]'
        in response
    )
    assert role == DEFAULT_ROLE


def test_create_prompt_multiple_messages(messages_fixture):
    response, role = create_prompt(messages_fixture)
    assert PREFIX.strip() in response
    assert INSTRUCTION_FOR_CONCISE.strip() in response
    assert "series of messages".strip() in response
    assert (
        'Messages in JSON:\n[{"role": "user", "content": "Hello! How are you?"}, {"role": "assistant", "content": "Hi! I am quite well, how can I help you today?"}, {"role": "user", "content": "Can you write me a song?"}]'
        in response
    )
    assert role == DEFAULT_ROLE


def test_create_prompt_none_messages():
    response, role = create_prompt(None)
    assert PREFIX.strip() in response
    assert INSTRUCTION_FOR_CONCISE.strip() in response
    assert "any messages".strip() in response
    assert role == DEFAULT_ROLE


mock_detokenized_outputs = ["response one", "response two"]
mock_time = 1624290792
mock_model = "gpt-3.5-turbo"
mock_request_id = "chatcmpl-123"
mock_ai_role = "assistant"


@patch("time.time", return_value=mock_time)
def test_create_chat_response(mock_time_func):
    # Call the function with the detokenized outputs directly
    response = create_chat_response(
        mock_request_id, mock_model, mock_detokenized_outputs, mock_ai_role, ""
    )

    # Assertions
    assert isinstance(response, ChatCompletionResponse)
    assert response.id == mock_request_id
    assert response.object == "chat.completion.chunk"
    assert response.created == mock_time
    assert response.model == mock_model
    assert len(response.choices) == 2

    for i, choice in enumerate(response.choices):
        assert isinstance(choice, Choice)
        assert choice.index == i
        assert isinstance(choice.message, Message)
        assert choice.message.role == mock_ai_role
        assert choice.message.content == mock_detokenized_outputs[i]


def test_generate_delta_no_previous_output():
    result = generate_delta(output_str="Hello!", role="assistant")
    assert result == Delta(role="assistant", content="Hello!")


def test_generate_delta_with_previous_output_present():
    result = generate_delta(
        output_str="user: Hello! assistant: Hi!",
        role="assistant",
        previous_output="user: Hello! assistant: ",
    )
    assert result == Delta(role="assistant", content="Hi!")


def test_generate_delta_with_previous_output_not_present():
    result = generate_delta(
        output_str="Hello!", role="assistant", previous_output="Hi!"
    )
    assert result == Delta(role="assistant", content="Hello!")


def test_generate_delta_with_previous_output_none():
    result = generate_delta(output_str="Hello!", role="assistant", previous_output=None)
    assert result == Delta(role="assistant", content="Hello!")


def test_generate_delta_with_empty_previous_output():
    result = generate_delta(output_str="Hello!", role="assistant", previous_output="")
    assert result == Delta(role="assistant", content="Hello!")


def test_generate_delta_with_empty_output_string():
    result = generate_delta(output_str="", role="assistant", previous_output="Hello!")
    assert result == Delta(role="assistant", content="")


# Mocking the generate_delta function
def mock_generate_delta(output_str, role, previous_output=None):
    return Delta(role=role, content=output_str.replace(previous_output or "", ""))


def get_test_date():
    return datetime(2023, 8, 4).timestamp()


@pytest.fixture
def request_id():
    return "chatcmpl-123"


@pytest.fixture
def model():
    return "gpt-4o-mini"


@pytest.fixture
def ai_role():
    return "assistant"


@pytest.fixture
def previous_output():
    return ["user: Hello! assistant: "] * 2


@pytest.fixture
def model_output_only_new():
    return ["response one", "response two"]


@pytest.fixture
def model_output_with_overlap():
    return [
        "user: Hello! assistant: response one",
        "user: Hello! assistant: response two",
    ]


@pytest.fixture
def finish_reason():
    return "stop"


@patch("time.time", MagicMock(return_value=get_test_date()))
@patch("triton_api_server.open_ai.chat.generate_delta", side_effect=mock_generate_delta)
def test_create_chunk_responses_all_new_content(
    mock_generate, request_id, model, model_output_only_new, ai_role
):
    response, new_previous_output = create_chunk_responses(
        request_id=request_id,
        model=model,
        model_output=model_output_only_new,
        ai_role=ai_role,
    )
    assert response.id == request_id
    assert response.object == "chat.completion.chunk"
    assert response.created == get_test_date()
    assert response.model == model
    assert response.system_fingerprint == request_id
    assert len(response.choices) == len(model_output_only_new)
    assert new_previous_output == model_output_only_new
    for idx, choice in enumerate(response.choices):
        assert choice.index == idx
        assert choice.delta == Delta(role=ai_role, content=model_output_only_new[idx])
        assert choice.logprobs is None
        assert choice.finish_reason is None


@patch("time.time", MagicMock(return_value=get_test_date()))
@patch("triton_api_server.open_ai.chat.generate_delta", side_effect=mock_generate_delta)
def test_create_chunk_responses_with_mixed_content(
    mock_generate,
    request_id,
    model,
    model_output_with_overlap,
    ai_role,
    previous_output,
):
    response, new_previous_output = create_chunk_responses(
        request_id=request_id,
        model=model,
        model_output=model_output_with_overlap,
        ai_role=ai_role,
        previous_output=previous_output,
    )
    assert len(response.choices) == len(model_output_with_overlap)
    expected_new_content = ["response one", "response two"]
    assert new_previous_output == model_output_with_overlap
    for idx, choice in enumerate(response.choices):
        assert choice.delta == Delta(role=ai_role, content=expected_new_content[idx])


@patch("time.time", MagicMock(return_value=get_test_date()))
@patch("triton_api_server.open_ai.chat.generate_delta", side_effect=mock_generate_delta)
def test_create_chunk_responses_with_no_new_content(
    mock_generate, request_id, model, previous_output, ai_role
):
    response, new_previous_output = create_chunk_responses(
        request_id=request_id,
        model=model,
        model_output=previous_output,  # Simulate model output being identical to previous output
        ai_role=ai_role,
        previous_output=previous_output,
    )
    assert len(response.choices) == len(previous_output)
    assert new_previous_output == previous_output
    for idx, choice in enumerate(response.choices):
        assert choice.delta == Delta(role=ai_role, content="")  # No new content


@patch("time.time", MagicMock(return_value=get_test_date()))
@patch("triton_api_server.open_ai.chat.generate_delta", side_effect=mock_generate_delta)
def test_create_chunk_responses_with_finish_reason(
    mock_generate, request_id, model, model_output_only_new, ai_role, finish_reason
):
    response, new_previous_output = create_chunk_responses(
        request_id=request_id,
        model=model,
        model_output=model_output_only_new,
        ai_role=ai_role,
        finish_reason=finish_reason,
    )
    assert len(response.choices) == len(model_output_only_new)
    assert new_previous_output == model_output_only_new
    for choice in response.choices:
        assert choice.finish_reason == finish_reason


@pytest.mark.asyncio()
@patch(
    "triton_api_server.open_ai.chat_vllm.detokenize_output", side_effect=mock_detokenize
)
@patch("triton_api_server.open_ai.chat_vllm.tokenize_prompt", side_effect=mock_tokenize)
async def test_parameters_passed_to_model(
    mock_detokenize_output,
    mock_tokenize_prompt,
    mock_connector_chat,
    async_client_chat: AsyncClient,
    app_chat: FastAPI,
):
    request_payload = {
        "model": "chat_model",
        "messages": [{"role": "user", "content": "Hello! How are you?"}],
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.2,
        "best_of": 3,
        "n": 1,
        "max_tokens": 15,
        "stream": False,
        "frequency_penalty": 0.0,
        "stop": [],
    }

    def response_generator(input_ids, model_name):
        detokenized_inputs = mock_detokenize(input_ids, model_name)
        output_str_list = [
            detokenized_input + " mocked_response"
            for detokenized_input in detokenized_inputs
        ]
        output_tokens_list = [
            mock_tokenize(output_str, model_name) for output_str in output_str_list
        ]
        output_tokens = np.concatenate(output_tokens_list)
        yield {"token_ids": output_tokens}

    mock_connector_chat.set_response_generator("chat_model", response_generator)

    async with async_client_chat as ac:
        response = await ac.post("/v1/chat/completions", json=request_payload)
    assert response.status_code == 200
    triton_request = mock_connector_chat.last_request
    sampling_params = triton_request.parameters["sampling_params"]
    assert sampling_params["top_p"] == 0.9
    assert sampling_params["top_k"] == 50
    assert sampling_params["repetition_penalty"] == 1.2
    assert sampling_params["best_of"] == 3
    assert sampling_params["n"] == 1
    assert sampling_params["max_tokens"] == 15
    assert sampling_params["frequency_penalty"] == 0.0
    assert sampling_params["stop"] == []
