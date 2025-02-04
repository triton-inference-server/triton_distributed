# NOTE: This test expects an OpenAI Service from ./openai_service.py
# to already be running at localhost:8000 for now. It should be
# updated/migrated to support launching/configuring the HTTP Service
# as a pytest fixture at the start of the test session.
#
#   python3 openai_service.py
#   pytest test_openai_service.py

import json
import pytest
import requests
import subprocess


def test_models():
    """
    Example usage of '/v1/models' route via curl:

        $ curl localhost:8000/v1/models

        {"object":"list","data":[{"id":"dummy_model","object":"object","created":1738186049,"owned_by":"nvidia"}]}
    """
    endpoint = "http://localhost:8000/v1/models"
    r = requests.get(endpoint)
    r.raise_for_status()

    model = "dummy_model"
    models = r.json()
    assert models["object"] == "list"
    assert models["data"][0]["id"] == model
    assert models["data"][0]["created"] > 0
    # TODO: Should each entry of an object type of 'model' instead?
    assert models["data"][0]["object"] == "object"
    assert models["data"][0]["owned_by"] == "nvidia"

def test_chat_no_stream():
    """
    Example usage of '/v1/chat/completions' route via curl:
        $ curl localhost:8000/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{
                "model": "dummy_model",
                "messages": [
                  {
                    "role": "system",
                    "content": "You are a helpful assistant."
                  },
                  {
                    "role": "user",
                    "content": "Hello!"
                  }
                ],
                "stream": false
            }'

        {"id":"chat-6079ad6d-e2be-4209-b596-71901cdfdf0b","choices":[...],"created":1738185790,"model":"dummy_model","object":"chat.completion","usage":null,"system_fingerprint":null}
    """
    messages = [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      },
    ]
    model = "dummy_model"
    payload = {"model": model, "messages": messages, "stream": False}
    headers = {'Content-Type': 'application/json'}
    endpoint = "http://localhost:8000/v1/chat/completions"
    r = requests.post(endpoint, data=json.dumps(payload), headers=headers)
    r.raise_for_status()
    chat_completion = r.json()
    assert chat_completion["model"] == model
    assert chat_completion["object"] == "chat.completion"
    assert chat_completion["choices"][0]["message"]["content"]
    assert chat_completion["choices"][0]["index"] == 0


@pytest.mark.parametrize("streaming", [True, False])
def test_genai_perf(streaming: bool):
    model = "dummy_model"
    url = "localhost:8000"
    request_count = 5
    command = [
        "genai-perf",
        "profile",
        "--model",
        model,
        "--url",
        url,
        "--service-kind",
        "openai",
        "--endpoint-type",
        "chat",
    ]
    if streaming:
        command += ["--streaming"]

    extra_args = [
        "--",
        "--request-count",
        request_count
    ]
    command += extra_args
    command = [str(token) for token in command]

    # Sanity check that genai-perf runs successfully without errors
    # against the OpenAI chat endpoint.
    subprocess.run(command, check=True)
