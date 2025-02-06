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


import asyncio
import os
from typing import Any, AsyncIterator, Dict, List

import uvloop
from triton_distributed_rs import DistributedRuntime, triton_worker
from tritonserver import ModelControlMode
from tritonserver import Server as TritonCore
from tritonserver import Tensor
from tritonserver._api._response import InferenceResponse


# FIXME: Can this be more generic to arbitrary Triton models?
class RequestHandler:
    """
    Request handler for the generate endpoint that uses TritonCore
    to process text generation requests.
    """

    def __init__(
        self,
        model_name: str = "mock_llm",
        model_repository: str = os.path.join(os.path.dirname(__file__), "models"),
    ):
        self.model_name: str = model_name

        # Initialize TritonCore
        self._triton_core = TritonCore(
            model_repository=model_repository,
            log_info=True,
            log_error=True,
            model_control_mode=ModelControlMode.EXPLICIT,
        ).start(wait_until_ready=True)

        # Load only the requested model
        self._triton_core.load(self.model_name)

        # Get a handle to the requested model for re-use
        self._model = self._triton_core.model(self.model_name)

        # Validate the model has the expected inputs and outputs
        self._validate_model_config()

        print(f"Model {self.model_name} ready to generate")

    # FIXME: Can this be more generic to arbitrary Triton models?
    def _validate_model_config(self):
        self._model_metadata = self._model.metadata()
        self._inputs = self._model_metadata["inputs"]
        self._outputs = self._model_metadata["outputs"]

        # Validate the model has the expected input and output
        self._expected_input_name: str = "text_input"
        if not any(
            input["name"] == self._expected_input_name for input in self._inputs
        ):
            raise ValueError(
                f"Model {self.model_name} does not have an input named {self._expected_input_name}"
            )

        self._expected_output_name: str = "text_output"
        if not any(
            output["name"] == self._expected_output_name for output in self._outputs
        ):
            raise ValueError(
                f"Model {self.model_name} does not have an output named {self._expected_output_name}"
            )

    async def generate(self, request: str) -> AsyncIterator[str]:
        # FIXME: Iron out request type/schema
        if not isinstance(request, str):
            raise ValueError("Request must be a string")

        try:
            print(f"Processing generation request: {request}")
            text_input: List[str] = [request]
            stream: List[bool] = [True]

            triton_core_inputs: Dict[str, Any] = {
                "text_input": text_input,
                "stream": stream,
            }
            responses: AsyncIterator[InferenceResponse] = self._model.async_infer(
                inputs=triton_core_inputs
            )

            async for response in responses:
                print(f"Received response: {response}")
                text_output: str = ""

                text_output_tensor: Tensor | None = response.outputs.get("text_output")
                if text_output_tensor:
                    text_output = text_output_tensor.to_string_array()[0]

                if response.error:
                    raise response.error

                yield text_output

        except Exception as e:
            print(f"Error processing request: {e}")
            raise


@triton_worker()
async def worker(runtime: DistributedRuntime):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    namespace: str = "triton_core_example"
    component = runtime.namespace(namespace).component("backend")
    await component.create_service()

    endpoint = component.endpoint("generate")
    print("Started server instance")
    await endpoint.serve_endpoint(RequestHandler().generate)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
