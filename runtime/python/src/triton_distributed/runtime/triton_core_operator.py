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

import asyncio
import json
import logging
import os
import uuid
from typing import Optional

try:
    import tritonserver
    from tritonserver import DataType as TritonDataType
    from tritonserver import InvalidArgumentError
    from tritonserver import MemoryBuffer as TritonMemoryBuffer
    from tritonserver import MemoryType as TritonMemoryType
    from tritonserver import Server as TritonCore
    from tritonserver import Tensor as TritonTensor
    from tritonserver._api._response import InferenceResponse
except ImportError as e:
    raise ImportError("Triton Core is not installed") from e

from google.protobuf import json_format, text_format
from tritonclient.grpc import model_config_pb2

from triton_distributed.icp.data_plane import DataPlane
from triton_distributed.icp.request_plane import RequestPlane
from triton_distributed.icp.tensor import Tensor
from triton_distributed.runtime.logger import get_logger
from triton_distributed.runtime.operator import Operator
from triton_distributed.runtime.remote_request import RemoteInferenceRequest
from triton_distributed.runtime.remote_response import RemoteInferenceResponse


class TritonCoreOperator(Operator):
    def __init__(
        self,
        name: str,
        version: int,
        request_plane: RequestPlane,
        data_plane: DataPlane,
        parameters: dict,
        repository: Optional[str] = None,
        logger: logging.Logger = get_logger(__name__),
        triton_core: Optional[TritonCore] = None,
    ):
        self._repository = repository
        self._name = name
        self._parameters = parameters
        self._triton_core = triton_core
        self._version = version
        self._logger = logger
        self._request_plane = request_plane
        self._data_plane = data_plane
        self._store_outputs_in_response = self._parameters.get(
            "store_outputs_in_response", False
        )

        if self._triton_core is None:
            raise ValueError("Triton Core required for TritonCoreOperator")

        if not self._repository:
            self._repository = "."

        if repository:
            self._triton_core.register_model_repository(repository)

        parameter_config = self._parameters.get("config", {})
        if "parameters" not in parameter_config:
            parameter_config["parameters"] = {}
        parameter_config["parameters"]["component_id"] = {"string_value": f"{self._request_plane.component_id}"}

        model_config = None

        try:
            model_config_path = os.path.join(
                self._repository, self._name, "config.pbtxt"
            )
            with open(model_config_path, "r") as config_file:
                model_config = text_format.Parse(
                    config_file.read(), model_config_pb2.ModelConfig()
                )
        except Exception:
            pass

        
        parameter_config = json_format.Parse(
                json.dumps(parameter_config), model_config_pb2.ModelConfig()
            )
        if model_config:
            model_config.MergeFrom(parameter_config)
        else:
            model_config = parameter_config
        model_config = {"config": json_format.MessageToJson(model_config)}
        self._triton_core_model = self._triton_core.load(self._name, model_config)

    @staticmethod
    def _triton_tensor(tensor: Tensor) -> TritonTensor:
        return TritonTensor(
            TritonDataType(tensor.data_type),
            tensor.shape,
            TritonMemoryBuffer(
                tensor.memory_buffer.data_ptr,
                TritonMemoryType(tensor.memory_buffer.memory_type),
                tensor.memory_buffer.memory_type_id,
                tensor.memory_buffer.size,
                tensor.memory_buffer.owner,
            ),
        )

    @staticmethod
    def _triton_core_request(
        request: RemoteInferenceRequest, model: tritonserver.Model
    ) -> tritonserver.InferenceRequest:
        triton_core_request = model.create_request()
        if request.request_id is not None:
            triton_core_request.request_id = request.request_id
        if request.priority is not None:
            triton_core_request.priority = request.priority
        if request.timeout is not None:
            triton_core_request.timeout = request.timeout

        if request.correlation_id is not None:
            triton_core_request.correlation_id = request.correlation_id
        TritonCoreOperator._set_inputs(request, triton_core_request)
        TritonCoreOperator._set_parameters(request, triton_core_request)

        return triton_core_request

    @staticmethod
    def _set_inputs(
        request: RemoteInferenceRequest, local_request: tritonserver.InferenceRequest
    ):
        for input_name, remote_tensor in request.inputs.items():
            local_request.inputs[input_name] = TritonCoreOperator._triton_tensor(
                remote_tensor.local_tensor
            )

    @staticmethod
    def _set_parameters(
        request: RemoteInferenceRequest, local_request: tritonserver.InferenceRequest
    ):
        for parameter_name, parameter_value in request.parameters.items():
            local_request.parameters[parameter_name] = parameter_value

    @staticmethod
    def _remote_response(
        triton_core_response: InferenceResponse, store_outputs_in_response: bool = False
    ) -> RemoteInferenceResponse:
        result = RemoteInferenceResponse(
            triton_core_response.model.name,
            triton_core_response.model.version,
            None,
            triton_core_response.request_id,
            final=triton_core_response.final,
        )

        for tensor_name, tensor_value in triton_core_response.outputs.items():
            result.outputs[tensor_name] = tensor_value
            if store_outputs_in_response:
                result.store_outputs_in_response.add(tensor_name)

        for parameter_name, parameter_value in triton_core_response.parameters.items():
            result.parameters[parameter_name] = parameter_value

        result.error = triton_core_response.error
        return result

    async def execute(self, requests: list[RemoteInferenceRequest]) -> None:
        request_id_map = {}
        response_queue: asyncio.Queue = asyncio.Queue()
        for request in requests:
            self._logger.debug("\n\nReceived request: \n\n%s\n\n", request)
            try:
                triton_core_request = TritonCoreOperator._triton_core_request(
                    request, self._triton_core_model
                )
            except Exception as e:
                message = f"Can't resolve tensors for request, ignoring request,{e}"
                self._logger.error(message)
                await request.response_sender().send(
                    error=InvalidArgumentError(message), final=True
                )
                continue

            request_id = str(uuid.uuid1())
            original_id = None
            if triton_core_request.request_id is not None:
                original_id = triton_core_request.request_id
            triton_core_request.request_id = request_id
            request_id_map[request_id] = (request.response_sender(), original_id)

            triton_core_request.response_queue = response_queue
            self._triton_core_model.async_infer(triton_core_request)

        while request_id_map:
            triton_core_response = await response_queue.get()

            remote_response = TritonCoreOperator._remote_response(
                triton_core_response, self._store_outputs_in_response
            )

            response_sender, original_id = request_id_map[
                triton_core_response.request_id
            ]
            remote_response.request_id = original_id

            if triton_core_response.final:
                del request_id_map[triton_core_response.request_id]
            self._logger.debug("\n\nSending response\n\n%s\n\n", remote_response)
            await response_sender.send(remote_response)
