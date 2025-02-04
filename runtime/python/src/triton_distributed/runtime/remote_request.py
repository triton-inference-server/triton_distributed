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

"""Class for sending inference requests to Triton Inference Server Models"""

from __future__ import annotations

import asyncio
import queue
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from triton_distributed.icp.data_plane import DataPlane
from triton_distributed.icp.protos.icp_pb2 import ModelInferRequest
from triton_distributed.icp.request_plane import RequestPlane, get_icp_component_id
from triton_distributed.icp.tensor import Tensor
from triton_distributed.runtime.remote_response import RemoteInferenceResponse
from triton_distributed.runtime.remote_tensor import RemoteTensor


@dataclass
class RemoteInferenceRequest:
    model_name: str
    model_version: int
    data_plane: DataPlane
    component_id: Optional[uuid.UUID] = None
    _request_plane: Optional[RequestPlane] = None
    _model_infer_request: Optional[ModelInferRequest] = None
    request_id: Optional[str] = None
    correlation_id: Optional[int | str] = None
    priority: Optional[int] = None
    timeout: Optional[int] = None
    inputs: dict[str, RemoteTensor | Any] = field(default_factory=dict)
    store_inputs_in_request: set[str] = field(default_factory=set)
    parameters: dict[str, str | int | bool | float] = field(default_factory=dict)
    response_queue: Optional[queue.SimpleQueue | asyncio.Queue] = None

    def _set_model_infer_request_inputs(
        self,
        remote_request: ModelInferRequest,
    ):
        for name, value in self.inputs.items():
            if not isinstance(value, RemoteTensor):
                if not isinstance(value, Tensor):
                    tensor = Tensor._from_object(value)
                else:
                    tensor = value
                use_tensor_contents = name in self.store_inputs_in_request
                remote_input = self.data_plane.put_input_tensor(
                    tensor, use_tensor_contents=use_tensor_contents
                )
            else:
                remote_input = self.data_plane.create_input_tensor_reference(
                    value.remote_tensor
                )

            remote_input.name = name
            remote_request.inputs.append(remote_input)

    def _set_model_infer_request_parameters(self, remote_request: ModelInferRequest):
        for key, value in self.parameters.items():
            remote_value = remote_request.parameters[key]
            if isinstance(value, str):
                remote_value.string_param = value
            elif isinstance(value, int):
                remote_value.int64_param = value
            elif isinstance(value, float):
                remote_value.double_param = value
            elif isinstance(value, bool):
                remote_value.bool_param = value
            else:
                raise ValueError(f"Invalid parameter type: {type(value)}")

    @staticmethod
    def _set_parameters_from_model_infer_request(
        result: RemoteInferenceRequest,
        inference_request: ModelInferRequest,
    ):
        for name, value in inference_request.parameters.items():
            if value.HasField("bool_param"):
                result.parameters[name] = value.bool_param
            elif value.HasField("int64_param"):
                result.parameters[name] = value.int64_param
            elif value.HasField("double_param"):
                result.parameters[name] = value.double_param
            elif value.HasField("string_param"):
                result.parameters[name] = value.string_param

    @staticmethod
    def _set_inputs_from_model_infer_request(
        result: RemoteInferenceRequest,
        inference_request: ModelInferRequest,
    ):
        for remote_input in inference_request.inputs:
            result.inputs[remote_input.name] = RemoteTensor(
                remote_input, result.data_plane
            )

    def cancel(self):
        raise NotImplementedError("Cancel not implemented")

    def response_sender(self):
        if self._request_plane is None or self._model_infer_request is None:
            raise ValueError(
                "Response only valid for requests received from request plane"
            )
        return RemoteResponseSender(
            self._model_infer_request, self._request_plane, self.data_plane
        )

    @staticmethod
    def from_model_infer_request(
        request: ModelInferRequest, data_plane: DataPlane, request_plane: RequestPlane
    ) -> RemoteInferenceRequest:
        result = RemoteInferenceRequest(
            request.model_name,
            int(request.model_version),
            data_plane,
            _request_plane=request_plane,
            _model_infer_request=request,
        )
        if request.id is not None:
            result.request_id = request.id
        result.component_id = get_icp_component_id(request)
        if "sequence_id" in request.parameters:
            if request.parameters["sequence_id"].HasField("string_param"):
                result.correlation_id = request.parameters["sequence_id"].string_param
            else:
                result.correlation_id = request.parameters["sequence_id"].int64_param
        if "priority" in request.parameters:
            result.priority = request.parameters["priority"].uint64_param
        if "timeout" in request.parameters:
            result.timeout = request.parameters["timeout"].uint64_param
        RemoteInferenceRequest._set_inputs_from_model_infer_request(result, request)
        RemoteInferenceRequest._set_parameters_from_model_infer_request(result, request)
        return result

    def to_model_infer_request(self) -> ModelInferRequest:
        remote_request = ModelInferRequest()
        remote_request.model_name = self.model_name
        remote_request.model_version = str(self.model_version)
        if self.request_id is not None:
            remote_request.id = self.request_id
        if self.priority is not None:
            remote_request.parameters["priority"].uint64_param = self.priority
        if self.timeout is not None:
            remote_request.parameters["timeout"].uint64_param = self.timeout
        if self.correlation_id is not None:
            if isinstance(self.correlation_id, str):
                remote_request.parameters[
                    "sequence_id"
                ].string_param = self.correlation_id
            else:
                remote_request.parameters[
                    "sequence_id"
                ].int64_param = self.correlation_id
        self._set_model_infer_request_inputs(remote_request)
        self._set_model_infer_request_parameters(remote_request)

        return remote_request


class RemoteResponseSender:
    response_counts: Counter = Counter()

    def __init__(
        self,
        model_infer_request: ModelInferRequest,
        request_plane: RequestPlane,
        data_plane: DataPlane,
    ):
        self._model_infer_request = model_infer_request
        self._request_plane = request_plane
        self._data_plane = data_plane

    def create_response(self, **kwargs) -> RemoteInferenceResponse:
        if "model_name" in kwargs:
            kwargs.pop("model_name")
        if "model_version" in kwargs:
            kwargs.pop("model_version")
        if "request_id" in kwargs:
            kwargs.pop("request_id")

        return RemoteInferenceResponse(
            model_name=self._model_infer_request.model_name,
            model_version=self._model_infer_request.model_version,
            request_id=self._model_infer_request.id,
            **kwargs,
        )

    async def send(
        self, inference_response: Optional[RemoteInferenceResponse] = None, **kwargs
    ) -> None:
        if inference_response is None:
            inference_response = RemoteInferenceResponse(
                model_name=self._model_infer_request.model_name,
                model_version=self._model_infer_request.model_version,
                request_id=self._model_infer_request.id,
                **kwargs,
            )
        await self._request_plane.post_response(
            self._model_infer_request,
            inference_response.to_model_infer_response(self._data_plane),
        )
        if inference_response.final:
            RemoteResponseSender.response_counts[
                (
                    self._model_infer_request.model_name,
                    self._model_infer_request.model_version,
                )
            ] += 1
