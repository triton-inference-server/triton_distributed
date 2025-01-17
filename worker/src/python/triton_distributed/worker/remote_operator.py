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

"""Class for interacting with Triton Inference Server Models"""

import asyncio
import uuid
from typing import Optional

from triton_distributed.icp.data_plane import DataPlane
from triton_distributed.icp.request_plane import RequestPlane
from triton_distributed.worker.remote_request import RemoteInferenceRequest
from triton_distributed.worker.remote_response import AsyncRemoteResponseIterator
from tritonserver import InvalidArgumentError


class RemoteOperator:
    def __init__(
        self,
        operator: str | tuple[str, int],
        request_plane: RequestPlane,
        data_plane: DataPlane,
        component_id: Optional[uuid.UUID] = None,
    ):
        if isinstance(operator, str):
            self.name = operator
            self.version = 1
        else:
            self.name = operator[0]
            self.version = operator[1]
        self._request_plane = request_plane
        self._data_plane = data_plane
        self.component_id = component_id

    @property
    def data_plane(self):
        return self._data_plane

    def create_request(self, **kwargs) -> RemoteInferenceRequest:
        if "model_name" in kwargs:
            kwargs.pop("model_name")
        if "model_version" in kwargs:
            kwargs.pop("model_version")
        if "data_plane" in kwargs:
            kwargs.pop("data_plane")
        if "_request_plane" in kwargs:
            kwargs.pop("_request_plane")
        if "_model_infer_request" in kwargs:
            kwargs.pop("_model_infer_request")

        return RemoteInferenceRequest(
            model_name=self.name,
            model_version=self.version,
            data_plane=self._data_plane,
            _request_plane=None,
            _model_infer_request=None,
            **kwargs,
        )

    async def async_infer(
        self,
        inference_request: Optional[RemoteInferenceRequest] = None,
        raise_on_error: bool = True,
        **kwargs,
    ) -> AsyncRemoteResponseIterator:
        if inference_request is None:
            inference_request = RemoteInferenceRequest(
                model_name=self.name,
                model_version=self.version,
                data_plane=self.data_plane,
                _request_plane=None,
                _model_infer_request=None,
                **kwargs,
            )
        else:
            inference_request.model_name = self.name
            inference_request.model_version = self.version
            if inference_request.data_plane != self.data_plane:
                raise InvalidArgumentError(
                    "Data plane mismatch between remote request and remote operator: \n\n Operator: {self.data_plane} \n\n Request: {inference_request.data_plane}"
                )

        if (inference_request.response_queue is not None) and (
            not isinstance(inference_request.response_queue, asyncio.Queue)
        ):
            raise InvalidArgumentError(
                "asyncio.Queue must be used for async response iterator"
            )
        response_iterator = AsyncRemoteResponseIterator(
            self._data_plane,
            inference_request,
            inference_request.response_queue,
            raise_on_error,
        )

        remote_inference_request = inference_request.to_model_infer_request()

        await self._request_plane.post_request(
            remote_inference_request,
            response_handler=response_iterator._response_handler,
            component_id=self.component_id,
        )

        return response_iterator
