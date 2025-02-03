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

"""Abstract Class for interacting with the Triton Inference Serving Platform Inter-Component Protocol Control Plane"""

import abc
import uuid
from typing import AsyncIterator, Awaitable, Callable, Optional

from tritonserver import TritonError

from triton_distributed.icp.protos.icp_pb2 import ModelInferRequest, ModelInferResponse

ICP_REQUEST_ID = "icp_request_id"
ICP_FINAL_RESPONSE = "icp_final_response"
ICP_RESPONSE_FROM_URI = "icp_response_from_uri"
ICP_COMPONENT_ID = "icp_component_id"
ICP_RESPONSE_TO_URI = "icp_response_to_uri"
ICP_REQUEST_TO_URI = "icp_request_to_uri"
ICP_REQUEST_CANCELLED = "icp_request_cancelled"
ICP_ERROR = "icp_response_error"


def get_icp_request_id(
    message: ModelInferRequest | ModelInferResponse,
) -> uuid.UUID | None:
    if ICP_REQUEST_ID not in message.parameters:
        return None
    return uuid.UUID(message.parameters[ICP_REQUEST_ID].string_param)


def set_icp_request_id(
    message: ModelInferRequest | ModelInferResponse, value: uuid.UUID
) -> None:
    message.parameters[ICP_REQUEST_ID].string_param = str(value)


def get_icp_response_error(message: ModelInferResponse) -> TritonError | None:
    if ICP_ERROR not in message.parameters:
        return None
    return TritonError(message.parameters[ICP_ERROR].string_param)


def set_icp_response_error(message: ModelInferResponse, value: TritonError) -> None:
    message.parameters[ICP_ERROR].string_param = str(value)


def get_icp_final_response(
    message: ModelInferResponse,
) -> bool:
    if ICP_FINAL_RESPONSE not in message.parameters:
        return False
    return message.parameters[ICP_FINAL_RESPONSE].bool_param


def set_icp_final_response(message: ModelInferResponse, value: bool) -> None:
    message.parameters[ICP_FINAL_RESPONSE].bool_param = value


def get_icp_response_to_uri(message: ModelInferRequest) -> str | None:
    if ICP_RESPONSE_TO_URI not in message.parameters:
        return None
    return message.parameters[ICP_RESPONSE_TO_URI].string_param


def get_icp_component_id(
    message: ModelInferRequest | ModelInferResponse,
) -> uuid.UUID | None:
    if ICP_COMPONENT_ID not in message.parameters:
        return None
    return uuid.UUID(message.parameters[ICP_COMPONENT_ID].string_param)


def set_icp_component_id(
    message: ModelInferRequest | ModelInferResponse, value: uuid.UUID
) -> None:
    message.parameters[ICP_COMPONENT_ID].string_param = str(value)


def set_icp_response_to_uri(message: ModelInferRequest, value: str) -> None:
    message.parameters[ICP_RESPONSE_TO_URI].string_param = value


def get_icp_request_to_uri(message: ModelInferRequest) -> str | None:
    if ICP_REQUEST_TO_URI not in message.parameters:
        return None
    return message.parameters[ICP_REQUEST_TO_URI].string_param


def set_icp_request_to_uri(message: ModelInferRequest, value: str) -> None:
    message.parameters[ICP_REQUEST_TO_URI].string_param = value


def get_icp_response_from_uri(message: ModelInferResponse) -> str | None:
    if ICP_RESPONSE_FROM_URI not in message.parameters:
        return None
    return message.parameters[ICP_RESPONSE_FROM_URI].string_param


def set_icp_response_from_uri(message: ModelInferResponse, value: str) -> None:
    message.parameters[ICP_RESPONSE_FROM_URI].string_param = value


class RequestPlane(abc.ABC):
    @property
    @abc.abstractmethod
    def component_id(self) -> uuid.UUID:
        pass

    @abc.abstractmethod
    async def connect(self) -> None:
        pass

    @abc.abstractmethod
    async def pull_requests(
        self,
        model_name: str,
        model_version: str,
        number_requests: int = 1,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[ModelInferRequest]:
        pass

    @abc.abstractmethod
    async def post_response(
        self,
        request: ModelInferRequest,
        responses: AsyncIterator[ModelInferResponse] | ModelInferResponse,
    ) -> None:
        pass

    @abc.abstractmethod
    async def post_request(
        self,
        request: ModelInferRequest,
        *,
        component_id: Optional[uuid.UUID] = None,
        response_iterator: bool = True,
        response_handler: Optional[
            Callable[[ModelInferResponse], None | Awaitable[None]]
        ] = None,
    ) -> AsyncIterator[ModelInferResponse]:
        pass
