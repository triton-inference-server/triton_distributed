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

"""Class for receiving inference responses to Triton Inference Server Models"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from triton_distributed.icp.data_plane import DataPlane
from triton_distributed.icp.protos.icp_pb2 import ModelInferResponse

if TYPE_CHECKING:
    from triton_distributed.worker.remote_request import RemoteInferenceRequest

import uuid

from tritonserver import InternalError, Tensor, TritonError
from tritonserver._api._response import InferenceResponse

from triton_distributed.icp.request_plane import (
    get_icp_component_id,
    get_icp_final_response,
    get_icp_response_error,
    set_icp_final_response,
    set_icp_response_error,
)
from triton_distributed.worker.logger import get_logger
from triton_distributed.worker.remote_tensor import RemoteTensor

logger = get_logger(__name__)


class AsyncRemoteResponseIterator:

    """Asyncio compatible response iterator

    Response iterators are returned from model inference methods and
    allow users to process inference responses in the order they were
    received for a request.

    """

    def __init__(
        self,
        data_plane: DataPlane,
        request: RemoteInferenceRequest,
        user_queue: Optional[asyncio.Queue] = None,
        raise_on_error: bool = False,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        """Initialize AsyncResponseIterator

        AsyncResponseIterator objects are obtained from Model inference
        methods and not instantiated directly. See `Model` documentation.

        Parameters
        ----------
        model : Model
            model associated with inference request
        request : TRITONSERVER_InferenceRequest
            Underlying C binding TRITONSERVER_InferenceRequest
            object. Private.
        user_queue : Optional[asyncio.Queue]
            Optional user queue for responses in addition to internal
            iterator queue.
        raise_on_error : bool
            if True response errors will be raised as exceptions.
        loop : Optional[asyncio.AbstractEventLoop]
            asyncio loop object

        """

        if loop is None:
            loop = asyncio.get_running_loop()
        self._loop = loop
        self._queue: asyncio.Queue = asyncio.Queue()
        self._user_queue = user_queue
        self._complete = False
        self._request = request
        self._data_plane = data_plane
        self._raise_on_error = raise_on_error

    def __aiter__(self) -> AsyncRemoteResponseIterator:
        """Return async iterator. For use with async for loops.

        Returns
        -------
        AsyncResponseIterator

        Examples
        --------

        responses = server.model("test").async_infer(inputs={"fp16_input":numpy.array([[1]],dtype=numpy.float16)})
        async for response in responses:
            print(nummpy.from_dlpack(response.outputs["fp16_output"]))


        """

        return self

    async def __anext__(self):
        """Returns the next response received for a request

        Returns the next response received for a request as an
        awaitable object.

        Raises
        ------
        response.error
            If raise_on_error is set to True, response errors are
            raised as exceptions
        StopAsyncIteration
            Indicates all responses for a request have been received.
            Final responses may or may not include outputs and must be
            checked.

        """

        if self._complete:
            raise StopAsyncIteration

        response = await self._queue.get()
        self._complete = response.final
        if response.error is not None and self._raise_on_error:
            raise response.error
        return response

    def cancel(self) -> None:
        """Cancel an inflight request

        Cancels an in-flight request. Cancellation is handled on a
        best effort basis and may not prevent execution of a request
        if it is already started or completed.

        See c:func:`TRITONSERVER_ServerInferenceRequestCancel`

        Examples
        --------

        responses = server.model("test").infer(inputs={"text_input":["hello"]})

        responses.cancel()

        """

        if self._request is not None:
            self._request.cancel()

    def _response_handler(self, response: ModelInferResponse):
        try:
            if self._request is None:
                raise InternalError("Response received after final response flag")

            final = False

            if response is None or get_icp_final_response(response):
                final = True

            remote_response = RemoteInferenceResponse.from_model_infer_response(
                self._request, response, self._data_plane, final
            )
            asyncio.run_coroutine_threadsafe(
                self._queue.put(remote_response), self._loop
            )
            if self._user_queue is not None:
                asyncio.run_coroutine_threadsafe(
                    self._user_queue.put(remote_response), self._loop
                )
            if final:
                del self._request
                self._request = None

        except Exception as e:
            message = f"Catastrophic failure in response callback: {e}"
            logger.exception(message)
            # catastrophic failure
            raise e from None


@dataclass
class RemoteInferenceResponse:
    """Dataclass representing an inference response.

    Inference response objects are returned from response iterators
    which are in turn returned from model inference methods. They
    contain output data, output parameters, any potential errors
    reported and a flag to indicate if the response is the final one
    for a request.

    See c:func:`TRITONSERVER_InferenceResponse` for more details

    Parameters
    ----------
    model : Model
        Model instance associated with the response.
    request_id : Optional[str], default None
        Unique identifier for the inference request (if provided)
    parameters : dict[str, str | int | bool], default {}
        Additional parameters associated with the response.
    outputs : dict [str, Tensor], default {}
        Output tensors for the inference.
    error : Optional[TritonError], default None
        Error (if any) that occurred in the processing of the request.
    classification_label : Optional[str], default None
        Classification label associated with the inference. Not currently supported.
    final : bool, default False
        Flag indicating if the response is final

    """

    model_name: str
    model_version: int
    component_id: Optional[uuid.UUID] = None
    request_id: Optional[str] = None
    parameters: dict[str, str | int | bool] = field(default_factory=dict)
    outputs: dict[str, RemoteTensor | Tensor] = field(default_factory=dict)
    store_outputs_in_response: set[str] = field(default_factory=set)
    error: Optional[TritonError] = None
    classification_label: Optional[str] = None
    final: bool = False

    def _set_parameters_from_model_infer_response_parameters(
        self, response: ModelInferResponse
    ):
        for name, value in response.parameters.items():
            if value.HasField("string_param"):
                self.parameters[name] = value.string_param
            elif value.HasField("int64_param"):
                self.parameters[name] = value.int64_param
            elif value.HasField("double_param"):
                self.parameters[name] = value.double_param
            elif value.HasField("bool_param"):
                self.parameters[name] = value.bool_param

    def _set_model_infer_response_outputs(
        self, response: ModelInferResponse, data_plane: DataPlane
    ):
        for name, value in self.outputs.items():
            if not isinstance(value, RemoteTensor):
                if not isinstance(value, Tensor):
                    tensor = Tensor._from_object(value)
                else:
                    tensor = value
                use_tensor_contents = name in self.store_outputs_in_response
                remote_output = data_plane.put_output_tensor(
                    tensor, use_tensor_contents=use_tensor_contents
                )
            else:
                remote_output = data_plane.create_output_tensor_reference(
                    value.remote_tensor
                )
            remote_output.name = name
            response.outputs.append(remote_output)

    def _set_model_infer_response_parameters(self, response: ModelInferResponse):
        for key, value in self.parameters.items():
            remote_value = response.parameters[key]
            if isinstance(value, str):
                remote_value.string_param = value
            elif isinstance(value, int):
                remote_value.int64_param = value
            elif isinstance(value, float):
                remote_value.double_param = value
            elif isinstance(value, bool):
                remote_value.bool_param = value

    def to_model_infer_response(self, data_plane: DataPlane):
        remote_response = ModelInferResponse()
        remote_response.model_name = self.model_name
        remote_response.model_version = str(self.model_version)
        if self.request_id:
            remote_response.id = self.request_id
        if self.error:
            set_icp_response_error(remote_response, self.error)
        if self.final:
            set_icp_final_response(remote_response, self.final)
        self._set_model_infer_response_parameters(remote_response)
        self._set_model_infer_response_outputs(remote_response, data_plane)
        return remote_response

    @staticmethod
    def from_local_response(
        local_response: InferenceResponse, store_outputs_in_response: bool = False
    ):
        result = RemoteInferenceResponse(
            local_response.model.name,
            local_response.model.version,
            None,
            local_response.request_id,
            final=local_response.final,
        )

        for tensor_name, tensor_value in local_response.outputs.items():
            result.outputs[tensor_name] = tensor_value
            if store_outputs_in_response:
                result.store_outputs_in_response.add(tensor_name)

        for parameter_name, parameter_value in local_response.parameters.items():
            result.parameters[parameter_name] = parameter_value

        result.error = local_response.error
        return result

    @staticmethod
    def from_model_infer_response(
        request: RemoteInferenceRequest,
        response: ModelInferResponse,
        data_plane: DataPlane,
        final_response: bool,
    ) -> RemoteInferenceResponse:
        result = RemoteInferenceResponse(
            request.model_name,
            request.model_version,
            None,
            request.request_id,
            final=final_response,
        )
        if response is None:
            return result
        result.request_id = response.id
        result.component_id = get_icp_component_id(response)
        outputs = {}
        for output in response.outputs:
            outputs[output.name] = RemoteTensor(output, data_plane)
        result.outputs = outputs

        result._set_parameters_from_model_infer_response_parameters(response)

        result.error = get_icp_response_error(response)

        return result
