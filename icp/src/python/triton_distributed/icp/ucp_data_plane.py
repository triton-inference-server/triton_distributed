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
import contextlib
import logging
import threading
import uuid
from enum import IntEnum, auto
from functools import cached_property
from typing import Dict, Optional, Tuple
from urllib.parse import urlsplit

import cupy
import numpy
import tritonserver
import ucp
from cupy_backends.cuda.api.runtime import CUDARuntimeError
from triton_distributed.icp.data_plane import (
    DataPlane,
    DataPlaneError,
    get_icp_data_type,
    get_icp_memory_type,
    get_icp_memory_type_id,
    get_icp_shape,
    get_icp_tensor_contents,
    get_icp_tensor_size,
    get_icp_tensor_uri,
    set_icp_data_type,
    set_icp_memory_type,
    set_icp_memory_type_id,
    set_icp_shape,
    set_icp_tensor_contents,
    set_icp_tensor_size,
    set_icp_tensor_uri,
)
from triton_distributed.icp.protos.icp_pb2 import ModelInferRequest, ModelInferResponse
from tritonserver import InvalidArgumentError, MemoryBuffer, MemoryType, Tensor

LOGGER = logging.getLogger(__name__)


class UCP_DATA_PLANE_COMMANDS(IntEnum):
    GET = auto()
    CREATE_REFERENCE = auto()
    RELEASE = auto()


# UCP has deadlocks when created multiple instances in a single process
# Create a singleton

_ucp_data_plane_singleton = None


def UcpDataPlane(
    hostname: Optional[str] = None, port: int = 0, keep_endpoints_open: bool = False
):
    global _ucp_data_plane_singleton
    if _ucp_data_plane_singleton is None:
        _ucp_data_plane_singleton = _UcpDataPlane(hostname, port, keep_endpoints_open)
    return _ucp_data_plane_singleton


class _UcpDataPlane(DataPlane):
    def __init__(
        self,
        hostname: Optional[str] = None,
        port: int = 0,
        keep_endpoints_open: bool = False,
    ) -> None:
        self._tensor_store: Dict[uuid.UUID, Tensor] = {}
        self._id_size = len(uuid.uuid1().bytes)
        self._port = port
        self._hostname = hostname or ucp.get_address()
        self._event_loop_thread = threading.Thread(
            target=self._run_event_loop, daemon=True
        )
        self._start_event = threading.Event()
        self._listener: Optional[ucp.Listener] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._closed = False
        self._keep_endpoints_open = keep_endpoints_open
        self._endpoints: Dict[Tuple[str, int], ucp.Endpoint] = {}
        LOGGER.debug(
            "Creating UCP data plane with keep_endpoints_open=%s", keep_endpoints_open
        )

    @cached_property
    def _cuda_is_available(self):
        # Note: cuda.is_avalailable initializes cuda
        #       and can't be called when forking subprocesses
        #       care should be taken to only call it within
        #       subprocesses or use 'spawn'
        try:
            return cupy.cuda.is_available()
        except CUDARuntimeError:
            return False

    @property
    def hostname(self) -> str:
        return self._hostname

    @property
    def port(self) -> int:
        return self._port

    def connect(self) -> None:
        if self._event_loop is None:
            self._event_loop_thread.start()
            self._start_event.wait()
            if self._listener is None or self._listener.closed():
                raise DataPlaneError("Unable to start data plane")

    async def _close(self, wait_for_release=0):
        self._closed = True
        if self._listener is not None:
            if wait_for_release:
                while self._tensor_store and wait_for_release:
                    await asyncio.sleep(1)
                    wait_for_release -= 1
            self._listener.close()
            self._listener = None

    def close(self, wait_for_release=0):
        if self._event_loop is None:
            return
        if self._event_loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(
            self._close(wait_for_release),
            self._event_loop,
        ).result()

    def __del__(self):
        self.close()

    def _run_event_loop(self):
        asyncio.run(self._serve())

    async def _serve(self):
        self._event_loop = asyncio.get_running_loop()
        try:
            self._listener = ucp.create_listener(self._send_receive, self._port)
            self._port = self._listener.port
            self._start_event.set()
        except Exception:
            self._listener = None
            self._start_event.set()

        while self._listener is not None and not self._listener.closed():
            await asyncio.sleep(1)

    async def _send_receive(self, ep):
        while True:
            tensor_id_bytes = numpy.empty(self._id_size, dtype="u1")
            await ep.recv(tensor_id_bytes)
            tensor_id = uuid.UUID(bytes=tensor_id_bytes.tobytes())
            command = numpy.empty(1, dtype="u1")

            await ep.recv(command)
            if command == UCP_DATA_PLANE_COMMANDS.GET:
                if tensor_id in self._tensor_store:
                    tensor = self._tensor_store[tensor_id]
                    array_module = numpy
                    if tensor.memory_type == tritonserver.MemoryType.CPU:
                        array_module = numpy
                        device_manager = contextlib.nullcontext()
                    elif tensor.memory_type == tritonserver.MemoryType.GPU:
                        array_module = cupy
                        device_manager = cupy.cuda.Device(
                            tensor.memory_buffer.memory_type_id
                        )
                    else:
                        raise InvalidArgumentError(
                            f"Invalid Memory Type {tensor.memory_type}"
                        )
                    with device_manager:
                        if tensor.data_type == tritonserver.DataType.BYTES:
                            array = tensor.memory_buffer.owner
                        else:
                            array = array_module.from_dlpack(tensor)
                        await ep.send(array)
            elif command == UCP_DATA_PLANE_COMMANDS.CREATE_REFERENCE:
                if tensor_id in self._tensor_store:
                    reference_tensor_id = uuid.uuid1()
                    self._tensor_store[reference_tensor_id] = self._tensor_store[
                        tensor_id
                    ]
                    await ep.send(numpy.array(reference_tensor_id.bytes))
            elif command == UCP_DATA_PLANE_COMMANDS.RELEASE:
                if tensor_id in self._tensor_store:
                    del self._tensor_store[tensor_id]
                    await ep.send(numpy.array(tensor_id.bytes))

            if not self._keep_endpoints_open:
                break

        await ep.close()

    def _put_tensor(
        self,
        result: ModelInferRequest.InferInputTensor
        | ModelInferResponse.InferOutputTensor,
        tensor: Tensor,
        tensor_id: Optional[uuid.UUID] = None,
        use_tensor_contents: bool = False,
    ):
        if self._closed:
            raise DataPlaneError("Adding tensor after close")
        set_icp_data_type(result, tensor.data_type)
        set_icp_shape(result, tensor.shape)

        if use_tensor_contents:
            set_icp_tensor_contents(result, tensor)
        else:
            if tensor_id is None:
                tensor_id = uuid.uuid1()
            self._tensor_store[tensor_id] = tensor

            tensor_uri = f"ucp://{self._hostname}:{self._port}/{tensor_id}"

            set_icp_tensor_uri(result, tensor_uri)
            set_icp_memory_type(result, tensor.memory_buffer.memory_type)
            set_icp_memory_type_id(result, tensor.memory_buffer.memory_type_id)
            set_icp_tensor_size(result, tensor.size)

    def put_input_tensor(
        self,
        tensor: Tensor,
        tensor_id: Optional[uuid.UUID] = None,
        use_tensor_contents: bool = False,
    ) -> ModelInferRequest.InferInputTensor:
        """Put an input tensor into the data plane or within
        returned ModelInferRequest.InferInputTensor itself.

        Args:
            tensor: The tensor to put.
            tensor_id: The id of the tensor to put.
                If not provided, a new id will be generated.
            use_tensor_contents: when True, tensor data will be
                added directly to ModelInferRequest.InferInputTensor
                contents field; otherwise tensor data will be sent
                separately on the data plane.

        Returns:
            ModelInferRequest.InferInputTensor object.

        """
        result = ModelInferRequest.InferInputTensor()
        self._put_tensor(
            result,
            tensor,
            tensor_id=tensor_id,
            use_tensor_contents=use_tensor_contents,
        )
        return result

    def put_output_tensor(
        self,
        tensor: Tensor,
        tensor_id: Optional[uuid.UUID] = None,
        use_tensor_contents: bool = False,
    ) -> ModelInferResponse.InferOutputTensor:
        """Put an output tensor into the data plane or within
        returned ModelInferResponse.InferOutputTensor itself.

        Args:
            tensor: The tensor to put.
            tensor_id: The id of the tensor to put.
                If not provided, a new id will be generated.
            use_tensor_contents: when True, tensor data will be
                added directly to ModelInferResponse.InferInputTensor
                contents field; otherwise tensor data will be sent
                separately on the data plane.

        Returns:
            ModelInferResponse.InferOutputTensor object.
        """
        result = ModelInferResponse.InferOutputTensor()
        self._put_tensor(
            result,
            tensor,
            tensor_id=tensor_id,
            use_tensor_contents=use_tensor_contents,
        )
        return result

    def _split_tensor_uri(
        self,
        remote_tensor: ModelInferRequest.InferInputTensor
        | ModelInferResponse.InferOutputTensor,
    ) -> tuple[uuid.UUID, str, int]:
        tensor_uri = get_icp_tensor_uri(remote_tensor)

        split_uri = urlsplit(tensor_uri)
        path = str(split_uri.path).replace("/", "")
        tensor_id = uuid.UUID(path)
        host = split_uri.hostname
        port = split_uri.port

        if host is None or not isinstance(host, str):
            raise DataPlaneError(f"Invalid host {host}")

        if port is None:
            raise DataPlaneError(f"Invalid Port {port}")

        return tensor_id, host, port

    async def _get_remote_tensor(
        self,
        remote_tensor: ModelInferRequest.InferInputTensor
        | ModelInferResponse.InferOutputTensor,
        requested_memory_type: Optional[MemoryType],
        requested_memory_type_id: Optional[int],
    ) -> Tensor:
        tensor_contents = get_icp_tensor_contents(remote_tensor)
        if tensor_contents is not None:
            return tensor_contents

        tensor_size = get_icp_tensor_size(remote_tensor)
        memory_type = get_icp_memory_type(remote_tensor)
        data_type = get_icp_data_type(remote_tensor)
        shape = get_icp_shape(remote_tensor)
        tensor_id, host, port = self._split_tensor_uri(remote_tensor)
        storage = None

        if tensor_size is None:
            raise DataPlaneError("tensor size can not be none")

        if requested_memory_type is not None:
            memory_type = requested_memory_type

        if memory_type == tritonserver.MemoryType.GPU and self._cuda_is_available:
            array_module = cupy
            if requested_memory_type_id is not None:
                device_manager = cupy.cuda.Device(requested_memory_type_id)
            else:
                device_manager = contextlib.nullcontext()
        else:
            array_module = numpy
            device_manager = contextlib.nullcontext()

        with device_manager:
            storage = array_module.empty(tensor_size, dtype="u1")

            try:
                endpoint = await self._create_endpoint(host, port)
                await endpoint.send(numpy.array(tensor_id.bytes))
                await endpoint.send(
                    numpy.array(UCP_DATA_PLANE_COMMANDS.GET, dtype="u1")
                )
                await endpoint.recv(storage)

                if not self._keep_endpoints_open:
                    await self._close_endpoint(host, port)
                return Tensor(data_type, shape, MemoryBuffer.from_dlpack(storage))
            except Exception as e:
                raise DataPlaneError(f"Error Getting Tensor:\n{remote_tensor}") from e

    async def _create_remote_tensor_reference(
        self,
        remote_tensor: ModelInferRequest.InferInputTensor
        | ModelInferResponse.InferOutputTensor,
        result: ModelInferRequest.InferInputTensor
        | ModelInferResponse.InferOutputTensor,
    ):
        tensor_size = get_icp_tensor_size(remote_tensor)
        memory_type = get_icp_memory_type(remote_tensor)
        memory_type_id = get_icp_memory_type_id(remote_tensor)

        if tensor_size is None or memory_type is None or memory_type_id is None:
            raise DataPlaneError("tensor size and memory type must not be none")

        set_icp_shape(result, get_icp_shape(remote_tensor))
        set_icp_data_type(result, get_icp_data_type(remote_tensor))
        set_icp_tensor_size(result, tensor_size)
        set_icp_memory_type(result, memory_type)
        set_icp_memory_type_id(result, memory_type_id)

        if remote_tensor.HasField("contents"):
            for value in remote_tensor.contents.bytes_contents:
                result.contents.bytes_contents.append(value)
            return

        tensor_id, host, port = self._split_tensor_uri(remote_tensor)

        try:
            endpoint = await self._create_endpoint(host, port)
            await endpoint.send(numpy.array(tensor_id.bytes))
            await endpoint.send(
                numpy.array(UCP_DATA_PLANE_COMMANDS.CREATE_REFERENCE, dtype="u1")
            )
            reference_tensor_id_bytes = numpy.empty(self._id_size, dtype="u1")
            await endpoint.recv(reference_tensor_id_bytes)
            if not self._keep_endpoints_open:
                await self._close_endpoint(host, port)
            reference_tensor_id = uuid.UUID(bytes=reference_tensor_id_bytes.tobytes())
            set_icp_tensor_uri(result, f"ucp://{host}:{port}/{reference_tensor_id}")
        except Exception as e:
            raise DataPlaneError("Error Referencing Tensor:\n{remote_tensor}") from e

    async def _release_remote_tensor(
        self,
        remote_tensor: ModelInferRequest.InferInputTensor
        | ModelInferResponse.InferOutputTensor,
    ):
        tensor_id, host, port = self._split_tensor_uri(remote_tensor)

        try:
            endpoint = await self._create_endpoint(host, port)
            await endpoint.send(numpy.array(tensor_id.bytes))
            await endpoint.send(
                numpy.array(UCP_DATA_PLANE_COMMANDS.RELEASE, dtype="u1")
            )
            ack_tensor_id = numpy.empty(self._id_size, dtype="u1")
            await endpoint.recv(ack_tensor_id)
            if not self._keep_endpoints_open:
                await self._close_endpoint(host, port)
        except Exception as e:
            raise DataPlaneError(f"Error Releasing Tensor:\n{remote_tensor}") from e

    def get_tensor(
        self,
        remote_tensor: ModelInferRequest.InferInputTensor
        | ModelInferResponse.InferOutputTensor,
        requested_memory_type: Optional[MemoryType] = None,
        requested_memory_type_id: Optional[int] = None,
    ) -> Tensor:
        if self._event_loop is None:
            raise DataPlaneError("Not connected")
        return asyncio.run_coroutine_threadsafe(
            self._get_remote_tensor(
                remote_tensor, requested_memory_type, requested_memory_type_id
            ),
            self._event_loop,
        ).result()

    def create_input_tensor_reference(
        self,
        remote_tensor: ModelInferRequest.InferInputTensor
        | ModelInferResponse.InferOutputTensor,
    ) -> ModelInferRequest.InferInputTensor:
        if self._event_loop is None:
            raise DataPlaneError("Not connected")
        result = ModelInferRequest.InferInputTensor()
        asyncio.run_coroutine_threadsafe(
            self._create_remote_tensor_reference(remote_tensor, result),
            self._event_loop,
        ).result()
        return result

    def create_output_tensor_reference(
        self,
        remote_tensor: ModelInferRequest.InferInputTensor
        | ModelInferResponse.InferOutputTensor,
    ) -> ModelInferResponse.InferOutputTensor:
        if self._event_loop is None:
            raise DataPlaneError("Not connected")
        result = ModelInferResponse.InferOutputTensor()
        asyncio.run_coroutine_threadsafe(
            self._create_remote_tensor_reference(remote_tensor, result),
            self._event_loop,
        ).result()

        return result

    def release_tensor(
        self,
        remote_tensor: ModelInferRequest.InferInputTensor
        | ModelInferResponse.InferOutputTensor,
    ) -> None:
        if remote_tensor.HasField("contents"):
            return None

        if self._event_loop is None:
            raise DataPlaneError("Not connected")
        return asyncio.run_coroutine_threadsafe(
            self._release_remote_tensor(remote_tensor), self._event_loop
        ).result()

    async def _create_endpoint(self, host: str, port: int):
        endpoint = self._endpoints.get((host, port))
        if endpoint is None:
            LOGGER.debug(f"Creating endpoint for {host}:{port}")
            endpoint = await ucp.create_endpoint(host, port)
            self._endpoints[(host, port)] = endpoint
        else:
            LOGGER.debug(f"Reusing endpoint for {host}:{port}")
        return endpoint

    async def _close_endpoint(self, host: str, port: int):
        endpoint = self._endpoints.pop((host, port), None)
        if endpoint is not None:
            LOGGER.debug(f"Closing endpoint for {host}:{port}")
            await endpoint.close()
        else:
            LOGGER.debug(f"Endpoint for {host}:{port} not found")
