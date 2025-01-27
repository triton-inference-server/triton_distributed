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

# A script to download a Python wheel, patch it, copy additional files,
# repackage it, and optionally install the new wheel.

import logging
import math
import socket
import threading
import typing
import uuid

import torch
import torch.distributed
import tritonserver
import zmq

from triton_distributed.icp.data_plane import (
    set_icp_data_type,
    set_icp_memory_type,
    set_icp_shape,
    set_icp_tensor_size,
    set_icp_tensor_uri,
)
from triton_distributed.icp.protos.icp_pb2 import ModelInferRequest
from triton_distributed.icp.ucp_data_plane import DataPlaneError, UcpDataPlane

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class VllmUcpDataPlane:
    def __init__(
        self,
        hostname: typing.Optional[str] = None,
        port: int = 0,
        keep_endpoints_open: bool = False,
    ) -> None:
        self._data_plane = UcpDataPlane(hostname, port, keep_endpoints_open)

    @property
    def hostname(self) -> str:
        return self._data_plane.hostname

    @property
    def port(self) -> int:
        return self._data_plane.port

    def connect(self) -> None:
        self._data_plane.connect()

    def close(self) -> None:
        self._data_plane.close()

    def put_input_tensor(
        self,
        tensor: torch.Tensor,
        tensor_id: typing.Optional[uuid.UUID] = None,
    ):
        logger.debug(
            f"Putting input tensor with id {tensor_id} on {self.hostname}:{self.port}"
        )
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.view(torch.float16)
        triton_tensor = tritonserver.Tensor.from_dlpack(tensor)
        self._data_plane.put_input_tensor(triton_tensor, tensor_id)

    def put_output_tensor(
        self,
        tensor: torch.Tensor,
        tensor_id: typing.Optional[uuid.UUID] = None,
    ):
        logger.debug(
            f"Putting input tensor with id {tensor_id} on {self.hostname}:{self.port}"
        )
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.view(torch.float16)
        triton_tensor = tritonserver.Tensor.from_dlpack(tensor)
        self._data_plane.put_output_tensor(triton_tensor, tensor_id)

    def get_tensor(
        self,
        tensor_uri: str,
        shape: typing.Sequence[int],
        dtype: torch.dtype,
        device_id: int,
    ) -> torch.Tensor:
        logger.debug("Getting tensor from %s", tensor_uri)
        result = ModelInferRequest.InferInputTensor()
        triton_dtype = {
            torch.float32: tritonserver.DataType.FP32,
            torch.float16: tritonserver.DataType.FP16,
            torch.bfloat16: tritonserver.DataType.FP16,
            torch.uint8: tritonserver.DataType.UINT8,
        }.get(dtype)
        if triton_dtype is None:
            raise DataPlaneError(f"Unsupported dtype {dtype}")
        tensor_size = math.prod(shape) * dtype.itemsize
        set_icp_data_type(result, triton_dtype)
        set_icp_shape(result, shape)
        set_icp_tensor_uri(result, tensor_uri)
        set_icp_memory_type(result, tritonserver.MemoryType.GPU)
        set_icp_tensor_size(result, tensor_size)
        triton_tensor = self._data_plane.get_tensor(
            remote_tensor=result,
            requested_memory_type=tritonserver.MemoryType.GPU,
            requested_memory_type_id=device_id,
        )
        tensor = torch.utils.dlpack.from_dlpack(triton_tensor)
        if dtype == torch.bfloat16:
            tensor = tensor.view(torch.bfloat16)
        logger.debug("Got tensor from %s", tensor_uri)
        return tensor


class VllmNcclDataPlane:
    def __init__(
        self,
        hostname: str = "",
        port: int = 0,
        # FIXME: world_size and rank both unused
        world_size: int = -1,
        rank: int = -1,
    ) -> None:
        if not torch.distributed.is_initialized():
            raise RuntimeError("NCCL backend not initialized")

        if not hostname:
            hostname = socket.gethostname()
        if port == 0:
            port = 13337 + torch.distributed.get_rank()
        self._hostname = hostname
        self._port = port
        self._rank = torch.distributed.get_rank()
        self._world_size: int = world_size
        self._current_device = torch.cuda.current_device()
        # FIXME: Use stricter type for req value in tuple
        self.store: typing.Dict[str, typing.Tuple[torch.Tensor, int, typing.Any]] = {}
        self.context = zmq.Context()
        self.rep_socket = self.context.socket(zmq.REP)
        logger.info(f"Rank {self._rank} binding to {self._hostname}:{self._port}")
        self.rep_socket.bind(f"tcp://{self._hostname}:{self._port}")
        self._listener_thread = threading.Thread(
            target=self.listen_for_requests, daemon=True
        )
        self._listener_thread.start()
        # FIXME: Use stricter ZMQ socket type hint
        self.req_sockets: typing.Dict[str, typing.Any] = {}
        logger.info(f"Rank {self._rank} connected to the server")

    @property
    def world_size(self):
        return self._world_size

    @property
    def rank(self):
        return self._rank

    def put_input_tensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        tensor_id: str,
        remote_address: typing.Optional[str] = None,
    ):
        return self._put_tensor(tensor, rank, tensor_id, remote_address)

    def put_output_tensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        tensor_id: str,
        remote_address: typing.Optional[str] = None,
    ):
        return self._put_tensor(tensor, rank, tensor_id, remote_address)

    def get_tensor(
        self,
        rank: int,
        tensor_id: str,
        remote_address: str,
    ) -> torch.Tensor:
        return self._get_tensor(rank, tensor_id, remote_address)

    def _put_tensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        tensor_id: str,
        remote_address: typing.Optional[str] = None,
    ):
        logger.debug(
            f"Rank {self._rank} storing tensor with id {tensor_id} of shape {tensor.shape} and dtype {tensor.dtype}"
        )
        if remote_address is None:
            self.store[tensor_id] = (tensor, rank, None)
        else:
            tensor_shape = "_".join(str(dim) for dim in tensor.shape)
            if remote_address not in self.req_sockets:
                self.req_sockets[remote_address] = self.context.socket(zmq.REQ)
                self.req_sockets[remote_address].connect(f"tcp://{remote_address}")

            req_socket = self.req_sockets[remote_address]
            req_socket.connect(f"tcp://{remote_address}")
            req_socket.send_string(f"PUT {self._rank} {tensor_shape} {tensor_id}")
            ret = req_socket.recv_string()
            assert ret == "OK"
            torch.distributed.isend(tensor, dst=rank)

    def _get_tensor(
        self,
        rank: int,
        tensor_id: str,
        remote_address: str,
    ) -> torch.Tensor:
        logger.debug(f"Rank {self._rank} receiving tensor from rank {rank}")
        if tensor_id in self.store:
            tensor, _, req = self.store.pop(tensor_id)
            req.wait()  # TODO ptarasiewicz we should run other request instead of wait
            logger.debug(f"Rank {self._rank} received tensor from rank {rank}")
            return tensor
        raise NotImplementedError("Getting tensor from remote rank not implemented")

    def _receive_tensor(
        self,
        tensor_id: str,
        rank: int,
        shape: typing.Sequence[int],
    ):
        tensor = torch.empty(
            shape, dtype=torch.uint8, device=f"cuda:{self._current_device}"
        )
        req = torch.distributed.irecv(tensor, src=rank)
        self.store[tensor_id] = (tensor, rank, req)

    def listen_for_requests(self):
        while True:
            cmd, _rank, _shape, tensor_id = self.rep_socket.recv_string().split()
            logger.debug(f"Rank {self._rank} received request for tensor {tensor_id}")
            self.rep_socket.send_string("OK")
            if cmd == "GET":
                raise NotImplementedError(
                    "Getting tensor from remote rank not implemented"
                )
            elif cmd == "PUT":
                rank = int(_rank)
                shape = [int(dim) for dim in _shape.split("_")]
                self._receive_tensor(tensor_id, rank, shape)
