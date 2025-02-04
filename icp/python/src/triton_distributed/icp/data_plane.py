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

"""Abstract Class for interacting with Triton Distributed Inter-Component Protocol Data Plane"""

import abc
import uuid
from typing import Optional, Sequence

import cupy
import numpy

from triton_distributed.icp.data_type import DATA_TYPE_TO_NUMPY_DTYPE, DataType, string_to_data_type
from triton_distributed.icp.memory_buffer import MemoryBuffer
from triton_distributed.icp.memory_type import MemoryType
from triton_distributed.icp.memory_type import string_to_memory_type
from triton_distributed.icp.protos.icp_pb2 import ModelInferRequest, ModelInferResponse
from triton_distributed.icp.tensor import Tensor


class DataPlaneError(Exception):
    pass


ICP_TENSOR_URI = "icp_tensor_uri"
ICP_MEMORY_TYPE = "icp_memory_type"
ICP_MEMORY_TYPE_ID = "icp_memory_type_id"
ICP_TENSOR_SIZE = "icp_tensor_size"


def set_icp_shape(
    message: ModelInferRequest.InferInputTensor | ModelInferResponse.InferOutputTensor,
    value: Sequence[int],
) -> None:
    for dim in value:
        message.shape.append(dim)


def get_icp_shape(
    message: ModelInferRequest.InferInputTensor | ModelInferResponse.InferOutputTensor,
) -> Sequence[int]:
    return message.shape


def set_icp_data_type(
    message: ModelInferRequest.InferInputTensor | ModelInferResponse.InferOutputTensor,
    value: DataType,
) -> None:
    message.datatype = value.name


def get_icp_data_type(
    message: ModelInferRequest.InferInputTensor | ModelInferResponse.InferOutputTensor,
) -> DataType:
    return string_to_data_type(message.datatype)


def set_icp_tensor_uri(
    message: ModelInferRequest.InferInputTensor | ModelInferResponse.InferOutputTensor,
    value: str,
) -> None:
    message.parameters[ICP_TENSOR_URI].string_param = value


def get_icp_tensor_uri(
    message: ModelInferRequest.InferInputTensor | ModelInferResponse.InferOutputTensor,
) -> str | None:
    if ICP_TENSOR_URI not in message.parameters:
        return None
    return message.parameters[ICP_TENSOR_URI].string_param


def set_icp_tensor_size(
    message: ModelInferRequest.InferInputTensor | ModelInferResponse.InferOutputTensor,
    value: int,
) -> None:
    message.parameters[ICP_TENSOR_SIZE].uint64_param = value


def get_icp_tensor_size(
    message: ModelInferRequest.InferInputTensor | ModelInferResponse.InferOutputTensor,
) -> int | None:
    if ICP_TENSOR_SIZE not in message.parameters:
        return None
    return message.parameters[ICP_TENSOR_SIZE].uint64_param


def set_icp_memory_type(
    message: ModelInferRequest.InferInputTensor | ModelInferResponse.InferOutputTensor,
    value: MemoryType,
) -> None:
    message.parameters[ICP_MEMORY_TYPE].string_param = value.name


def get_icp_memory_type(
    message: ModelInferRequest.InferInputTensor | ModelInferResponse.InferOutputTensor,
) -> MemoryType | None:
    if ICP_MEMORY_TYPE not in message.parameters:
        return None
    return string_to_memory_type(message.parameters[ICP_MEMORY_TYPE].string_param)


def set_icp_memory_type_id(
    message: ModelInferRequest.InferInputTensor | ModelInferResponse.InferOutputTensor,
    value: int,
) -> None:
    message.parameters[ICP_MEMORY_TYPE_ID].int64_param = value


def get_icp_memory_type_id(
    message: ModelInferRequest.InferInputTensor | ModelInferResponse.InferOutputTensor,
) -> int | None:
    if ICP_MEMORY_TYPE_ID not in message.parameters:
        return None
    return message.parameters[ICP_MEMORY_TYPE_ID].int64_param


def set_icp_tensor_contents(
    message: ModelInferRequest.InferInputTensor | ModelInferResponse.InferOutputTensor,
    tensor: Tensor,
) -> None:
    set_icp_memory_type(message, MemoryType.CPU)
    set_icp_memory_type_id(message, 0)
    set_icp_tensor_size(message, tensor.size)
    if tensor.data_type == DataType.BYTES:
        array = tensor.to_bytes_array()
        for i in list(array.flat):
            message.contents.bytes_contents.append(i)
    else:
        if tensor.memory_type == MemoryType.CPU:
            # Directly use the memory buffer when contents on the CPU.
            array = tensor.memory_buffer.owner
        elif tensor.memory_type == MemoryType.GPU:
            with cupy.cuda.Device(tensor.memory_buffer.memory_type_id):
                array = cupy.from_dlpack(tensor)
        else:
            raise ValueError(f"Invalid Tensor Memory Type {tensor.memory_type}")
        message.contents.bytes_contents.append(array.tobytes())


def get_icp_tensor_contents(
    message: ModelInferRequest.InferInputTensor | ModelInferResponse.InferOutputTensor,
) -> Tensor | None:
    if not message.HasField("contents"):
        # Return None if the content is not part of message
        return None

    datatype = get_icp_data_type(message)
    shape = get_icp_shape(message)

    tensor = None
    if datatype == DataType.BYTES:
        array = numpy.array(
            [
                message.contents.bytes_contents[i]
                for i in range(len(message.contents.bytes_contents))
            ]
        )
        array = numpy.reshape(array, shape)
        tensor = Tensor.from_bytes_array(array)
    else:
        array = numpy.array(
            numpy.frombuffer(
                message.contents.bytes_contents[0],
                dtype=DATA_TYPE_TO_NUMPY_DTYPE[datatype],
            )
        )
        tensor = Tensor(datatype, shape, MemoryBuffer.from_dlpack(array))

    return tensor


class DataPlane(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def connect(self) -> None:
        pass

    @abc.abstractmethod
    def put_input_tensor(
        self, tensor: Tensor, tensor_id: Optional[uuid.UUID], use_tensor_contents: bool
    ) -> ModelInferRequest.InferInputTensor:
        pass

    @abc.abstractmethod
    def put_output_tensor(
        self, tensor: Tensor, tensor_id: Optional[uuid.UUID], use_tensor_contents: bool
    ) -> ModelInferResponse.InferOutputTensor:
        pass

    @abc.abstractmethod
    def get_tensor(
        self,
        remote_tensor: ModelInferRequest.InferInputTensor
        | ModelInferResponse.InferOutputTensor,
        requested_memory_type: Optional[MemoryType] = None,
        requested_memory_type_id: Optional[int] = None,
    ) -> Tensor:
        pass

    @abc.abstractmethod
    def create_input_tensor_reference(
        self,
        remote_tensor: ModelInferRequest.InferInputTensor
        | ModelInferResponse.InferOutputTensor,
    ) -> ModelInferRequest.InferInputTensor:
        pass

    @abc.abstractmethod
    def create_output_tensor_reference(
        self,
        remote_tensor: ModelInferRequest.InferInputTensor
        | ModelInferResponse.InferOutputTensor,
    ) -> ModelInferResponse.InferOutputTensor:
        pass

    @abc.abstractmethod
    def release_tensor(
        self,
        remote_tensor: ModelInferRequest.InferInputTensor
        | ModelInferResponse.InferOutputTensor,
    ) -> None:
        pass
