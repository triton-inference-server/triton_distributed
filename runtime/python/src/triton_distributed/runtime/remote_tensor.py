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

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cupy
from cupy_backends.cuda.api.runtime import CUDARuntimeError

from triton_distributed.icp._dlpack import DeviceOrMemoryType, DLDeviceType
from triton_distributed.icp.data_plane import (
    DataPlane,
    get_icp_data_type,
    get_icp_memory_type,
    get_icp_shape,
    get_icp_tensor_size,
)
from triton_distributed.icp.data_type import DataType
from triton_distributed.icp.memory_type import MemoryType
from triton_distributed.icp.protos.icp_pb2 import ModelInferRequest, ModelInferResponse
from triton_distributed.icp.tensor import Tensor

# Run cupy's cuda.is_available once to
# avoid the exception hitting runtime code.
try:
    cupy.cuda.is_available()
except CUDARuntimeError:
    pass


@dataclass
class RemoteTensor:
    remote_tensor: ModelInferRequest.InferInputTensor | ModelInferResponse.InferOutputTensor
    data_plane: DataPlane
    _local_tensor: Optional[Tensor] = None
    # FIXME: This is a hack to avoid double deletion of the tensor
    # Tensor must be explicitly released by the user before data plane connection is closed
    deleted: bool = False

    @property
    def data_type(self) -> DataType | None:
        return get_icp_data_type(self.remote_tensor)

    @property
    def shape(self) -> Sequence[int] | None:
        return get_icp_shape(self.remote_tensor)

    @property
    def memory_type(self) -> MemoryType | None:
        return get_icp_memory_type(self.remote_tensor)

    @property
    def size(self) -> int | None:
        return get_icp_tensor_size(self.remote_tensor)

    @property
    def local_tensor(self) -> Tensor:
        if not self._local_tensor:
            self._local_tensor = self.data_plane.get_tensor(self.remote_tensor)
            if self._local_tensor is None:
                raise ValueError("Not able to resolve Tensor locally")
        return self._local_tensor

    @property
    def data_ptr(self) -> int:
        return self.local_tensor.data_ptr

    def __dlpack__(self, *, stream=None):
        return self.local_tensor.__dlpack__(stream=stream)

    def __dlpack_device__(self) -> tuple[DLDeviceType, int]:
        return self.local_tensor.__dlpack_device__()

    def to_string_array(self):
        return self.local_tensor.to_string_array()

    def to_bytes_array(self):
        return self.local_tensor.to_bytes_array()

    def to_host(self) -> Tensor:
        return self.local_tensor.to_host()

    def to_device(self, device: DeviceOrMemoryType) -> Tensor:
        return self.local_tensor.to_device(device)

    def __del__(self):
        # FIXME: This is a hack to avoid double deletion of the tensor
        # Tensor must be explicitly released by the user before data plane connection is closed
        if not self.deleted:
            self.data_plane.release_tensor(self.remote_tensor)
        self.deleted = True
