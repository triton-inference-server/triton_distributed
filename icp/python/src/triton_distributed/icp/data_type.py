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

from enum import IntEnum

import numpy

from triton_distributed.icp._custom_key_error_dict import CustomKeyErrorDict

DataType = IntEnum(
    "DataType",
    names=(
        "INVALID",
        "BOOL",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "FP16",
        "FP32",
        "FP64",
        "BYTES",
        "BF16",
    ),
    start=0,
)


def string_to_data_type(data_type_string: str) -> DataType:
    try:
        return DataType[data_type_string]
    except KeyError:
        raise ValueError(
            f"Unsupported Data Type String. Can't convert {data_type_string} to DataType"
        ) from None


NUMPY_TO_DATA_TYPE: dict[type, DataType] = CustomKeyErrorDict(
    "Numpy dtype",
    "Data type",
    {
        bool: DataType.BOOL,
        numpy.bool_: DataType.BOOL,
        numpy.int8: DataType.INT8,
        numpy.int16: DataType.INT16,
        numpy.int32: DataType.INT32,
        numpy.int64: DataType.INT64,
        numpy.uint8: DataType.UINT8,
        numpy.uint16: DataType.UINT16,
        numpy.uint32: DataType.UINT32,
        numpy.uint64: DataType.UINT64,
        numpy.float16: DataType.FP16,
        numpy.float32: DataType.FP32,
        numpy.float64: DataType.FP64,
        numpy.bytes_: DataType.BYTES,
        numpy.str_: DataType.BYTES,
        numpy.object_: DataType.BYTES,
    },
)

DATA_TYPE_TO_NUMPY_DTYPE: dict[DataType, type] = CustomKeyErrorDict(
    "Data type",
    "Numpy dtype",
    {
        **{value: key for key, value in NUMPY_TO_DATA_TYPE.items()},
        **{DataType.BYTES: numpy.object_},
        **{DataType.BOOL: numpy.bool_},
    },
)
