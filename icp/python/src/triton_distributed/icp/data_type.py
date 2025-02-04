# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

def string_to_data_type(data_type_string:str)->DataType:
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
