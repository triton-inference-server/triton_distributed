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

################################################################################
# This file contains the DLPack API wrapped in Python style (see
# 'dlpack.h' for detail) and the utilities for Triton client to interact
# with DLPack
#
# Ref:
# https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h
# https://github.com/dmlc/dlpack/blob/main/apps/numpy_dlpack/dlpack/from_numpy.py
################################################################################

import ctypes

from triton_distributed.icp._custom_key_error_dict import CustomKeyErrorDict
from triton_distributed.icp.data_type import DataType
from triton_distributed.icp.memory_type import MemoryType, string_to_memory_type

try:
    import cupy
except ImportError:
    cupy = None

# Need to explicit set the res / arg types for pythonapi functions to
# work properly
ctypes.pythonapi.PyMem_RawMalloc.restype = ctypes.c_void_p
ctypes.pythonapi.PyMem_RawFree.argtypes = [ctypes.c_void_p]

ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
ctypes.pythonapi.PyCapsule_New.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_void_p,
]

ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

from typing import Union

c_str_dltensor = b"dltensor"


class DLDeviceType(ctypes.c_int):
    kDLCPU = 1
    kDLCUDA = 2
    kDLCUDAHost = 3
    kDLOpenCL = 4
    kDLVulkan = 7
    kDLMetal = 8
    kDLVPI = 9
    kDLROCM = 10
    kDLROCMHost = 11
    kDLExtDev = 12
    kDLCUDAManaged = 13
    kDLOneAPI = 14
    kDLWebGPU = 15
    kDLHexagon = 16


DeviceOrMemoryType = Union[
    tuple[MemoryType, int], MemoryType, tuple[DLDeviceType, int], str
]


class DLDevice(ctypes.Structure):
    _fields_ = [
        ("device_type", ctypes.c_int),
        ("device_id", ctypes.c_int),
    ]


class DLDataTypeCode(ctypes.c_uint8):
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLOpaquePointer = 3
    kDLBfloat = 4
    kDLComplex = 5
    kDLBool = 6


class DLDataType(ctypes.Structure):
    _fields_ = [
        ("type_code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
    ]


# Utilities


def _raise_error(msg):
    """
    Raise error with the provided message
    """
    raise Exception(msg) from None


# Use as managed context in DLPack that doesn't hold ownership of the
# data content.
class DataViewContext:
    def __init__(self, shape) -> None:
        # Convert the Python object to ctypes objects expected by
        # DLPack
        self._shape = (ctypes.c_int64 * len(shape))(*shape)
        # No strides: compact and row-major
        self._strides = ctypes.POINTER(ctypes.c_int64)()

    def as_manager_ctx(self) -> ctypes.c_void_p:
        py_obj = ctypes.py_object(self)
        py_obj_ptr = ctypes.pointer(py_obj)
        ctypes.pythonapi.Py_IncRef(py_obj)
        ctypes.pythonapi.Py_IncRef(ctypes.py_object(py_obj_ptr))
        return ctypes.cast(py_obj_ptr, ctypes.c_void_p)


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def managed_tensor_deleter(handle: ctypes.c_void_p) -> None:
    dl_managed_tensor = DLManagedTensor.from_address(handle)  # type: ignore
    py_obj_ptr = ctypes.cast(
        dl_managed_tensor.manager_ctx, ctypes.POINTER(ctypes.py_object)
    )
    py_obj = py_obj_ptr.contents
    ctypes.pythonapi.Py_DecRef(py_obj)
    ctypes.pythonapi.Py_DecRef(ctypes.py_object(py_obj_ptr))
    ctypes.pythonapi.PyMem_RawFree(handle)


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def pycapsule_deleter(handle: ctypes.c_void_p) -> None:
    pycapsule: ctypes.py_object = ctypes.cast(handle, ctypes.py_object)
    if ctypes.pythonapi.PyCapsule_IsValid(pycapsule, c_str_dltensor):
        dl_managed_tensor = ctypes.pythonapi.PyCapsule_GetPointer(
            pycapsule, c_str_dltensor
        )
        managed_tensor_deleter(dl_managed_tensor)
        ctypes.pythonapi.PyCapsule_SetDestructor(pycapsule, None)


def is_contiguous_data(
    ndim: ctypes.c_int,
    shape: ctypes.POINTER(ctypes.c_int64),  # type: ignore
    stride: ctypes.POINTER(ctypes.c_int64),  # type: ignore
):
    # If 'stride' doesn't capture valid value
    if (stride is None) or (not bool(stride)):
        return True
    calculated_stride = 1
    # iterate stride in reverse order [ndim-1, -1)
    for i in reversed(range(ndim)):  # type: ignore
        if stride[i] != calculated_stride:
            return False
        calculated_stride *= shape[i]
    return True


def get_byte_size(
    dtype: DLDataType, ndim: ctypes.c_int, shape: ctypes.POINTER(ctypes.c_int64)  # type: ignore
):
    element_byte_size = dtype.bits * dtype.lanes // 8  # Assume 8 bits in a byte
    for i in range(ndim):  # type: ignore
        element_byte_size *= shape[i]
    return element_byte_size


def get_dlpack_capsule(dlpack_obj, stream=None):
    # Extract PyCapsule of the DLPack object
    if hasattr(dlpack_obj, "__dlpack__"):
        if not hasattr(dlpack_obj, "__dlpack_device__"):
            _raise_error(
                "DLPack expects '__dlpack_device__' if '__dlpack__' has been defined"
            )
        device = dlpack_obj.__dlpack_device__()
        # Have to condition on the device type as, using numpy as example,
        # some DLPack implementation doesn't accept 'stream' as arguments
        if device != DLDeviceType.kDLCUDA:
            return dlpack_obj.__dlpack__()
        else:
            return dlpack_obj.__dlpack__(stream)
    else:
        # Old interface where PyCapsule object is passed directly
        return dlpack_obj


def get_dlpack_device(dlpack_obj):
    if hasattr(dlpack_obj, "__dlpack_device__"):
        return dlpack_obj.__dlpack_device__()
    return None


def get_managed_tensor(dlcapsule):
    ptr = ctypes.pythonapi.PyCapsule_GetPointer(dlcapsule, c_str_dltensor)
    return DLManagedTensor.from_address(ptr)


class DLPackObject:
    def __init__(self, value) -> None:
        try:
            stream = None
            device, device_id = value.__dlpack_device__()
            if device == DLDeviceType.kDLCUDA:
                if cupy is None:
                    raise ValueError(
                        f"DLPack synchronization on device {device,device_id} not supported"
                    )
                with cupy.cuda.Device(device_id):
                    stream = 1  # legacy default stream
                    self._capsule = get_dlpack_capsule(value, stream)
                    self._tensor = get_managed_tensor(self._capsule).dl_tensor
            else:
                self._capsule = get_dlpack_capsule(value)
                self._tensor = get_managed_tensor(self._capsule).dl_tensor
        except Exception as e:
            raise ValueError(f"Object does not support DLPack protocol: {e}") from None

    def __eq__(self, other) -> bool:
        if not isinstance(other, DLPackObject):
            return False
        if self.byte_size != other.byte_size:
            return False
        if self.memory_type != other.memory_type:
            return False
        if self.memory_type_id != other.memory_type_id:
            return False
        if self.shape != other.shape:
            return False
        if self.data_ptr != other.data_ptr:
            return False
        if self.contiguous != other.contiguous:
            return False
        if self.data_type != other.data_type:
            return False
        return True

    @property
    def byte_size(self) -> int:
        return get_byte_size(self._tensor.dtype, self._tensor.ndim, self._tensor.shape)

    @property
    def memory_type(self) -> MemoryType:
        return DLPACK_DEVICE_TYPE_TO_MEMORY_TYPE[self._tensor.device.device_type]

    @property
    def memory_type_id(self) -> int:
        return self._tensor.device.device_id

    @property
    def shape(self) -> list[int]:
        return [self._tensor.shape[i] for i in range(self._tensor.ndim)]

    @property
    def data_type(self) -> DataType:
        return DLPACK_TO_DATA_TYPE[self.dlpack_data_type]

    @property
    def dlpack_data_type(self) -> tuple[DLDataTypeCode, int]:
        return (self._tensor.dtype.type_code, self._tensor.dtype.bits)

    @property
    def data_ptr(self) -> ctypes.c_void_p:
        return self._tensor.data + self._tensor.byte_offset

    @property
    def contiguous(self) -> bool:
        return is_contiguous_data(
            self._tensor.ndim, self._tensor.shape, self._tensor.strides
        )


DLPACK_DEVICE_TYPE_TO_MEMORY_TYPE: dict[DLDeviceType, MemoryType] = CustomKeyErrorDict(
    "DLPack device type",
    "Memory type",
    {
        DLDeviceType.kDLCUDA: MemoryType.GPU,
        DLDeviceType.kDLCPU: MemoryType.CPU,
    },
)

MEMORY_TYPE_TO_DLPACK_DEVICE_TYPE: dict[MemoryType, DLDeviceType] = CustomKeyErrorDict(
    "Memory type",
    "DLPack device type",
    {
        **{value: key for key, value in DLPACK_DEVICE_TYPE_TO_MEMORY_TYPE.items()},
        **{MemoryType.CPU_PINNED: DLDeviceType.kDLCPU},
    },
)


def parse_device_or_memory_type(
    device_or_memory_type: DeviceOrMemoryType,
) -> tuple[MemoryType, int]:
    memory_type = None
    memory_type_id = 0
    if isinstance(device_or_memory_type, tuple):
        if isinstance(device_or_memory_type[0], MemoryType):
            memory_type = device_or_memory_type[0]
            memory_type_id = device_or_memory_type[1]
        elif isinstance(device_or_memory_type[0], DLDeviceType):
            memory_type = DLPACK_DEVICE_TYPE_TO_MEMORY_TYPE[device_or_memory_type[0]]
            memory_type_id = device_or_memory_type[1]
        else:
            raise ValueError(f"Invalid memory type {device_or_memory_type}")
    elif isinstance(device_or_memory_type, MemoryType):
        memory_type = device_or_memory_type
        memory_type_id = 0
    elif isinstance(device_or_memory_type, str):
        memory_str_tuple = device_or_memory_type.split(":")
        if len(memory_str_tuple) > 2:
            raise ValueError(f"Invalid memory type string {device_or_memory_type}")
        memory_type = string_to_memory_type(memory_str_tuple[0].upper())
        if len(memory_str_tuple) == 2:
            try:
                memory_type_id = int(memory_str_tuple[1])
            except ValueError:
                raise ValueError(
                    f"Invalid memory type string {device_or_memory_type}"
                ) from None
        else:
            memory_type_id = 0
    return (memory_type, memory_type_id)


DLPACK_TO_DATA_TYPE: dict[tuple[DLDataTypeCode, int], DataType] = CustomKeyErrorDict(
    "DLPack data type",
    "Data type",
    {
        (DLDataTypeCode.kDLBool, 8): DataType.BOOL,
        (DLDataTypeCode.kDLInt, 8): DataType.INT8,
        (
            DLDataTypeCode.kDLInt,
            16,
        ): DataType.INT16,
        (
            DLDataTypeCode.kDLInt,
            32,
        ): DataType.INT32,
        (
            DLDataTypeCode.kDLInt,
            64,
        ): DataType.INT64,
        (
            DLDataTypeCode.kDLUInt,
            8,
        ): DataType.UINT8,
        (
            DLDataTypeCode.kDLUInt,
            16,
        ): DataType.UINT16,
        (
            DLDataTypeCode.kDLUInt,
            32,
        ): DataType.UINT32,
        (
            DLDataTypeCode.kDLUInt,
            64,
        ): DataType.UINT64,
        (
            DLDataTypeCode.kDLFloat,
            16,
        ): DataType.FP16,
        (
            DLDataTypeCode.kDLFloat,
            32,
        ): DataType.FP32,
        (
            DLDataTypeCode.kDLFloat,
            64,
        ): DataType.FP64,
        (
            DLDataTypeCode.kDLBfloat,
            16,
        ): DataType.BF16,
    },
)

DATA_TYPE_TO_DLPACK_DTYPE: dict[DataType, DLDataType] = CustomKeyErrorDict(
    "Data type",
    "DLPack data type",
    {
        value: DLDataType(type_code=key[0], bits=key[1], lanes=1)
        for key, value in DLPACK_TO_DATA_TYPE.items()
    },
)
