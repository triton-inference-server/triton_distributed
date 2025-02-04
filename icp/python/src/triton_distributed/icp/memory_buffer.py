from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from triton_distributed.icp._dlpack import DLPackObject
from triton_distributed.icp.memory_type import MemoryType


@dataclass
class MemoryBuffer:
    """Memory allocated for a Tensor.

    This object does not own the memory but holds a reference to the
    owner.

    Parameters
    ----------
    data_ptr : int
        Pointer to the allocated memory.
    memory_type : MemoryType
        memory type
    memory_type_id : int
        memory type id (typically the same as device id)
    size : int
        Size of the allocated memory in bytes.
    owner : Any
        Object that owns or manages the memory buffer.  Allocated
        memory must not be freed while a reference to the owner is
        held.

    Examples
    --------
    >>> buffer = MemoryBuffer.from_dlpack(numpy.array([100],dtype=numpy.uint8))

    """

    data_ptr: int
    memory_type: MemoryType
    memory_type_id: int
    size: int
    owner: Any

    @staticmethod
    def from_dlpack(owner: Any) -> MemoryBuffer:
        if not hasattr(owner, "__dlpack__"):
            raise ValueError("Object does not support DLpack protocol")

        dlpack_object = DLPackObject(owner)

        return MemoryBuffer._from_dlpack_object(owner, dlpack_object)

    @staticmethod
    def _from_dlpack_object(owner: Any, dlpack_object: DLPackObject) -> MemoryBuffer:
        if not dlpack_object.contiguous:
            raise ValueError("Only contiguous memory is supported")

        return MemoryBuffer(
            int(dlpack_object.data_ptr),
            dlpack_object.memory_type,
            dlpack_object.memory_type_id,
            dlpack_object.byte_size,
            owner,
        )
