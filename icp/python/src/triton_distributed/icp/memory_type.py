from enum import IntEnum
from triton_distributed.icp._custom_key_error_dict import CustomKeyErrorDict

MemoryType = IntEnum("MemoryType", names=("CPU", "CPU_PINNED", "GPU"), start=0)

def string_to_memory_type(memory_type_string:str)-> MemoryType:
    try:
        return MemoryType[memory_type_string]
    except KeyError:
        raise ValueError(
            f"Unsupported Memory Type String. Can't convert {memory_type_string} to MemoryType"
        ) from None
