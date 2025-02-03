from enum import IntEnum

MemoryType = IntEnum("MemoryType", names=("CPU", "CPU_PINNED", "GPU"), start=0)
