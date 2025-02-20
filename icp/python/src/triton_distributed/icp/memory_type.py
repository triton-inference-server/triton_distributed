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

from enum import IntEnum

MemoryType = IntEnum("MemoryType", names=("CPU", "CPU_PINNED", "GPU"), start=0)

# Is more touching necessary to trigger pipeline?
# It is necessary to trigger changes in Python to force valid gitlab execution. 
# Let's touch this file again to trigger tests.


def string_to_memory_type(memory_type_string: str) -> MemoryType:
    try:
        return MemoryType[memory_type_string]
    except KeyError:
        raise ValueError(
            f"Unsupported Memory Type String. Can't convert {memory_type_string} to MemoryType"
        ) from None
