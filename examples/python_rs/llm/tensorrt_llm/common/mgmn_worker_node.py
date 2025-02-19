#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging

from mpi4py.futures import MPICommExecutor
from mpi4py.MPI import COMM_WORLD

with MPICommExecutor(COMM_WORLD) as executor:
    if executor is not None:
        raise RuntimeError(f"rank{COMM_WORLD.rank} should not have executor")

logging.warning(f"worker rank{COMM_WORLD.rank} quited")
