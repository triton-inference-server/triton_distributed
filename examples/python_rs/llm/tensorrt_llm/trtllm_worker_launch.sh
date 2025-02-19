#!/bin/bash

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

set -e

task_with_command="${@:1}"
native_mpi_rank=$OMPI_COMM_WORLD_RANK
# Works with Slurm launching with `--mpi=pmix`
mpi_rank=${PMIX_RANK:-$native_mpi_rank}
echo "mpi_rank: $mpi_rank" >> /dev/stderr

if [ -z "$mpi_rank" ] || [ "$mpi_rank" -eq 0 ]; then
    echo "${mpi_rank} run ${task_with_command} ..." >> /dev/stderr
    $task_with_command
else
    echo "${mpi_rank} launch worker ..." >> /dev/stderr
    python3 -m common.mgmn_worker_node
fi