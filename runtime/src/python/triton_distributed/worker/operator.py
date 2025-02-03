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

"""Interface for Operators"""

import abc
from dataclasses import dataclass, field
from typing import Any, Optional, Type

from tritonserver import Server

from triton_distributed.icp.data_plane import DataPlane
from triton_distributed.icp.request_plane import RequestPlane
from triton_distributed.worker.remote_request import RemoteInferenceRequest


class Operator(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        name: str,
        version: int,
        triton_core: Server,
        request_plane: RequestPlane,
        data_plane: DataPlane,
        parameters: Optional[dict[str, str | int | bool | bytes]] = field(
            default_factory=dict
        ),
        repository: Optional[str] = None,
        logger: Optional[Any] = None,
    ):
        pass

    @abc.abstractmethod
    async def execute(self, requests: list[RemoteInferenceRequest]) -> None:
        pass


@dataclass
class OperatorConfig:
    """
    Holds the properties of a hosted operator
    """

    name: str
    implementation: str | Type[Operator]
    repository: Optional[str] = None
    version: int = 1
    max_inflight_requests: int = 5
    parameters: Optional[dict[str, str | int | bool | bytes]] = field(
        default_factory=dict
    )
    log_level: Optional[int] = None
