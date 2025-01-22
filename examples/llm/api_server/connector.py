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

import abc
import dataclasses
import typing


class TritonInferenceError(Exception):
    """Error occurred during Triton inference."""


@dataclasses.dataclass
class InferenceRequest:
    """Inference request."""

    inputs: typing.Dict[str, typing.Any] = dataclasses.field(default_factory=dict)
    parameters: typing.Dict[str, typing.Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class InferenceResponse:
    """Inference response."""

    outputs: typing.Dict[str, typing.Any] = dataclasses.field(default_factory=dict)
    error: typing.Optional[str] = None
    final: bool = False
    parameters: typing.Dict[str, typing.Any] = dataclasses.field(default_factory=dict)


class BaseTriton3Connector(abc.ABC):
    """Base class for Triton 3 connector."""

    @abc.abstractmethod
    async def inference(
        self, model_name: str, request: InferenceRequest
    ) -> typing.AsyncGenerator[InferenceResponse, None]:
        """Inference request to Triton 3 system.

        Args:
            model_name: Model name.
            request: Inference request.

        Returns:
            Inference response.

        Raises:
            TritonInferenceError: error occurred during inference.
        """
        raise NotImplementedError

    async def list_models(self) -> typing.List[str]:
        """List models available in Triton 3 system.

        Returns:
            List of model names.
        """
        raise NotImplementedError
