# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

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
