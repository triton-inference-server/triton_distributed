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

import asyncio
import json
import typing
from typing import Optional

import numpy as np
from .connector import (
    BaseTriton3Connector,
    InferenceRequest,
    InferenceResponse,
)

from .remote_connector import RemoteConnector
from triton_distributed.worker.remote_operator import RemoteOperator
from triton_distributed.worker.remote_tensor import RemoteTensor


class RemoteModelConnector(BaseTriton3Connector):
    """Connector for Triton 3 model."""

    def __init__(
        self,
        nats_url: str,
        model_name: str,
        model_version: Optional[str] = None,
        data_plane_host: Optional[str] = None,
        data_plane_port: int = 0,
        keep_dataplane_endpoints_open: bool = False,
    ):
        """Initialize Triton 3 connector.

        Args:
            nats_url: NATS URL (e.g. "localhost:4222").
            model_name: Model name.
            model_version: Model version. Default is "1".
            data_plane_host: Data plane host (e.g. "localhost").
            data_plane_port: Data plane port (e.g. 8001). You can use 0 to let the system choose a port.
            keep_dataplane_endpoints_open: Keep data plane endpoints open to avoid reconnecting. Default is False.

        Example:
            remote_model_connector = RemoteModelConnector(
                nats_url="localhost:4222",
                data_plane_host="localhost",
                data_plane_port=8001,
                model_name="model_name",
                model_version="1",
            )
            async with remote_model_connector:
                request = InferenceRequest(inputs={"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])})
                async for response in remote_model_connector.inference(model_name="model_name", request=request):
                    print(response.outputs)
        """
        self._connector = RemoteConnector(
            nats_url,
            data_plane_host,
            data_plane_port,
            keep_dataplane_endpoints_open=keep_dataplane_endpoints_open,
        )
        self._model_name = model_name
        if model_version is None:
            model_version = "1"
        self._model_version = model_version

    async def connect(self):
        """Connect to Triton 3 server."""
        await self._connector.connect()
        self._model = RemoteOperator(
            name=self._model_name,
            version=self._model_version,
            request_plane=self._connector._request_plane,
            data_plane=self._connector._data_plane,
        )

    async def close(self):
        """Disconnect from Triton 3 server."""
        await self._connector.close()

    async def __aenter__(self):
        """Enter context manager."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit context manager."""
        await self.close()

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
        if not self._connector._connected:
            await self.connect()
        else:
            if self._model_name != model_name:
                self._model_name = model_name
                self._model_version = "1"
                self._model = RemoteOperator(
                    self._model_name,
                    self._model_version,
                    self._connector._request_plane,
                    self._connector._data_plane,
                )
        results = []

        for key, value in request.parameters.items():
            if isinstance(value, dict):
                request.parameters[key] = "JSON:" + json.dumps(value)

        results.append(
            self._model.async_infer(
                inputs=request.inputs,
                parameters=request.parameters,
            )
        )

        for result in asyncio.as_completed(results):
            responses = await result
            async for response in responses:
                triton_response = response.to_model_infer_response(
                    self._connector._data_plane
                )
                outputs = {}
                for output in triton_response.outputs:
                    remote_tensor = RemoteTensor(output, self._connector._data_plane)
                    try:
                        local_tensor = remote_tensor.local_tensor
                        numpy_tensor = np.from_dlpack(local_tensor)
                    finally:
                        # FIXME: This is a workaround for the issue that the remote tensor
                        # is released after connection is closed.
                        remote_tensor.__del__()
                    outputs[output.name] = numpy_tensor
                infer_response = InferenceResponse(
                    outputs=outputs,
                    final=response.final,
                    parameters=response.parameters,
                )
                yield infer_response
