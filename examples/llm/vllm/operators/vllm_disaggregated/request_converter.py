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

import json
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Optional, Tuple

import numpy as np
from pydantic import BaseModel
from tritonserver import Tensor as TritonTensor
from tritonserver._api._response import InferenceResponse as TritonInferenceResponse

from triton_distributed.worker.remote_request import RemoteInferenceRequest
from triton_distributed.worker.remote_response import RemoteInferenceResponse

from .remote_connector import RemoteConnector

class LocalModel(BaseModel):
    name: str
    version: str


class RequestConverter:
    """Request converter. Class converts requests to convenient format for processing."""

    def __init__(
        self,
        nats_url: str,
        data_plane_host: Optional[str] = None,
        data_plane_port: int = 0,
        keep_dataplane_endpoints_open: bool = False,
        model_name: Optional[str] = None,
    ):
        """Initialize RequestAdapter.

        Args:
            nats_url: NATS URL (e.g. "localhost:4222").
            data_plane_host: Data plane host (e.g. "localhost").
            data_plane_port: Data plane port (e.g. 8001). You can use 0 to let the system choose a port.
            keep_dataplane_endpoints_open: Keep data plane endpoints open to avoid reconnecting. Default is False.

        Example for async model:
            worker = RequestConverter(
                nats_url="localhost:4222",
                data_plane_host="localhost",
                data_plane_port=8001,
            )
            async with worker:
                # This flow will process 10 requests at a time
                processors = []
                async def processing(request, callable):
                    request, callable = await queue.get()
                    inputs = request["inputs"]
                    parameters = request["parameters"]
                    output_tensor = inputs["a"] + inputs["b"]
                    try:
                        await callable({"c": output_tensor})
                        for _ in range(parameters["increment"]):
                            output_tensor += 1
                            await callable({"c": output_tensor})
                    finally:
                        await callable({"c": output_tensor}, final=True)
                async for request, callable in worker.pull(model_name="model_name", batch_size=10):
                    # Check if batch size was reached
                    if len(processors) >= 10:
                        done, pending = asyncio.wait(processors, return_when=asyncio.FIRST_COMPLETED)
                        processors = list(pending)
                    processors.append(processing(request, callable))

        """
        self._connector = RemoteConnector(
            nats_url,
            data_plane_host,
            data_plane_port,
            keep_dataplane_endpoints_open=keep_dataplane_endpoints_open,
        )
        self._local_model = LocalModel(name=model_name, version="1")

    async def connect(self):
        """Connect to Triton 3 server."""
        await self._connector.connect()

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

    async def pull(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        batch_size: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> AsyncGenerator[
        Tuple[Dict[str, Any], Callable[[Dict[str, Any]], Awaitable[None]]], None
    ]:
        """Pull requests from request plane and data plane.

        Pull returns an async generator that yields a tuple of request and callable.
        Request contains inputs and parameters. Inputs are a dictionary of input names and numpy arrays. Parameters are
        a dictionary of scalar parameters like sampling parameters in language models.

        Callable is a function that takes outputs, error and final as arguments. Outputs are a dictionary of output names
        and numpy arrays. Error is Exception. Final is a boolean that indicates if the response is final.

        Args:
            model_name: Model name.
            model_version: Model version. Default is "1".
            batch_size: Batch size. Default is 1.
            timeout: Max duration of the pull request before it expires. Default is None.

        Returns:
            Inference request and callable.

        Example:
            worker = PythonWorkerConnector(
                nats_url="localhost:4222",
                data_plane_host="localhost",
                data_plane_port=8001,
            )
            asyn with worker:
                # This flow will process single request at a time
                async for request, callable in worker.pull(model_name="model_name"):
                    # This is siple add model with incrementing the input tensor by increment parameter
                    inputs = request["inputs"]
                    parameters = request["parameters"]
                    output_tensor = inputs["a"] + inputs["b"]
                    try:
                        await callable({"c": output_tensor})
                        for _ in range(parameters["increment"]):
                            output_tensor += 1
                            await callable({"c": output_tensor})
                    finally:
                        await callable({"c": output_tensor}, final=True)
        """
        if not self._connector._connected:
            await self.connect()
        if model_version is None:
            model_version = "1"
        if batch_size is None:
            batch_size = 1
        local_model = LocalModel(
            name=model_name,
            version=model_version,
        )
        kwargs = {
            "model_name": model_name,
            "model_version": model_version,
            "number_requests": batch_size,
        }

        if timeout is not None:
            kwargs["timeout"] = timeout
        while True:
            requests_iterator = await self._connector._request_plane.pull_requests(
                **kwargs
            )

            async for request in requests_iterator:
                inputs, remote_request, return_callable = await self.adapt_request(
                    request, local_model
                )

                yield {
                    "inputs": inputs,
                    "parameters": remote_request.parameters,
                }, return_callable

    async def adapt_request(self, request, local_model: Optional[LocalModel] = None):
        if local_model is None:
            local_model = self._local_model

        if isinstance(request, RemoteInferenceRequest):
            remote_request = request
            request = remote_request.to_model_infer_request()
        else:
            remote_request = RemoteInferenceRequest.from_model_infer_request(
                request,
                self._connector._data_plane,
                self._connector._request_plane,
            )

        def produce_callable(request):
            async def return_callable(
                outputs: Dict[str, Any],
                parameters: Optional[Dict[str, Any]] = None,
                error: Optional[str] = None,
                final: Optional[bool] = False,
            ) -> None:
                # FIXME
                request_id = request.parameters["icp_request_id"].string_param

                infer_kwargs = {
                    "model": local_model,
                    "request_id": request_id,
                }
                if error is not None:
                    infer_kwargs["error"] = error
                else:
                    outputs_tensors = {}
                    for name, value in outputs.items():
                        outputs_tensors[name] = TritonTensor.from_dlpack(value)
                    infer_kwargs["outputs"] = outputs_tensors
                if final is not None:
                    infer_kwargs["final"] = final
                if parameters is not None:
                    infer_kwargs["parameters"] = parameters
                local_response = TritonInferenceResponse(**infer_kwargs)
                remote_response = RemoteInferenceResponse.from_local_response(
                    local_response,
                ).to_model_infer_response(self._connector._data_plane)
                # FIXME: This is a WAR for scenario where connector isn't
                # connected when posting a response to request plane.
                if not self._connector._connected:
                    await self.connect()

                await self._connector._request_plane.post_response(
                    request,
                    remote_response,
                )

            return return_callable

        return_callable = produce_callable(request)
        inputs = {}
        for name, input_tensor in remote_request.inputs.items():
            local_tensor = input_tensor.local_tensor
            numpy_tensor = np.from_dlpack(local_tensor)
            input_tensor.__del__()
            inputs[name] = numpy_tensor
        for key, value in remote_request.parameters.items():
            if isinstance(value, str) and value.startswith("JSON:"):
                remote_request.parameters[key] = json.loads(value[5:])
        return inputs, remote_request, return_callable
