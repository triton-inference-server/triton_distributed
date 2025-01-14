# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
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
#
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

import numpy
from triton_distributed.worker import Operator, RemoteInferenceRequest, RemoteOperator


class AddMultiplyDivide(Operator):
    def __init__(
        self,
        name,
        version,
        triton_core,
        request_plane,
        data_plane,
        parameters,
        repository,
        logger,
    ):
        self._triton_core = triton_core
        self._request_plane = request_plane
        self._data_plane = data_plane
        self._parameters = parameters
        self._add_model = RemoteOperator(
            "add", 1, self._request_plane, self._data_plane
        )
        self._multiply_model = RemoteOperator(
            "multiply", 1, self._request_plane, self._data_plane
        )
        self._divide_model = RemoteOperator(
            "divide", 1, self._request_plane, self._data_plane
        )

    async def execute(self, requests: list[RemoteInferenceRequest]):
        print("in execute!", flush=True)
        for request in requests:
            outputs = {}

            print(request.inputs, flush=True)
            array = None
            try:
                array = numpy.from_dlpack(request.inputs["int64_input"])
            except Exception as e:
                print(e)
            print(array)
            response = [
                response
                async for response in await self._add_model.async_infer(
                    inputs={"int64_input": array}
                )
            ][0]

            print(response, flush=True)

            for output_name, output_value in response.outputs.items():
                outputs[f"{response.model_name}_{output_name}"] = output_value

            addition_output_partial = response.outputs["int64_output_partial"]

            addition_output_total = response.outputs["int64_output_total"]

            multiply_respnoses = self._multiply_model.async_infer(
                inputs={"int64_input": addition_output_partial}, raise_on_error=False
            )

            divide_responses = self._divide_model.async_infer(
                inputs={
                    "int64_input": addition_output_partial,
                    "int64_input_divisor": addition_output_total,
                },
                raise_on_error=False,
            )

            error = None
            for result in asyncio.as_completed([multiply_respnoses, divide_responses]):
                responses = await result
                async for response in responses:
                    print("response!", response, flush=True)
                    print("error!", response.error, flush=True)
                    if response.error is not None:
                        error = response.error
                        break
                    for output_name, output_value in response.outputs.items():
                        outputs[f"{response.model_name}_{output_name}"] = output_value
            if error is not None:
                await request.response_sender().send(error=error, final=True)
            else:
                await request.response_sender().send(outputs=outputs, final=True)
            for output in outputs.values():
                del output
