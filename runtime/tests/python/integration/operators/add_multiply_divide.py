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

import asyncio

import numpy

from triton_distributed.runtime import Operator, RemoteInferenceRequest, RemoteOperator


class AddMultiplyDivide(Operator):
    def __init__(
        self,
        name,
        version,
        request_plane,
        data_plane,
        parameters,
        repository,
        logger,
        triton_core,
    ):
        self._triton_core = triton_core
        self._request_plane = request_plane
        self._data_plane = data_plane
        self._parameters = parameters
        self._add_model = RemoteOperator("add", self._request_plane, self._data_plane)
        self._multiply_model = RemoteOperator(
            "multiply", self._request_plane, self._data_plane
        )
        self._divide_model = RemoteOperator(
            "divide", self._request_plane, self._data_plane
        )
        self._logger = logger

    async def execute(self, requests: list[RemoteInferenceRequest]):
        self._logger.debug("in execute!")
        for request in requests:
            outputs = {}

            self._logger.debug(request.inputs)
            array = None
            try:
                array = numpy.from_dlpack(request.inputs["int64_input"])
            except Exception:
                self._logger.exception("Failed to retrieve inputs")
            self._logger.debug(array)
            response = [
                response
                async for response in await self._add_model.async_infer(
                    inputs={"int64_input": array}
                )
            ][0]

            self._logger.debug(response)

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
                    self._logger.debug(f"response! {response}")
                    self._logger.debug(f"error! {response.error}")
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
