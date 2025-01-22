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


import numpy

from triton_distributed.worker import Operator, RemoteInferenceRequest


class Identity(Operator):
    """
    This is a dummy workflow that sends a single input as an output.
    """

    def __init__(
        self,
        name,
        version,
        triton_core,
        request_plane,
        data_plane,
        params,
        repository,
        logger,
    ):
        self._triton_core = triton_core
        self._request_plane = request_plane
        self._data_plane = data_plane
        self._params = params

    async def execute(self, requests: list[RemoteInferenceRequest]):
        for request in requests:
            try:
                array = numpy.from_dlpack(request.inputs["input"])
            except Exception as e:
                print(e)
                await request.response_sender().send(final=True, error=e)
                return

            outputs: dict[str, numpy.ndarray] = {"output": array}

            store_outputs_in_response = False

            if "store_outputs_in_response" in self._params:
                store_outputs_in_response = self._params["store_outputs_in_response"]

            store_outputs_in_response_set = set()

            if store_outputs_in_response:
                store_outputs_in_response_set.add("output")

            await request.response_sender().send(
                outputs=outputs,
                final=True,
                store_outputs_in_response=store_outputs_in_response_set,
            )
            for output in outputs.values():
                del output
