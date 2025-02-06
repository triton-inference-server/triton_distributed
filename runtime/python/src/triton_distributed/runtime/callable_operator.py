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

import msgspec

from triton_distributed.runtime import Operator, RemoteInferenceRequest


class CallableOperator(Operator):
    def __init__(
        self,
        callable_,
        *,
        name,
        version,
        request_plane,
        data_plane,
        parameters,
        repository,
        logger,
        triton_core,
    ):
        self._logger = logger
        self._callable

    async def call(self, prompt: str):
        print(prompt)
        return prompt

    async def execute(self, requests: list[RemoteInferenceRequest]):
        for request in requests:
            self._logger.info("got request!")
            print(request.inputs["args"])
            print(request.inputs["args"].to_bytes_array()[0])
            args = msgspec.msgpack.decode(
                request.inputs["args"].to_bytes_array()[0], type=list
            )
            kwargs = msgspec.msgpack.decode(
                request.inputs["kwargs"].to_bytes_array()[0], type=dict
            )
            result = await self.call(*args, **kwargs)
            await request.response_sender().send(
                outputs={"result": msgspec.msgpack.encode(result)}, final=True
            )
