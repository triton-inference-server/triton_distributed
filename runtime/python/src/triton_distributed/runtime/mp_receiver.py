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

import asyncio

import msgspec

from triton_distributed.icp.request_plane import (
    get_icp_request_id,
    get_icp_response_to_uri,
)
from triton_distributed.runtime.remote_request import RemoteInferenceRequest
from triton_distributed.runtime.worker import WorkerConfig


class MPReceiver:
    def __init__(self, conn, model_name, model_version, input_types):
        self._conn = conn
        self._model_name = model_name
        self._model_version = model_version
        self._input_types = input_types
        self._config = WorkerConfig()  # TODO: non default initialization?
        self._request_plane = self._config.request_plane(
            *self._config.request_plane_args[0], **self._config.request_plane_args[1]
        )
        self._data_plane = self._config.data_plane(
            *self._config.data_plane_args[0], **self._config.data_plane_args[1]
        )
        self._stop = False  # TODO: shutdown?
        asyncio.run(self._run())

    async def _run(self):
        await self._request_plane.connect()
        self._data_plane.connect()
        while not self._stop:
            requests = await self._request_plane.pull_requests(
                self._model_name, str(self._model_version)
            )
            async for request in requests:
                icp_request_id = get_icp_request_id(request)
                icp_response_to_uri = get_icp_response_to_uri(request)
                request_id = request.id
                remote_request = RemoteInferenceRequest.from_model_infer_request(
                    request, self._data_plane, self._request_plane
                )
                arg = msgspec.msgpack.decode(
                    remote_request.inputs["arg"].to_bytes_array()[0],
                    type=self._input_types,
                )
                mp_request = {
                    "icp_request_id": icp_request_id,
                    "icp_response_to_uri": icp_response_to_uri,
                    "request_id": request_id,
                    "arg": arg,
                }
                # print(f"{time.time_ns()}")
                self._conn.send(mp_request)
        self._data_plane.close()
        await self._request_plane.close()
