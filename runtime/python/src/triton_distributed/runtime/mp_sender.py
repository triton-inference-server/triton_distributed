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

from triton_distributed.icp.protos.icp_pb2 import ModelInferRequest
from triton_distributed.icp.request_plane import (
    set_icp_request_id,
    set_icp_response_to_uri,
)
from triton_distributed.runtime.remote_request import RemoteInferenceResponse
from triton_distributed.runtime.worker import WorkerConfig


class MPSender:
    def __init__(self, conn, model_name, model_version):
        self._conn = conn
        self._model_name = model_name
        self._model_version = model_version
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
        loop = asyncio.get_event_loop()
        data_avail = asyncio.Event()
        loop.add_reader(self._conn.fileno(), data_avail.set)
        # background_tasks = set()
        while not self._stop:
            while not self._conn.poll():
                await data_avail.wait()
                data_avail.clear()
            mp_response = self._conn.recv()
            await self._send_response(mp_response)
            # task = asyncio.create_task(self._send_response(mp_response))
            # background_tasks.add(task)
            # task.add_done_callback(background_tasks.discard)
        loop.remove_reader(self._conn.fileno())
        self._data_plane.close()
        await self._request_plane.close()

    async def _send_response(self, mp_response):
        if mp_response["final"]:
            remote_response = RemoteInferenceResponse(
                model_name=self._model_name,
                model_version=self._model_version,
                request_id=mp_response["request_id"],
                final=True,
            )
        else:
            remote_response = RemoteInferenceResponse(
                model_name=self._model_name,
                model_version=self._model_version,
                request_id=mp_response["request_id"],
                outputs={"result": [msgspec.msgpack.encode(mp_response["result"])]},
                store_outputs_in_response={"result"},
            )
        dummy_request = ModelInferRequest()
        set_icp_request_id(dummy_request, mp_response["icp_request_id"])
        set_icp_response_to_uri(dummy_request, mp_response["icp_response_to_uri"])
        await self._request_plane.post_response(
            dummy_request,
            remote_response.to_model_infer_response(self._data_plane),
        )
