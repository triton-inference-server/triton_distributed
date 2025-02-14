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
import multiprocessing
from typing import Optional, get_type_hints

from triton_distributed.runtime.mp_receiver import MPReceiver
from triton_distributed.runtime.mp_sender import MPSender
from triton_distributed.runtime.worker import WorkerConfig


class MPWorker:
    def __init__(self, config: Optional[WorkerConfig] = None, **kwargs):
        self._operator_config = kwargs["operators"][0]
        self._model_name = self._operator_config.name
        self._model_version = self._operator_config.version
        self._callable = self._operator_config.parameters[
            "callable_object"
        ]  # TODO: other operator types?
        self._input_types = list(get_type_hints(self._callable).items())[0][1]
        self._cpu_count = multiprocessing.cpu_count()
        self._receivers_count = max(4, self._cpu_count // 4)
        self._receivers = []
        for i in range(self._receivers_count):
            recv_conn, send_conn = multiprocessing.Pipe(duplex=False)
            receiver = multiprocessing.Process(
                target=MPReceiver,
                args=(
                    send_conn,
                    self._model_name,
                    self._model_version,
                    self._input_types,
                ),
            )
            receiver.start()
            self._receivers.append((recv_conn, receiver))
        self._curr_sender_idx = 0
        self._senders_count = max(8, self._cpu_count // 2)
        self._senders = []
        for i in range(self._senders_count):
            recv_conn, send_conn = multiprocessing.Pipe(duplex=False)
            sender = multiprocessing.Process(
                target=MPSender, args=(recv_conn, self._model_name, self._model_version)
            )
            sender.start()
            self._senders.append((send_conn, sender))

    def start(self):
        self._stop = False  # TODO: shutdown?
        asyncio.run(self._run())

    async def _run(self):
        receiving_tasks = []
        for recv_conn, _ in self._receivers:
            task = asyncio.create_task(self._pull_request(recv_conn))
            receiving_tasks.append(task)
        await asyncio.gather(*receiving_tasks)

    async def _pull_request(self, recv_conn):
        loop = asyncio.get_event_loop()
        data_avail = asyncio.Event()
        loop.add_reader(recv_conn.fileno(), data_avail.set)
        background_tasks = set()
        while not self._stop:
            while not recv_conn.poll():
                await data_avail.wait()
                data_avail.clear()
            mp_request = recv_conn.recv()
            task = asyncio.create_task(self._process_request(mp_request))
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)
        loop.remove_reader(recv_conn.fileno())

    async def _process_request(self, mp_request):
        send_conn, _ = self._senders[self._curr_sender_idx]
        self._curr_sender_idx = (self._curr_sender_idx + 1) % len(self._senders)
        arg = mp_request["arg"]
        async for result in self._callable(arg):
            mp_response = {
                "icp_request_id": mp_request["icp_request_id"],
                "icp_response_to_uri": mp_request["icp_response_to_uri"],
                "request_id": mp_request["request_id"],
                "result": result,
                "final": False,
            }
            send_conn.send(mp_response)
        mp_response = {
            "icp_request_id": mp_request["icp_request_id"],
            "icp_response_to_uri": mp_request["icp_response_to_uri"],
            "request_id": mp_request["request_id"],
            "result": None,
            "final": True,
        }
        send_conn.send(mp_response)
