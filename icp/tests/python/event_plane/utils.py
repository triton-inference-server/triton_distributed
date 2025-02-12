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
import logging
import shutil
import subprocess
import time
import uuid
from contextlib import asynccontextmanager

import pytest_asyncio

from triton_distributed.icp.nats_event_plane import (
    DEFAULT_EVENTS_HOST,
    DEFAULT_EVENTS_PORT,
    DEFAULT_EVENTS_URI,
    NatsEventPlane,
)

logger = logging.getLogger(__name__)


def is_port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


class NatsServer:
    def __init__(
        self,
        port: int = DEFAULT_EVENTS_PORT,
        host: str = DEFAULT_EVENTS_HOST,
        store_dir: str = "/tmp/nats_store",
        debug: bool = False,
        clear_store: bool = True,
        dry_run: bool = False,
    ) -> None:
        self._process = None
        self.port = port
        self.url = f"nats://localhost:{port}"
        self.host = host
        command = [
            "/usr/local/bin/nats-server",
            "--jetstream",
            "--addr",
            DEFAULT_EVENTS_HOST,
            "--port",
            str(port),
            "--store_dir",
            store_dir,
        ]

        if debug:
            command.extend(["--debug", "--trace"])

        # Raise more intuitive error to developer if port is already in-use.
        if is_port_in_use(port):
            raise RuntimeError(
                f"ERROR: NATS Port {port} host {host} already in use. Is a nats-server already running?"
            )

        if clear_store:
            logger.info(f"Clearing store directory: {store_dir}")
            shutil.rmtree(store_dir, ignore_errors=True)

        logger.info(f"Running: [{' '.join(command)}]")
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
        )
        self._process = process

        while not is_port_in_use(port):
            logger.info(f"Waiting for NATS server to start on port {port} host {host}")
            time.sleep(1)
        logger.info(f"NATS server started on port {port} host {host}")

    def __del__(self):
        if self._process:
            logger.info(f"Terminating NATS server on port {self.port} host {self.host}")
            self._process.terminate()
            self._process.kill()
            self._process.wait()


@pytest_asyncio.fixture(loop_scope="session")
async def nats_server():
    """Fixture to start and stop a NATS server."""
    server = NatsServer(debug=True, clear_store=True)
    time.sleep(1)
    yield server
    server.__del__()


@asynccontextmanager
async def event_plane_context():
    # with nats_server_context() as server:
    print(f"Print loop plane context: {id(asyncio.get_running_loop())}")
    server_url = DEFAULT_EVENTS_URI
    component_id = uuid.uuid4()
    plane = NatsEventPlane(server_url, component_id)
    await plane.connect()
    yield plane
    await plane.disconnect()


@pytest_asyncio.fixture(loop_scope="function")
async def event_plane():
    print(f"Print loop plane: {id(asyncio.get_running_loop())}")
    server_url = DEFAULT_EVENTS_URI
    component_id = uuid.uuid4()
    plane = NatsEventPlane(server_url, component_id)
    await plane.connect()
    yield plane
    await plane.disconnect()
