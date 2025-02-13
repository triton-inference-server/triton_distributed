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


def is_port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


@pytest_asyncio.fixture(loop_scope="session")
async def nats_server():
    """Fixture to start and stop a NATS server."""
    process = None
    try:
        # Raise more intuitive error to developer if port is already in-use.
        if is_port_in_use(DEFAULT_EVENTS_PORT):
            raise RuntimeError(
                f"ERROR: NATS Port {DEFAULT_EVENTS_PORT} already in use. Is a nats-server already running?"
            )

        # Start NATS server
        process = subprocess.Popen(
            [
                "nats-server",
                "-p",
                str(DEFAULT_EVENTS_PORT),
                "-addr",
                DEFAULT_EVENTS_HOST,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)  # Allow the server time to start
        yield process
    finally:
        # Stop the NATS server
        if process:
            process.terminate()
            process.wait()


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
