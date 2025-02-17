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
import subprocess
import time
from contextlib import asynccontextmanager

import pytest_asyncio

from triton_distributed.icp import (
    DEFAULT_EVENTS_HOST,
    DEFAULT_EVENTS_PORT,
    NatsEventPlane,
)

logger = logging.getLogger(__name__)


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
        logger.info("NATS server starting")
        process = subprocess.Popen(
            [
                "nats-server",
                "-p",
                str(DEFAULT_EVENTS_PORT),
                "-addr",
                DEFAULT_EVENTS_HOST,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        while not is_port_in_use(DEFAULT_EVENTS_PORT):
            logger.debug("Waiting for NATS server to start...")
            time.sleep(0.2)
        logger.info("NATS server started")
        yield process
    finally:
        # Stop the NATS server
        if process:
            logger.debug("Closing NATS server")

            process.terminate()
            # communicate() ensures we consume all stdout/stderr so they can close
            out, err = process.communicate()

            # If you want to log them:
            logger.debug("NATS server stdout: %s", out.decode())
            logger.debug("NATS server stderr: %s", err.decode())

            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()

            # Stop the NATS server
            process.wait()


@asynccontextmanager
async def event_plane_context():
    # with nats_server_context() as server:
    print(f"Print loop plane context: {id(asyncio.get_running_loop())}")
    plane = NatsEventPlane()
    await plane.connect()
    yield plane
    await plane.disconnect()


@pytest_asyncio.fixture(loop_scope="function")
async def event_plane():
    print(f"Print loop plane: {id(asyncio.get_running_loop())}")
    plane = NatsEventPlane()
    await plane.connect()
    yield plane
    await plane.disconnect()
