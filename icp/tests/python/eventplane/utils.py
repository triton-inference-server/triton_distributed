import asyncio
import subprocess
import time
import uuid
from contextlib import asynccontextmanager

import pytest_asyncio

from triton_distributed.icp.eventplane_nats import EventPlaneNats


@pytest_asyncio.fixture(loop_scope="session")
async def nats_server():
    """Fixture to start and stop a NATS server."""
    try:
        # Start NATS server
        process = subprocess.Popen(
            ["nats-server", "-p", "4222"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(1)  # Allow the server time to start
        yield process
    finally:
        # Stop the NATS server
        process.terminate()
        process.wait()


@asynccontextmanager
async def event_plane_context():
    # with nats_server_context() as server:
    print(f"Print loop plane context: {id(asyncio.get_running_loop())}")
    server_url = "nats://localhost:4222"
    component_id = uuid.uuid4()
    plane = EventPlaneNats(server_url, component_id)
    await plane.connect()
    yield plane
    await plane.disconnect()


@pytest_asyncio.fixture(loop_scope="function")
async def event_plane():
    print(f"Print loop plane: {id(asyncio.get_running_loop())}")
    server_url = "nats://localhost:4222"
    component_id = uuid.uuid4()
    plane = EventPlaneNats(server_url, component_id)
    await plane.connect()
    yield plane
    await plane.disconnect()
