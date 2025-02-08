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
import os
import subprocess
import time
from pathlib import Path

import pytest
import pytest_asyncio

from triton_distributed.icp.nats_request_plane import NatsServer

logger = logging.getLogger(__name__)


NATS_PORT = 4223
TEST_API_SERVER_MODEL_REPO_PATH = "integration/api_server/models"


def pytest_addoption(parser):
    parser.addoption(
        "--basetemp-permissions",
        action="store",
        help="Permissions of the base temporary directory used by tmp_path, as octal value. Examples: 700 (default), 750, 770",
    )


@pytest.fixture(scope="session")
def log_dir(request, tmp_path_factory):
    log_dir = tmp_path_factory.mktemp("logs")
    try:
        permissions = request.config.getoption("--basetemp-permissions")
    except ValueError:
        permissions = False
    if permissions:
        basetemp = request.config._tmp_path_factory.getbasetemp()
        os.chmod(basetemp, int(permissions, 8))
        os.chmod(log_dir, int(permissions, 8))
    return log_dir


@pytest.fixture(scope="session")
def nats_server(log_dir):
    server = NatsServer(log_dir=log_dir / "nats")
    yield server
    del server


@pytest.fixture(scope="session")
def api_server(log_dir):
    command = [
        "tritonserver",
        "--model-store",
        str(Path(__file__).parent.resolve() / TEST_API_SERVER_MODEL_REPO_PATH),
    ]
    api_server_log_dir = log_dir / "api_server"
    os.makedirs(api_server_log_dir, exist_ok=True)
    with open(api_server_log_dir / "api_server.stdout.log", "wt") as output_:
        with open(api_server_log_dir / "api_server.stderr.log", "wt") as output_err:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=output_, stderr=output_err
            )
            time.sleep(10)
            yield process
            process.terminate()
            process.wait()
            print("Successfully cleaned-up T2 API server")


@pytest_asyncio.fixture
async def aio_benchmark(benchmark):
    async def run_async_coroutine(func, *args, **kwargs):
        return await func(*args, **kwargs)

    def _wrapper(func, *args, **kwargs):
        if asyncio.iscoroutinefunction(func):

            @benchmark
            def _():
                future = asyncio.ensure_future(
                    run_async_coroutine(func, *args, **kwargs)
                )
                return asyncio.get_event_loop().run_until_complete(future)

        else:
            benchmark(func, *args, **kwargs)

    return _wrapper
