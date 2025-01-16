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

import pytest
import pytest_asyncio
from triton_distributed.icp.nats_request_plane import NatsServer
from triton_distributed.worker.log_formatter import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


NATS_PORT = 4223
TEST_API_SERVER_MODEL_REPO_PATH = (
    "/workspace/worker/python/tests/integration/api_server/models"
)


@pytest.fixture(scope="session")
def nats_server():
    server = NatsServer()
    yield server
    del server


@pytest.fixture(scope="session")
def api_server():
    command = ["tritonserver", "--model-store", str(TEST_API_SERVER_MODEL_REPO_PATH)]
    with open("api_server.stdout.log", "wt") as output_:
        with open("api_server.stderr.log", "wt") as output_err:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=output_, stderr=output_err
            )
            time.sleep(5)
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
