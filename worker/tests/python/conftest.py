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
import multiprocessing
import signal
import subprocess
import sys
import time

import pytest
import pytest_asyncio
from triton_distributed.icp.nats_request_plane import NatsServer
from triton_distributed.worker.log_formatter import LOGGER_NAME, setup_logger
from triton_distributed.worker.worker import Worker

logger = logging.getLogger(LOGGER_NAME)


NATS_PORT = 4223
TEST_API_SERVER_MODEL_REPO_PATH = (
    "/workspace/worker/python/tests/integration/api_server/models"
)


async def _wait_for_tasks(loop):
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        print("Encountered an error in task clean-up: %s", e)
    print("Stopping the event loop")
    loop.stop()


def _run_worker(name, queue, worker_config):
    tensor_store_keys = None
    try:
        with open(f"{name}.worker.stdout.log", "w") as output_:
            with open(f"{name}.worker.stderr.log", "w") as output_err:
                with open(f"{name}.worker.triton.log", "w"):
                    sys.stdout = output_
                    sys.stderr = output_err

                    triton_log_filename = f"{name}.worker.triton.log"
                    setup_logger(log_level=worker_config.log_level)
                    worker_config.triton_log_file = triton_log_filename
                    worker_config.name = name
                    try:
                        worker = Worker(worker_config)
                    except Exception as e:
                        queue.put(f"Failed to start {name}: {e}")
                        logger.exception("Failed to instantiate a worker class")

                    loop = asyncio.new_event_loop()
                    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
                    for sig in signals:
                        loop.add_signal_handler(
                            sig, lambda s=sig: asyncio.create_task(worker.shutdown(s))  # type: ignore
                        )

                    try:
                        queue.put("READY")
                        loop.run_until_complete(worker.serve())
                    except asyncio.CancelledError:
                        print("server cancellation detected")
                    finally:
                        loop.run_until_complete(_wait_for_tasks(loop))
                        loop.close()
                        tensor_store_keys = list(
                            worker._data_plane._tensor_store.keys()
                        )
                        sys.exit(len(tensor_store_keys))

    except Exception as e:
        print(f"Worker Serving Failed to start: {e}")
        queue.put(f"Failed to start {name}: {e}")
        raise e


class WorkerManager:
    ctx = multiprocessing.get_context("spawn")

    @staticmethod
    def setup_worker_process(operators, name, queue, worker_config):
        worker_config.name = name
        worker_config.operators = operators
        process = WorkerManager.ctx.Process(
            target=_run_worker,
            args=(name, queue, worker_config),
            name=name,
        )
        process.start()
        return process

    @staticmethod
    def cleanup_workers(workers, check_status=True):
        for worker in workers:
            print(f"Terminating {worker.name} worker", flush=True)
            worker.terminate()

        for worker in workers:
            worker.join()
            print(f"{worker.name} exited with {worker.exitcode} stored tensors")
            assert (
                worker.exitcode == 0 if check_status else True
            ), f"{worker.name} exited with {worker.exitcode} stored tensors"


@pytest.fixture
def worker_manager():
    return WorkerManager


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
