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
import importlib
import logging
import multiprocessing
import os
import pathlib
import signal
import sys
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Type

import tritonserver
from triton_distributed.icp.data_plane import DataPlane
from triton_distributed.icp.nats_request_plane import NatsRequestPlane
from triton_distributed.icp.request_plane import RequestPlane
from triton_distributed.icp.ucp_data_plane import UcpDataPlane
from triton_distributed.worker.log_formatter import LOGGER_NAME, setup_logger
from triton_distributed.worker.operator import Operator, OperatorConfig
from triton_distributed.worker.remote_request import (
    RemoteInferenceRequest,
    RemoteResponseSender,
)

logger = logging.getLogger(LOGGER_NAME)


class Worker:
    def __init__(
        self,
        request_plane: RequestPlane,
        data_plane: DataPlane,
        log_level: int,
        operators: list[OperatorConfig],
        triton_log_path: Optional[str] = None,
        name: str = str(uuid.uuid1()),
        metrics_port=0,
        log_dir=None,
    ):
        self._name = name
        self._request_plane = request_plane
        self._log_level = log_level
        self._data_plane = data_plane
        self._stop_requested = False
        self._triton_log_path = triton_log_path
        self._requests_received = Counter()
        self._background_tasks = {}
        self._completion_conds = {}
        self._inflight_requests = {}
        self._max_inflght_requests = {}
        self._operator_configs = operators
        self._operators = {}
        self._metrics_port = metrics_port
        self._log_dir = log_dir
        self._metrics_server = None

    def _import_operators(self):
        for operator_config in self._operator_configs:
            if operator_config.repository:
                repository_path = pathlib.Path(operator_config.repository)
                sys.path.append(str(repository_path.absolute()))
            else:
                repository_path = pathlib.Path(".")

            if isinstance(operator_config.implementation, str):
                split_workflow = operator_config.implementation.split(":")
                module = ":".join(split_workflow[:-1])
                class_name = split_workflow[-1]
                module_path = pathlib.Path(module)
                parent_paths = list(module_path.parents)
                root_parent = "."
                if parent_paths:
                    root_parent = parent_paths[-1]
                if root_parent == ".":
                    module_path = repository_path.joinpath(module_path)
                if str(module_path.parent.absolute()) not in sys.path:
                    sys.path.append(str(module_path.parent.absolute()))
                try:
                    module = importlib.import_module(module_path.name)
                    class_ = getattr(module, class_name)
                except Exception as e:
                    logger.exception(
                        "can't instantiate operator: %s %s", operator_config.name, e
                    )
                    raise e
            elif issubclass(operator_config.implementation, Operator):
                class_ = operator_config.implementation
            else:
                logger.exception(
                    "can't instantiate operator: %s",
                    operator_config.name,
                )
                raise Exception("invalid implementation type")

            try:
                if operator_config.log_level is None:
                    operator_config.log_level = self._log_level
                operator_logger = setup_logger(
                    log_level=operator_config.log_level,
                    logger_name=f"OPERATOR{(operator_config.name,operator_config.version)}",
                )
                operator = class_(
                    operator_config.name,
                    operator_config.version,
                    self._triton_core,
                    self._request_plane,
                    self._data_plane,
                    operator_config.parameters,
                    operator_config.repository,
                    operator_logger,
                )
            except Exception as e:
                logger.exception(
                    "can't instantiate operator: %s %s", operator_config.name, e
                )
                raise e

            operator_key = (operator_config.name, operator_config.version)
            self._operators[operator_key] = operator
            self._max_inflght_requests[operator] = operator_config.max_inflight_requests
            self._inflight_requests[operator] = 0
            self._background_tasks[operator] = set()
            self._completion_conds[operator] = asyncio.Condition()

    async def _process_request(self, request):
        logger.info("\n\nserver received request: \n\n%s\n\n", request)

        operator_key = (request.model_name, int(request.model_version))

        if operator_key in self._operators:
            operator = self._operators[operator_key]
            self._requests_received[operator] += 1
            remote_request = RemoteInferenceRequest.from_model_infer_request(
                request, self._data_plane, self._request_plane
            )
            await operator.execute([remote_request])
        else:
            logger.warn("Received request for unknown operator")

    async def _process_request_task(self, operator, name, version):
        requests = await self._request_plane.pull_requests(name, str(version))

        # When the request is received, notify the handler to
        # pull next requests if capacity permits.
        async with self._completion_conds[operator]:
            self._inflight_requests[operator] += 1
            logger.debug(f"{operator} inflight: {self._inflight_requests[operator]}")
            self._completion_conds[operator].notify()

        # Process request received from the request plane
        async for request in requests:
            await self._process_request(request)

        # The request is processed and new requests may be
        # pulled.
        async with self._completion_conds[operator]:
            self._inflight_requests[operator] -= 1
            logger.debug(f"{operator} inflight {self._inflight_requests[operator]}")
            self._completion_conds[operator].notify()

    async def _add_process_request_task(self, operator, name, version):
        task = asyncio.create_task(self._process_request_task(operator, name, version))
        self._background_tasks[operator].add(task)
        task.add_done_callback(self._background_tasks[operator].discard)

    async def _request_handler(self, operator, name, version):
        while not self._stop_requested:
            async with self._completion_conds[operator]:
                # TODO: Instead of pulling a fixed number of requests try
                # querying the model status to understand whether or not
                # to pull more requests.
                if (
                    self._inflight_requests[operator]
                    < self._max_inflght_requests[operator]
                ):
                    await self._add_process_request_task(operator, name, version)

                # Block the handler till task is notified
                # We want to create new tasks only when they
                # are needed so that at a given time, there
                # is only a single model task pulling from the
                # request plane.
                await self._completion_conds[operator].wait()

    async def _initialize_request_handlers(self):
        handlers = []
        for (name, version), operator in self._operators.items():
            logger.info(f"Starting {name} handler...")
            handlers.append(self._request_handler(operator, name, version))

        await asyncio.gather(*handlers)

    async def serve(self):
        self._triton_core = tritonserver.Server(
            model_repository=".",
            log_error=True,
            log_verbose=self._log_level,
            strict_model_config=False,
            model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
            log_file=self._triton_log_path,
        ).start(wait_until_ready=True)
        try:
            await self._request_plane.connect()
        except Exception as e:
            logger.exception(
                "Encountered an error when trying to connect to request plane"
            )
            raise e

        try:
            self._data_plane.connect()
        except Exception as e:
            logger.exception(
                "Encountered and error when trying to connect to data plane"
            )
            raise e

        try:
            self._import_operators()
            logger.info("Worker started...")
            await self._initialize_request_handlers()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("Encountered an error in worker: %s", e)
            self._stop_requested = True
        logger.info("worker store: %s", list(self._data_plane._tensor_store.keys()))
        logger.info("Worker stopped...")
        logger.info(
            "Hosted Operators: %s Requests Received: %s Responses Sent: %s",
            self._operators,
            self._requests_received,
            RemoteResponseSender.response_counts,
        )

        await self._request_plane.close()
        self._data_plane.close()
        if self._metrics_server:
            self._metrics_server.should_exit = True
            await self._metrics_server.shutdown()

    async def shutdown(self, signal):
        logger.info("Received exit signal %s...", signal.name)
        self._stop_requested = True
        try:
            if self._data_plane:
                self._data_plane.close()
        except Exception as e:
            logger.exception("Failed to close the data plane: %s", e)

        try:
            if self._request_plane:
                await self._request_plane.close()
        except Exception as e:
            logger.exception("Failed to close the request plane: %s", e)

        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        logger.info("Cancelling %s outstanding tasks", len(tasks))
        [task.cancel() for task in tasks]
        self._triton_core.stop()
        if self._metrics_server:
            self._metrics_server.should_exit = True
            await self._metrics_server.shutdown()

    def _setup_metrics_server(self):
        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import PlainTextResponse

        app = FastAPI()
        config = uvicorn.Config(app, port=self._metrics_port)
        server = uvicorn.Server(config)

        @app.get("/metrics", response_class=PlainTextResponse)
        def metrics() -> str:
            if self._triton_core:
                return self._triton_core.metrics()
            else:
                return ""

        return server

    async def _wait_for_tasks(self, loop):
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError as e:
            logger.exception("Cancelled in task clean-up: %s", e)
        except Exception as e:
            logger.exception("Encountered an error in task clean-up: %s", e)
        logger.info("Stopping the event loop")
        loop.stop()

    def start(self):
        if self._log_dir:
            os.makedirs(self._log_dir, exist_ok=True)
            stdout_path = os.path.join(self._log_dir, f"{self._name}.stdout.log")
            stderr_path = os.path.join(self._log_dir, f"{self._name}.stderr.log")
            if not self._triton_log_path:
                self._triton_log_path = os.path.join(
                    self._log_dir, f"{self._name}.triton.log"
                )
            sys.stdout = open(stdout_path, "w", buffering=1)
            sys.stderr = open(stderr_path, "w", buffering=1)
            triton_log = open(self._triton_log_path, "w", buffering=1)
            triton_log.close()
        setup_logger(log_level=self._log_level)
        loop = asyncio.get_event_loop()
        signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
        for sig in signals:
            loop.add_signal_handler(
                sig, lambda s=sig: asyncio.create_task(self.shutdown(s))
            )
        try:
            if self._metrics_port:
                loop.create_task(self.serve())
                self._metrics_server = self._setup_metrics_server()
                loop.run_until_complete(self._metrics_server.serve())
            else:
                loop.run_until_complete(self.serve())
        except asyncio.CancelledError:
            pass
            logger.info("Worker cancelled!")
        finally:
            loop.run_until_complete(self._wait_for_tasks(loop))
            loop.close()
            logger.info("Successfully shutdown worker.")
            sys.stdout.flush()
            sys.stderr.flush()
            if self._log_dir:
                sys.stdout.close()
                sys.stderr.close()


@dataclass
class WorkerProcess:
    request_plane: Type[RequestPlane] = NatsRequestPlane
    data_plane: Type[DataPlane] = UcpDataPlane
    request_plane_args: tuple[list, dict] = field(default_factory=lambda: ([], {}))
    data_plane_args: tuple[list, dict] = field(default_factory=lambda: ([], {}))
    log_level: int = 0
    operators: list[OperatorConfig] = field(default_factory=list)
    triton_log_path: Optional[str] = None
    name: str = str(uuid.uuid1())
    log_dir: Optional[str] = None
    metrics_port: int = 0
    _process: multiprocessing.Process = field(default=None, init=False, repr=False)

    _process_context = multiprocessing.get_context("spawn")

    def start(self):
        self._process = WorkerProcess._process_context.Process(
            target=self._start, name=self.name
        )
        self._process.start()
        return self._process

    def _start(self):
        self._create_worker().start()

    def __del__(self):
        self.shutdown(join=True)

    def shutdown(self, join=False):
        if self._process:
            self._process.terminate()
            if join:
                self._process.join()

    def _create_worker(self):
        return Worker(
            request_plane=self.request_plane(
                *self.request_plane_args[0], **self.request_plane_args[1]
            ),
            data_plane=self.data_plane(
                *self.data_plane_args[0], **self.data_plane_args[1]
            ),
            log_level=self.log_level,
            operators=self.operators,
            triton_log_path=self.triton_log_path,
            name=self.name,
            metrics_port=self.metrics_port,
            log_dir=self.log_dir,
        )
