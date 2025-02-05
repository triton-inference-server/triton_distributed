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
import os
import pathlib
import signal
import sys
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Type

try:
    from tritonserver import ModelControlMode as ModelControlMode
    from tritonserver import Server as TritonCore

    from triton_distributed.runtime.triton_core_operator import TritonCoreOperator

    TRITON_CORE_AVAILABLE = True
except ImportError:
    TRITON_CORE_AVAILABLE = False
    TritonCoreOperator = type(None)
    TritonCore = type(None)  # type: ignore[misc,assignment]

from triton_distributed.icp.data_plane import DataPlane
from triton_distributed.icp.nats_request_plane import NatsRequestPlane
from triton_distributed.icp.request_plane import RequestPlane
from triton_distributed.icp.ucp_data_plane import UcpDataPlane
from triton_distributed.runtime.logger import get_logger, get_logger_config
from triton_distributed.runtime.operator import Operator, OperatorConfig
from triton_distributed.runtime.remote_request import (
    RemoteInferenceRequest,
    RemoteResponseSender,
)

if TYPE_CHECKING:
    import uvicorn

logger = get_logger(__name__)


@dataclass
class WorkerConfig:
    request_plane: Type[RequestPlane] = NatsRequestPlane
    data_plane: Type[DataPlane] = UcpDataPlane
    request_plane_args: tuple[list, dict] = field(default_factory=lambda: ([], {}))
    data_plane_args: tuple[list, dict] = field(default_factory=lambda: ([], {}))
    log_level: Optional[int] = None
    operators: list[OperatorConfig] = field(default_factory=list)
    name: str = str(uuid.uuid1())
    log_dir: Optional[str] = None
    consolidate_logs = False
    metrics_port: int = 0


class Worker:
    def __init__(
        self, config: Optional[WorkerConfig] = None, **kwargs  #: Unpack[WorkerConfig]
    ) -> None:
        if config is None:
            config = WorkerConfig(**kwargs)

        self._request_plane = config.request_plane(
            *config.request_plane_args[0], **config.request_plane_args[1]
        )

        self._data_plane = config.data_plane(
            *config.data_plane_args[0], **config.data_plane_args[1]
        )
        self._name = config.name
        self._log_level = config.log_level
        if self._log_level is None:
            self._log_level = 0
        self._operator_configs = config.operators
        self._log_dir = config.log_dir
        self._consolidate_logs = config.consolidate_logs
        self._stop_requested = False
        self._requests_received: Counter = Counter()
        self._background_tasks: dict[object, set] = {}
        self._completion_conds: dict[object, asyncio.Condition] = {}
        self._inflight_requests: dict[object, int] = {}
        self._max_inflght_requests: dict[object, int] = {}
        self._operators: dict[tuple[str, int], Operator] = {}
        self._metrics_port = config.metrics_port
        self._metrics_server: Optional[uvicorn.Server] = None
        self._component_id = self._request_plane.component_id
        self._triton_core: Optional[TritonCore] = None
        self._log_file: Optional[pathlib.Path] = None
        if self._log_dir:
            path = pathlib.Path(self._log_dir)
            path.mkdir(parents=True, exist_ok=True)
            pid = os.getpid()
            self._log_file = path / f"{self._name}.{self._component_id}.{pid}.log"

    def _import_operators(self):
        for operator_config in self._operator_configs:
            if operator_config.repository:
                repository_path = pathlib.Path(operator_config.repository)
                sys.path.append(str(repository_path.absolute()))
            else:
                repository_path = pathlib.Path(".")

            if isinstance(operator_config.implementation, str):
                split_workflow = operator_config.implementation.split(":")
                module_name = ":".join(split_workflow[:-1])
                class_name = split_workflow[-1]
                module_path = pathlib.Path(module_name)
                parent_paths = list(module_path.parents)
                root_parent = pathlib.Path(".")
                if parent_paths:
                    root_parent = parent_paths[-1]
                if root_parent == pathlib.Path("."):
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
                operator_logger = get_logger(
                    log_level=operator_config.log_level,
                    logger_name=f"OPERATOR{(operator_config.name,operator_config.version)}",
                    log_file=self._log_file,
                )

                if (
                    class_ == TritonCoreOperator
                    or issubclass(class_, TritonCoreOperator)
                ) and not self._triton_core:
                    if not TRITON_CORE_AVAILABLE:
                        raise ValueError(
                            "Please install Triton Core to use a Triton Core Operator"
                        )
                    if not self._consolidate_logs and self._log_file:
                        log_file = pathlib.Path(self._log_file)
                        stem = log_file.stem
                        suffix = log_file.suffix
                        triton_log_path = str(
                            log_file.parent / f"{stem}.triton{suffix}"
                        )
                    else:
                        triton_log_path = str(self._log_file)
                    self._triton_core = TritonCore(
                        model_repository=".",
                        log_error=True,
                        log_verbose=self._log_level,
                        strict_model_config=False,
                        model_control_mode=ModelControlMode.EXPLICIT,
                        log_file=triton_log_path,
                    ).start(wait_until_ready=True)

                operator = class_(
                    operator_config.name,
                    operator_config.version,
                    self._request_plane,
                    self._data_plane,
                    operator_config.parameters,
                    operator_config.repository,
                    operator_logger,
                    self._triton_core,
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
        logger.debug("\n\nserver received request: \n\n%s\n\n", request)

        operator_key = (request.model_name, int(request.model_version))

        if operator_key in self._operators:
            operator = self._operators[operator_key]
            self._requests_received[operator] += 1
            remote_request = RemoteInferenceRequest.from_model_infer_request(
                request, self._data_plane, self._request_plane
            )
            await operator.execute([remote_request])
        else:
            logger.warning("Received request for unknown operator")

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
        error = None
        try:
            self._import_operators()
            logger.info("Worker started...")
            await self._initialize_request_handlers()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("Encountered an error in worker: %s", e)
            self._stop_requested = True
            error = e
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
        return error

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
        if self._triton_core:
            self._triton_core.stop()
        if self._metrics_server:
            self._metrics_server.should_exit = True
            await self._metrics_server.shutdown()

    def _setup_metrics_server(self):
        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import PlainTextResponse

        app = FastAPI()
        log_config = get_logger_config(
            logger_name="uvicorn.error",
            log_level=self._log_level,
            log_file=self._log_file,
        )
        config = uvicorn.Config(
            app,
            port=self._metrics_port,
            log_level=self._log_level,
            log_config=log_config,
        )
        server = uvicorn.Server(config)

        @app.get("/metrics", response_class=PlainTextResponse)
        def metrics() -> str:
            if self._triton_core:
                return self._triton_core.metrics()
            else:
                return ""

        return server

    @staticmethod
    def exception_handler(loop, context):
        # get details of the exception
        exception = context["exception"]
        message = context["message"]
        # log exception
        logger.error(f"Task failed, msg={message}, exception={exception}")

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
        exit_condition = None
        logger = get_logger(log_level=self._log_level, log_file=self._log_file)
        logger.info(f"Starting Worker ==> {self._name}")
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(Worker.exception_handler)
        signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)

        # Note: mypy has known issues inferring
        # types of lambdas that include capturing loop variables

        for sig in signals:
            loop.add_signal_handler(
                sig, lambda s=sig: asyncio.create_task(self.shutdown(s))  # type: ignore
            )
        serve_result = None
        try:
            if self._metrics_port:
                serve_result = loop.create_task(self.serve())
                self._metrics_server = self._setup_metrics_server()
                assert self._metrics_server, "Unable to start metrics server"
                loop.run_until_complete(self._metrics_server.serve())
            else:
                serve_result = loop.run_until_complete(self.serve())
        except asyncio.CancelledError:
            logger.info("Worker cancelled!")
        finally:
            loop.run_until_complete(self._wait_for_tasks(loop))
            loop.close()
            logger.info("Successfully shutdown worker.")
            if isinstance(serve_result, asyncio.Task):
                exit_condition = serve_result.result()
            else:
                exit_condition = serve_result

        if exit_condition is not None:
            sys.exit(1)
        else:
            sys.exit(0)
