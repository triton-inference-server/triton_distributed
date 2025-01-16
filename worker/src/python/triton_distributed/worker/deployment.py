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
import multiprocessing
from pprint import pformat
from typing import Optional, Type

from triton_distributed.icp import (
    DataPlane,
    NatsRequestPlane,
    NatsServer,
    RequestPlane,
    UcpDataPlane,
)
from triton_distributed.worker.log_formatter import setup_logger
from triton_distributed.worker.worker import Worker, WorkerConfig
from tritonserver import InvalidArgumentError

LOGGER_NAME = __name__


class Deployment:
    def __init__(
        self,
        worker_configs: list[WorkerConfig | tuple[WorkerConfig, int]],
        log_level=3,
        initialize_request_plane=False,
        initialize_data_plane=False,
        request_plane_args: Optional[tuple[list, dict]] = None,
        request_plane: Optional[Type[RequestPlane]] = NatsRequestPlane,
        data_plane: Optional[Type[DataPlane]] = UcpDataPlane,
        data_plane_args: Optional[tuple[list, dict]] = None,
        log_dir="logs",
        starting_metrics_port=50000,
    ):
        self._process_context = multiprocessing.get_context("spawn")
        self._worker_configs = worker_configs
        self._workers: list[multiprocessing.context.SpawnProcess] = []
        self._logger = setup_logger(log_level, LOGGER_NAME)
        self._default_request_plane = request_plane
        self._default_request_plane_args = request_plane_args
        self._default_data_plane = data_plane
        self._default_data_plane_args = data_plane_args
        self._initialize_request_plane = initialize_request_plane
        self._initialize_data_plane = initialize_data_plane
        self.request_plane_server: NatsServer = None
        self._default_log_dir = log_dir
        self._default_log_level = log_level
        self._starting_metrics_port = starting_metrics_port

    @staticmethod
    def _start_worker(worker_config):
        Worker(worker_config).start()

    def start(self):
        if self._initialize_request_plane:
            if self._default_request_plane == NatsRequestPlane:
                self.request_plane_server = NatsServer(log_dir=self._default_log_dir)
            else:
                raise InvalidArgumentError(
                    f"Unknown Request Plane Type, can not initialize {self._default_request_plane}"
                )

        for worker_config in self._worker_configs:
            worker_instances = 1
            if isinstance(worker_config, tuple):
                worker_instances = worker_config[1]
                worker_config = worker_config[0]

            base_name = worker_config.name
            base_port = worker_config.metrics_port

            if not base_port and self._starting_metrics_port:
                base_port = self._starting_metrics_port
                self._starting_metrics_port += worker_instances

            request_plane_args, request_plane_kwargs = worker_config.request_plane_args

            if not request_plane_args and not request_plane_kwargs:
                if self._default_request_plane_args:
                    worker_config.request_plane_args = self._default_request_plane_args
                elif self.request_plane_server:
                    worker_config.request_plane_args = (
                        [self.request_plane_server.url],
                        {},
                    )

            if not worker_config.log_dir:
                worker_config.log_dir = self._default_log_dir

            if not worker_config.log_level:
                worker_config.log_level = self._default_log_level

            for index in range(worker_instances):
                worker_config.name = f"{base_name}.{index}"
                worker_config.metrics_port = base_port + index
                self._workers.append(
                    self._process_context.Process(
                        target=Deployment._start_worker,
                        name=worker_config.name,
                        args=[worker_config],
                    )
                )
                self._logger.info(
                    "\n\nStarting Worker:\n\n\tConfig:\n\t%s\n\t%s\n",
                    pformat(worker_config),
                    self._workers[-1],
                )
                self._workers[-1].start()

    def stop(self):
        self.shutdown()

    def shutdown(self, join=True, timeout=10):
        for worker in self._workers:
            self._logger.info("\n\nStopping Worker:\n\n\n\t%s\n", worker)
            worker.terminate()
        if join:
            for worker in self._workers:
                worker.join(timeout)
            for worker in self._workers:
                if worker.is_alive():
                    worker.kill()
                worker.join(timeout)
                self._logger.info("\n\nWorker Stopped:\n\n\n\t%s\n", worker)
