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

from triton_distributed.worker.log_formatter import setup_logger
from triton_distributed.worker.worker import Worker, WorkerConfig

LOGGER_NAME = __name__


class Deployment:
    def __init__(
        self, worker_configs: list[WorkerConfig | tuple[WorkerConfig, int]], log_level=3
    ):
        self._process_context = multiprocessing.get_context("spawn")
        self._worker_configs = worker_configs
        self._workers: list[multiprocessing.context.SpawnProcess] = []
        self._logger = setup_logger(log_level, LOGGER_NAME)

    @staticmethod
    def _start_worker(worker_config):
        Worker(worker_config).start()

    def start(self):
        for worker_config in self._worker_configs:
            worker_instances = 1
            if isinstance(worker_config, tuple):
                worker_instances = worker_config[1]
                worker_config = worker_config[0]
            worker_config.log_level = 6
            base_name = worker_config.name
            base_port = worker_config.metrics_port
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
            worker.terminate()
        if join:
            for worker in self._workers:
                worker.join(timeout)
            for worker in self._workers:
                if worker.is_alive():
                    worker.kill()
                    worker.join(timeout)
