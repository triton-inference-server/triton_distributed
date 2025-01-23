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
import shutil
import signal
import sys
import time
from pathlib import Path

from ..operators.dummy import DummyOperator
from .args_utils import parse_args
from ..operators.vllm import VllmContextOperator, VllmGenerateOperator

from triton_distributed.worker import (
    Deployment,
    OperatorConfig,
    WorkerConfig,
)

deployment = None


def handler(signum, frame):
    exit_code = 0
    if deployment:
        print("Stopping Workers")
        exit_code = deployment.stop()
    print(f"Workers Stopped Exit Code {exit_code}")
    sys.exit(exit_code)


signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
for sig in signals:
    try:
        signal.signal(sig, handler)
    except Exception:
        pass


def _create_context_op(name, args, max_inflight_requests):
    return OperatorConfig(
        name=name,
        implementation=VllmContextOperator,
        max_inflight_requests=int(max_inflight_requests),
        parameters=vars(args)
    )


def _create_generate_op(name, args, max_inflight_requests):
    return OperatorConfig(
        name=name,
        implementation=VllmGenerateOperator,
        max_inflight_requests=int(max_inflight_requests),
        parameters=vars(args)
    )


def main(args):
    global deployment
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)

    worker_configs = []
    if args.generate_worker_count == 1:
        generate_op = _create_generate_op("generate", args, 1000)
        generate = WorkerConfig(
            operators=[generate_op],
            name="generate",

        )
        worker_configs.append((generate, 1))

    if args.context_worker_count == 1:
        context_op = _create_context_op("context", args, 1000)
        context = WorkerConfig(
            operators=[context_op],
            name="context",
        )
        worker_configs.append((context, 1))
    if args.dummy_worker_count == 1:
        dummy_op = OperatorConfig(
            name="dummy",
            implementation=DummyOperator,
            max_inflight_requests=1000,
            parameters=vars(args)
        )
        dummy = WorkerConfig(
            operators=[dummy_op],
            name="dummy",

        )
        worker_configs.append((dummy, 1))

    deployment = Deployment(
        worker_configs,
        initialize_request_plane=True,
        log_dir=args.log_dir,
        log_level=1,
        starting_metrics_port=args.starting_metrics_port
    )
    deployment.start()
    print("Workers started ... press Ctrl-C to Exit")

    while True:
        time.sleep(10)


if __name__ == "__main__":
    args = parse_args()
    main(args)
