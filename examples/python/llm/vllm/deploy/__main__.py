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

import signal
import sys
import time
from pathlib import Path

from llm.vllm.operators.vllm import (
    VllmContextOperator,
    VllmGenerateOperator,
    VllmOperator,
)

from triton_distributed.runtime import Deployment, OperatorConfig, WorkerConfig

from .parser import parse_args

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
        parameters=vars(args),
    )


def _create_generate_op(name, args, max_inflight_requests):
    return OperatorConfig(
        name=name,
        implementation=VllmGenerateOperator,
        max_inflight_requests=int(max_inflight_requests),
        parameters=vars(args),
    )


def _create_baseline_op(name, args, max_inflight_requests):
    return OperatorConfig(
        name=name,
        implementation=VllmOperator,
        max_inflight_requests=int(max_inflight_requests),
        parameters=vars(args),
    )


def main(args):
    global deployment

    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(exist_ok=True)

    worker_configs = []
    # Context/Generate workers used for Disaggregated Serving
    if args.context_worker_count == 1:
        context_op = _create_context_op(args.worker_name, args, 1000)
        context = WorkerConfig(
            operators=[context_op],
            # Context worker gets --worker-name as it is the model that will
            # be hit first in a disaggregated setting.
            name=args.worker_name,
        )
        worker_configs.append((context, 1))

    if args.generate_worker_count == 1:
        generate_op = _create_generate_op("generate", args, 1000)
        generate = WorkerConfig(
            operators=[generate_op],
            # Generate worker gets a hard-coded name "generate" as the context
            # worker will talk directly to it.
            name="generate",
        )
        worker_configs.append((generate, 1))

    # NOTE: Launching baseline worker and context/generate workers at
    # the same time is not currently supported.
    if args.baseline_worker_count == 1:
        # Baseline worker has a hard-coded name just for testing purposes
        baseline_op = _create_baseline_op("baseline", args, 1000)
        baseline = WorkerConfig(
            operators=[baseline_op],
            name="baseline",
        )
        worker_configs.append((baseline, 1))

    deployment = Deployment(
        worker_configs,
        initialize_request_plane=args.initialize_request_plane,
        log_dir=args.log_dir,
        log_level=args.log_level,
        starting_metrics_port=args.starting_metrics_port,
        request_plane_args=([], {"request_plane_uri": args.request_plane_uri}),
    )
    deployment.start()
    print("Workers started ... press Ctrl-C to Exit")

    while True:
        time.sleep(10)


if __name__ == "__main__":
    args = parse_args()
    main(args)
