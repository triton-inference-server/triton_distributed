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

from llm.tensorrtllm.operators.disaggregated_serving import DisaggregatedServingOperator

from triton_distributed.worker import (
    Deployment,
    OperatorConfig,
    TritonCoreOperator,
    WorkerConfig,
)

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


def _create_disaggregated_serving_op(name, args, max_inflight_requests):
    model_repository = str(Path(args.operator_repository) / "triton_core_models")
    return OperatorConfig(
        name=name,
        implementation=DisaggregatedServingOperator,
        max_inflight_requests=int(max_inflight_requests),
        repository=model_repository,
    )


def _create_triton_core_op(
    name,
    max_inflight_requests,
    args,
):
    # TODO: argparse repo
    return OperatorConfig(
        name=name,
        implementation=TritonCoreOperator,
        max_inflight_requests=int(max_inflight_requests),
        repository=str(
            Path(args.operator_repository)
            / "tensorrtllm_models"
            / args.model
            / "NVIDIA_RTX_A6000"
            / "TP_1"
        ),
        parameters={"store_outputs_in_response": True},
    )


def main(args):
    global deployment

    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(exist_ok=True)

    worker_configs = []

    if args.aggregate_worker_count == 1:
        aggregate_op = _create_triton_core_op(
            name=args.model, max_inflight_requests=1000, args=args
        )
        aggregate = WorkerConfig(operators=[aggregate_op], name=args.model)
        worker_configs.append((aggregate, 1))

    # Context/Generate workers used for Disaggregated Serving
    if args.context_worker_count == 1:
        prefill_op = _create_triton_core_op(
            name="context",
            max_inflight_requests=1000,
            args=args,
        )

        prefill = WorkerConfig(
            operators=[prefill_op],
            # Context worker gets --worker-name as it is the model that will
            # be hit first in a disaggregated setting.
            name=args.worker_name,
        )
        worker_configs.append((prefill, 1))

    if args.generate_worker_count == 1:
        decoder_op = _create_triton_core_op(
            name="generate",
            max_inflight_requests=1000,
            args=args,
        )

        decoder = WorkerConfig(
            operators=[decoder_op],
            # Generate worker gets a hard-coded name "generate" as the context
            # worker will talk directly to it.
            name="generate",
        )
        worker_configs.append((decoder, 1))

        # Add the disaggregated serving operator when both workers are present
        # This coordinates between context and generate workers
        prefill_decode_op = _create_disaggregated_serving_op(
            name="mock",
            max_inflight_requests=1000,
            args=args,
        )

        prefill_decode = WorkerConfig(
            operators=[prefill_decode_op],
            name="mock",
        )
        worker_configs.append((prefill_decode, 1))

    print("Starting Workers")
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
