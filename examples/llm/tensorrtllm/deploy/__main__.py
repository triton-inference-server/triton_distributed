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
    model_repository = str(
        Path(args.operator_repository)
        / "tensorrtllm_models"
        / "llama-3.1-8b-instruct"
        / "NVIDIA_H100_NVL"
        / "TP_1"
    )
    return OperatorConfig(
        name=name,
        implementation=DisaggregatedServingOperator,
        max_inflight_requests=int(max_inflight_requests),
        repository=model_repository,
    )


def _create_triton_core_op(
    name,
    max_inflight_requests,
    instances_per_worker,
    kind,
    delay_per_token,
    input_copies,
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
            / "llama-3.1-8b-instruct"
            / "NVIDIA_H100_NVL"
            / "TP_1"
        ),
    )


def main(args):
    global deployment

    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(exist_ok=True)

    # TODO: argparse these
    prefill_op = _create_triton_core_op(
        name="context",
        max_inflight_requests=1000,
        instances_per_worker=1,
        kind="GPU",
        delay_per_token=0,
        input_copies=1,
        args=args,
    )

    prefill = WorkerConfig(
        operators=[prefill_op],
        name="prefill",
    )

    # TODO: argparse these
    decoder_op = _create_triton_core_op(
        name="generate",
        max_inflight_requests=1000,
        instances_per_worker=1,
        kind="GPU",
        delay_per_token=0,
        input_copies=1,
        args=args,
    )

    decoder = WorkerConfig(
        operators=[decoder_op],
        name="decode",
    )

    # TODO: argparse these
    prefill_decode_op = _create_disaggregated_serving_op(
        name="prefill_decode",
        max_inflight_requests=1000,
        args=args,
    )

    prefill_decode = WorkerConfig(
        operators=[prefill_decode_op],
        name="prefill_decode",
    )

    worker_configs = [(prefill, 1), (decoder, 1), (prefill_decode, 1)]

    print("Starting Workers")

    deployment = Deployment(
        worker_configs,
        initialize_request_plane=args.initialize_request_plane,
        log_dir=args.log_dir,
        log_level=args.log_level,
        starting_metrics_port=args.starting_metrics_port,
    )

    deployment.start()
    print("Workers started ... press Ctrl-C to Exit")

    while True:
        time.sleep(10)


if __name__ == "__main__":
    args = parse_args()
    main(args)
