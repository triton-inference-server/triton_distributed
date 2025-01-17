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


def _create_encoder_decoder_op(name, max_inflight_requests, args):
    return OperatorConfig(
        name=name,
        implementation="EncodeDecodeOperator",
        max_inflight_requests=int(max_inflight_requests),
        repository=args.operator_repository,
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
    return OperatorConfig(
        name=name,
        repository=args.triton_core_models,
        implementation=TritonCoreOperator,
        max_inflight_requests=int(max_inflight_requests),
        parameters={
            "config": {
                "instance_group": [
                    {"count": int(instances_per_worker), "kind": f"KIND_{kind}"}
                ],
                "parameters": {
                    "delay": {"string_value": f"{delay_per_token}"},
                    "input_copies": {"string_value": f"{input_copies}"},
                },
            }
        },
    )


async def main(args):
    global deployment
    log_dir = Path(args.log_dir)

    if args.clear_logs:
        shutil.rmtree(log_dir)

    log_dir.mkdir(exist_ok=True)

    (
        encoder_worker_instances,
        encoder_max_inflight_requests,
        encoder_instances_per_worker,
        encoder_device_kind,
    ) = args.encoders

    (
        decoder_worker_instances,
        decoder_max_inflight_requests,
        decoder_instances_per_worker,
        decoder_device_kind,
    ) = args.decoders

    (
        encoder_decoder_worker_instances,
        encoder_decoder_max_inflight_requests,
    ) = args.encoder_decoders

    encoder_op = _create_triton_core_op(
        name="encoder",
        max_inflight_requests=encoder_max_inflight_requests,
        instances_per_worker=encoder_instances_per_worker,
        kind=encoder_device_kind,
        delay_per_token=args.encoder_delay_per_token,
        input_copies=args.encoder_input_copies,
        args=args,
    )

    encoder = WorkerConfig(
        operators=[encoder_op],
        name="encoder",
    )

    decoder_op = _create_triton_core_op(
        name="decoder",
        max_inflight_requests=decoder_max_inflight_requests,
        instances_per_worker=decoder_instances_per_worker,
        kind=decoder_device_kind,
        delay_per_token=args.decoder_delay_per_token,
        input_copies=args.encoder_input_copies,
        args=args,
    )

    decoder = WorkerConfig(
        operators=[decoder_op],
        name="decoder",
    )

    encoder_decoder_op = _create_encoder_decoder_op(
        name="encoder_decoder",
        max_inflight_requests=encoder_decoder_max_inflight_requests,
        args=args,
    )

    encoder_decoder = WorkerConfig(
        operators=[encoder_decoder_op],
        name="encoder_decoder",
    )

    print("Starting Workers")

    deployment = Deployment(
        [
            # (worker_config, repeat_count )
            (encoder, int(encoder_decoder_worker_instances)),
            (decoder, int(decoder_worker_instances)),
            (encoder_decoder, int(encoder_decoder_worker_instances)),
        ],
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
    asyncio.run(main(args))
