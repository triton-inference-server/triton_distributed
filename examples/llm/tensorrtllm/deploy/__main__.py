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
from llm.tensorrtllm.operators.kv_aware_routing import KvAwareRoutingOperator
from llm.tensorrtllm.scripts.gpu_info import get_gpu_product_name

from triton_distributed.worker import (
    OperatorConfig,
    TritonCoreOperator,
    Worker,
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
        Path(args.operator_repository) / "triton_core_models"
    )  # stores our simple pre/post processing
    return OperatorConfig(
        name=name,
        implementation=DisaggregatedServingOperator,
        max_inflight_requests=int(max_inflight_requests),
        repository=model_repository,
    )

def _create_kv_aware_routing_op(name, args, max_inflight_requests):
    model_repository = str(
        Path(args.operator_repository) / "triton_core_models"
    )  # stores our simple pre/post processing
    return OperatorConfig(
        name=name,
        implementation=KvAwareRoutingOperator,
        max_inflight_requests=int(max_inflight_requests),
        repository=model_repository,
    )

def _create_triton_core_op(
    name,
    max_inflight_requests,
    args,
):
    # TODO: argparse repo
    gpu_name = get_gpu_product_name()
    return OperatorConfig(
        name=name,
        implementation=TritonCoreOperator,
        max_inflight_requests=int(max_inflight_requests),
        repository=str(
            Path(args.operator_repository)
            / "tensorrtllm_models"
            / args.model
            / gpu_name
            / "TP_1"
        ),
        parameters={
            "store_outputs_in_response": True,
            "config": {
                "parameters": {
                    "participant_ids": {"string_value": f"{args.gpu_device_id}"},
                    "gpu_device_ids": {"string_value": f"{args.gpu_device_id}"},
                    "event_buffer_max_size": {"string_value": "1024"},
                }
            },
        },
    )


def main(args):
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(exist_ok=True)

    worker_configs = []

    if args.worker_type == "aggregate":
        aggregate_op = _create_triton_core_op(
            name=args.model, max_inflight_requests=1000, args=args
        )
        aggregate = WorkerConfig(
            operators=[aggregate_op],
            name=args.model,
            request_plane_args=([], {"request_plane_uri": args.request_plane_uri}),
            metrics_port=args.metrics_port,
        )
        worker_configs.append(aggregate)

    # Context/Generate workers used for Disaggregated Serving
    elif args.worker_type == "context":
        prefill_op = _create_triton_core_op(
            name="context",
            max_inflight_requests=1000,
            args=args,
        )

        prefill = WorkerConfig(
            operators=[prefill_op],
            name="context",
            log_level=args.log_level,
            metrics_port=args.metrics_port,
            request_plane_args=([], {"request_plane_uri": args.request_plane_uri}),
        )
        worker_configs.append(prefill)

    elif args.worker_type == "generate":
        decoder_op = _create_triton_core_op(
            name="generate",
            max_inflight_requests=1000,
            args=args,
        )

        decoder = WorkerConfig(
            operators=[decoder_op],
            name="generate",
            log_level=args.log_level,
            metrics_port=args.metrics_port,
            request_plane_args=([], {"request_plane_uri": args.request_plane_uri}),
        )
        worker_configs.append(decoder)

    elif args.worker_type == "disaggregated-serving":
        prefill_decode_op = _create_disaggregated_serving_op(
            name=args.model,
            max_inflight_requests=1000,
            args=args,
        )

        prefill_decode = WorkerConfig(
            operators=[prefill_decode_op],
            name=args.worker_name,
            log_level=args.log_level,
            metrics_port=args.metrics_port,
            request_plane_args=([], {"request_plane_uri": args.request_plane_uri}),
        )
        worker_configs.append(prefill_decode)
    elif args.worker_type == "kv-aware-routing":
        print("Creating KvAwareRouting Operator")
        router_op = _create_kv_aware_routing_op(
            name=args.model,
            max_inflight_requests=1000,
            args=args,
        )

        router = WorkerConfig(
            operators=[router_op],
            name=args.worker_name,
            log_level=args.log_level,
            metrics_port=args.metrics_port,
            request_plane_args=([], {"request_plane_uri": args.request_plane_uri}),
        )
        worker_configs.append(router)

    print("Starting Worker")
    for worker_config in worker_configs:
        worker = Worker(worker_config)
        print(f"worker: {worker}")
        worker.start()

    print("Worker started ... press Ctrl-C to Exit")

    while True:
        time.sleep(10)


if __name__ == "__main__":
    args = parse_args()
    main(args)
