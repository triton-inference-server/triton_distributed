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

import argparse
from pathlib import Path

from triton_distributed.worker.worker import DEFAULT_REQUESTS_URI


def parse_args(args=None):
    example_dir = Path(__file__).parent.absolute().parent.absolute()

    default_log_dir = example_dir.joinpath("logs")

    default_operator_repository = example_dir.joinpath("operators")

    default_triton_core_models = default_operator_repository.joinpath(
        "triton_core_models"
    )

    parser = argparse.ArgumentParser(description="Hello World Deployment")

    parser.add_argument(
        "--initialize-request-plane",
        default=False,
        action="store_true",
        help="Initialize the request plane, should only be done once per deployment",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(default_log_dir),
        help="log dir folder",
    )

    parser.add_argument(
        "--clear-logs", default=False, action="store_true", help="clear log directory"
    )

    parser.add_argument(
        "--log-level", type=int, default=1, help="log level applied to all workers"
    )

    parser.add_argument(
        "--request-plane-uri",
        type=str,
        default=DEFAULT_REQUESTS_URI,
        help="URI of request plane",
    )

    parser.add_argument(
        "--starting-metrics-port",
        type=int,
        default=50000,
        help="Metrics port for first worker. Each worker will expose metrics on subsequent ports, ex. worker 1: 50000, worker 2: 50001, worker 3: 50002",
    )

    parser.add_argument(
        "--operator-repository",
        type=str,
        default=str(default_operator_repository),
        help="operator repository",
    )

    parser.add_argument(
        "--triton-core-models",
        type=str,
        default=str(default_triton_core_models),
        help="model repository for triton core models.",
    )

    parser.add_argument(
        "--encoder-delay-per-token",
        type=float,
        default=0,
        help="Delay per input token. In this toy example can be used to vary the simulated compute load for encoding stage.",
    )

    parser.add_argument(
        "--encoder-input-copies",
        type=int,
        default=1,
        help="Number of copies of input to create during encoding. In this toy example can be used to vary the memory transferred between encoding and decoding stages.",
    )

    parser.add_argument(
        "--encoders",
        type=str,
        nargs=4,
        default=["1", "1", "1", "CPU"],
        help="Number of encoding workers to deploy. Specified as #Workers, #MaxInflightRequests, #ModelInstancesPerWorker, CPU || GPU",
    )

    parser.add_argument(
        "--decoders",
        type=str,
        nargs=4,
        default=["1", "1", "1", "CPU"],
        help="Number of decoding workers to deploy. Specified as #Workers, #MaxInflightRequests,#ModelInstancesPerWorker, CPU || GPU",
    )

    parser.add_argument(
        "--decoder-delay-per-token",
        type=float,
        default=0,
        help="Delay per input token. In this toy example can be used to vary the simulated compute load for decoding stage.",
    )

    parser.add_argument(
        "--encoder-decoders",
        type=str,
        nargs=2,
        default=["1", "1"],
        help="Number of encode-decode workers to deploy. Specified as #Worker, #MaxInflightRequests",
    )

    args = parser.parse_args(args)

    return args
