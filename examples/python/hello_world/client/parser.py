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

from triton_distributed.worker.worker import DEFAULT_REQUESTS_URI


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Hello World Client")

    parser.add_argument(
        "--request-plane-uri",
        type=str,
        default=DEFAULT_REQUESTS_URI,
        help="URI of request plane",
    )

    parser.add_argument(
        "--requests-per-client",
        type=int,
        default=100,
        help="number of requests to send per client",
    )

    parser.add_argument(
        "--operator",
        type=str,
        choices=["encoder_decoder", "encoder", "decoder"],
        default="encoder_decoder",
        help="operator to send requests to. In this example all operators have the same input and output names.",
    )

    parser.add_argument(
        "--input-size-mean",
        type=int,
        default=1000,
        help="average input size for requests",
    )

    parser.add_argument(
        "--input-size-stdev",
        type=float,
        default=0,
        help="standard deviation for input size",
    )

    parser.add_argument(
        "--clients", type=int, default=1, help="number of concurrent clients to launch."
    )

    args = parser.parse_args(args)

    return args
