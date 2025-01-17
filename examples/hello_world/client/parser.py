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


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Hello World Client")

    parser.add_argument(
        "--request-plane-uri", type=str, default="nats://localhost:4223"
    )

    parser.add_argument("--requests-per-client", type=int, default=100)

    parser.add_argument(
        "--operator",
        type=str,
        choices=["encoder_decoder", "encoder", "decoder"],
        default="encoder_decoder",
    )

    parser.add_argument("--input-size-mean", type=int, default=1000)

    parser.add_argument("--input-size-stdev", type=float, default=0)

    parser.add_argument("--clients", type=int, default=1)

    args = parser.parse_args(args)

    return args
