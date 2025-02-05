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

from llm.api_server.triton_distributed_engine import DEFAULT_REQUESTS_URI


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenAI-Compatible API server.", prog="OpenAI API Sever"
    )

    # API Server
    parser.add_argument(
        "--api-server-host",
        type=str,
        required=False,
        default="127.0.0.1",
        help="API Server host",
    )

    parser.add_argument(
        "--api-server-port",
        type=int,
        required=False,
        default=8000,
        help="API Server port",
    )

    # Request Plane
    parser.add_argument(
        "--request-plane-uri",
        type=str,
        required=False,
        default=DEFAULT_REQUESTS_URI,
        help="URL of request plane",
    )

    # Data Plane
    parser.add_argument(
        "--data-plane-host",
        type=str,
        required=False,
        default=None,
        help="Data plane host",
    )

    parser.add_argument(
        "--data-plane-port",
        type=int,
        required=False,
        default=0,
        help="Data plane port. (default: 0 means the system will choose a port)",
    )

    # Misc
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=False,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Tokenizer to use for chat template in chat completions API",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=False,
        default="prefill",
        help="Model name",
    )

    parser.add_argument(
        "--log-level",
        type=int,
        required=False,
        default=1,
        help="Logging level",
    )

    parser.add_argument(
        "--backend",
        type=str,
        required=False,
        default="vllm",
        choices=["vllm", "tensorrtllm"],
        help="Backendtype",
    )

    return parser, parser.parse_args()
