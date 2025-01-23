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

from frontend.fastapi_frontend import FastApiFrontend
from llm.api_server.triton_distributed_engine import TritonDistributedEngine

from triton_distributed.worker.log_formatter import get_logger

from .parser import parse_args


def main(args):
    print(args)
    logger = get_logger(args.log_level, args.program_name)

    logger.info("Starting")

    # Wrap Triton Distributed in an interface-conforming "LLMEngine"
    engine: TritonDistributedEngine = TritonDistributedEngine(
        nats_url=args.request_plane_uri,
        data_plane_host=args.data_plane_host,
        data_plane_port=args.data_plane_port,
        model_name=args.model_name,
        tokenizer=args.tokenizer,
    )

    # Attach TritonLLMEngine as the backbone for inference and model management
    openai_frontend: FastApiFrontend = FastApiFrontend(
        engine=engine,
        host=args.api_server_host,
        port=args.api_server_port,
        log_level=args.log_level,
    )

    # Blocking call until killed or interrupted with SIGINT
    openai_frontend.start()


if __name__ == "__main__":
    parser, args = parse_args()
    args.program_name = parser.prog
    main(args)
