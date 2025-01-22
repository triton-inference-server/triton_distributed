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

from .parser import parse_args


def main(args):
    print(args)
    # logging.basicConfig(level=args.log_level.upper(), format=args.log_format)

    # # Wrap Triton Distributed in an interface-conforming "LLMEngine"
    # engine: TritonDistributedEngine = TritonDistributedEngine(
    #     nats_url=args.nats_url,
    #     data_plane_host=args.data_plane_host,
    #     data_plane_port=args.data_plane_port,
    #     model_name=args.model_name,
    #     tokenizer=args.tokenizer,
    # )

    # # Attach TritonLLMEngine as the backbone for inference and model management
    # openai_frontend: FastApiFrontend = FastApiFrontend(
    #     engine=engine,
    #     host=args.api_server_host,
    #     port=args.api_server_port,
    #     log_level=args.log_level.lower(),
    # )

    # # Blocking call until killed or interrupted with SIGINT
    # openai_frontend.start()


if __name__ == "__main__":
    args = parse_args()
    main(args)
