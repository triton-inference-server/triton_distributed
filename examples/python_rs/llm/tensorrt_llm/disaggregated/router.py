# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import copy
from dataclasses import asdict

import uvloop
from common.protocol import DisaggregatedRequest, DisaggregatedResponse
from tensorrt_llm.llmapi import DisaggregatedParams
from tensorrt_llm.llmapi.disagg_utils import (
    CtxGenServerConfig,
    parse_disagg_config_file,
)
from tensorrt_llm.logger import logger

from triton_distributed.runtime import DistributedRuntime, triton_worker

logger.set_level("info")


class Router:
    def __init__(self, ctx_clients, gen_clients):
        self.ctx_server_idx = 0
        self.gen_server_idx = 0
        self.ctx_clients = ctx_clients
        self.gen_clients = gen_clients
        logger.info("INITIALIZED ROUTER")

    def get_next_server(self, servers, server_type):
        """Round-robin selection of next available server"""
        if not servers:
            raise ValueError(f"No {server_type} servers available")

        if server_type == "ctx":
            server = servers[self.ctx_server_idx]
            self.ctx_server_idx = (self.ctx_server_idx + 1) % len(servers)
        else:
            server = servers[self.gen_server_idx]
            self.gen_server_idx = (self.gen_server_idx + 1) % len(servers)

        return server

    async def generate(self, request):
        request = DisaggregatedRequest.parse_raw(request)
        gen_req = copy.deepcopy(request)

        # Pick a context server
        ctx_client = self.get_next_server(self.ctx_clients, "ctx")

        # Send request to context server
        request.disaggregated_params = asdict(
            DisaggregatedParams(request_type="context_only")
        )
        logger.debug(f"Sending request {request} to ctx server: {ctx_client}")

        request.sampling_params["max_tokens"] = 1
        ctx_resp = [
            resp async for resp in await ctx_client.generate(request.model_dump_json())
        ]
        if len(ctx_resp) > 1:
            raise ValueError(
                "Context server returned more than one response. This is currently not supported in disaggregated server."
            )

        ctx_resp_obj = DisaggregatedResponse.parse_raw(ctx_resp[0].data())
        if request.streaming:
            # NOTE: this might change in the future if trtllm context server returns raw tokens
            yield ctx_resp_obj.text

        gen_req.disaggregated_params = ctx_resp_obj.disaggregated_params
        gen_req.disaggregated_params.request_type = "generation_only"

        # Pick a generation server
        gen_client = self.get_next_server(self.gen_clients, "gen")
        logger.debug(f"Sending request {gen_req} to gen server: {gen_client}")

        async for response in await gen_client.generate(gen_req.model_dump_json()):
            yield response.data()


@triton_worker()
async def worker(runtime: DistributedRuntime, server_configs: list[CtxGenServerConfig]):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("triton-init").component("router")
    await component.create_service()

    ctx_clients = []
    gen_clients = []

    for i, server_config in enumerate(server_configs):
        if server_config.type == "ctx":
            ctx_client = (
                await runtime.namespace("triton-init")
                .component(f"tensorrt-llm-ctx-{i}")
                .endpoint("generate")
                .client()
            )
            ctx_clients.append(ctx_client)
        elif server_config.type == "gen":
            gen_client = (
                await runtime.namespace("triton-init")
                .component(f"tensorrt-llm-gen-{i}")
                .endpoint("generate")
                .client()
            )
            gen_clients.append(gen_client)
        else:
            raise ValueError(f"Invalid server type: {server_config.type}")

    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(Router(ctx_clients, gen_clients).generate)


if __name__ == "__main__":
    uvloop.install()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llmapi-disaggregated-config",
        "-c",
        type=str,
        default="disaggregated/llmapi_disaggregated_configs/single_node_config.yaml",
        help="Path to the llmapi disaggregated config file",
    )
    args = parser.parse_args()
    disagg_config = parse_disagg_config_file(args.llmapi_disaggregated_config)
    server_configs = disagg_config.server_configs

    asyncio.run(worker(server_configs))
