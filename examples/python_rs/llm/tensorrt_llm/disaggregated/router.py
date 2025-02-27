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
from common.protocol import (
    DisaggregatedRequest,
    DisaggregatedResponse,
    Request,
    Response,
)
from tensorrt_llm.llmapi import DisaggregatedParams
from tensorrt_llm.llmapi.disagg_utils import (
    CtxGenServerConfig,
    parse_disagg_config_file,
)
from tensorrt_llm.logger import logger

from triton_distributed.runtime import (
    DistributedRuntime,
    triton_endpoint,
    triton_worker,
)

logger.set_level("info")


class Router:
    def __init__(self, ctx_client, gen_client):
        self.ctx_server_idx = 0
        self.gen_server_idx = 0
        self.ctx_client = ctx_client
        self.gen_client = gen_client
        logger.info("INITIALIZED ROUTER")

    @triton_endpoint(Request, Response)
    async def generate(self, request):
        request = DisaggregatedRequest.parse_raw(request)
        gen_req = copy.deepcopy(request)

        # Send request to context server
        request.disaggregated_params = asdict(
            DisaggregatedParams(request_type="context_only")
        )
        request.sampling_params["max_tokens"] = 1
        ctx_resp = [
            resp
            async for resp in await self.ctx_client.round_robin(
                request.model_dump_json()
            )
        ]
        if len(ctx_resp) > 1:
            raise ValueError(
                "Context server returned more than one response. This is currently not supported in disaggregated server."
            )

        ctx_resp_obj = DisaggregatedResponse.parse_raw(ctx_resp[0].data())
        if request.streaming:
            # When streaming, the context server returns the first token and the rest of the tokens
            # are returned in the generation server. We are return the first token here to ensure
            # low TTFT
            # NOTE: this might change in the future if trtllm context server returns raw tokens
            yield ctx_resp_obj.text

        gen_req.disaggregated_params = ctx_resp_obj.disaggregated_params
        gen_req.disaggregated_params.request_type = "generation_only"

        async for response in await self.gen_client.round_robin(
            gen_req.model_dump_json()
        ):
            yield response.data()


@triton_worker()
async def worker(runtime: DistributedRuntime, server_configs: list[CtxGenServerConfig]):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("triton-init").component("router")
    await component.create_service()

    ctx_client = (
        await runtime.namespace("triton-init")
        .component("tensorrt-llm-ctx")
        .endpoint("generate")
        .client()
    )
    gen_client = (
        await runtime.namespace("triton-init")
        .component("tensorrt-llm-gen")
        .endpoint("generate")
        .client()
    )

    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(Router(ctx_client, gen_client).generate)


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
