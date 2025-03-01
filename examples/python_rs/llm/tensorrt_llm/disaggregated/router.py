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
import uuid

import uvloop
from common.processor import ChatProcessor
from common.protocol import (
    ChatCompletionStreamResponse,
    DisaggChatCompletionRequest,
    DisaggregatedResponse,
    nvChatCompletionRequest,
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

logger.set_level("debug")


class Router:
    def __init__(self, ctx_client, gen_client):
        self.ctx_server_idx = 0
        self.gen_server_idx = 0
        self.ctx_client = ctx_client
        self.gen_client = gen_client
        logger.info("INITIALIZED ROUTER")
        self.chat_processor = ChatProcessor("disagg_router", None)

    @triton_endpoint(nvChatCompletionRequest, ChatCompletionStreamResponse)
    async def generate(self, request):
        # Send request to context serve
        # These settings are needed to satisfy request checks.
        request.skip_special_tokens = False
        request.add_special_tokens = False
        request.spaces_between_special_tokens = False
        request.id = str(uuid.uuid4())

        disaggregated_request = DisaggChatCompletionRequest(**request.model_dump())
        logger.debug(f"Received request {disaggregated_request}")

        gen_req = copy.deepcopy(disaggregated_request)

        disaggregated_request.max_tokens = 1
        disaggregated_request.disaggregated_params = DisaggregatedParams(
            request_type="context_only"
        )
        logger.debug(
            f"[router] Sending request to context server: {disaggregated_request}"
        )
        ctx_resp = [
            resp
            async for resp in await self.ctx_client.round_robin(
                disaggregated_request.model_dump_json()
            )
        ]
        if len(ctx_resp) > 1:
            raise ValueError(
                "Context server returned more than one response. This is currently not supported in disaggregated server."
            )

        ctx_resp_obj = DisaggregatedResponse.parse_raw(ctx_resp[0].data())
        logger.debug(f"[router] Got response from context server: {ctx_resp_obj}")
        if request.stream:
            # TODO: Return the first token and the rest of the tokens
            # are returned in the generation server.
            pass

        gen_req.disaggregated_params = ctx_resp_obj.disaggregated_params
        gen_req.disaggregated_params.request_type = "generation_only"

        logger.debug(f"[router] Sending request to generation server: {gen_req}")
        async for response in await self.gen_client.round_robin(
            gen_req.model_dump_json()
        ):
            logger.debug(f"[router] Got response from generation server: {response}")
            data = ChatCompletionStreamResponse.parse_raw(
                response.data()
            ).model_dump_json()
            yield data


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
