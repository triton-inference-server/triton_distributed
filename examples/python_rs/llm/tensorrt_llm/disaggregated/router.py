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

import asyncio
import copy
from enum import Enum

import uvloop
from common.processor import ChatProcessor
from common.protocol import DisaggregatedResponse
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
    CompletionRequest,
    CompletionStreamResponse,
    DisaggregatedParams,
)

from triton_distributed.runtime import (
    DistributedRuntime,
    triton_endpoint,
    triton_worker,
)

logger.set_level("debug")


class RequestType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"


class Router:
    def __init__(
        self,
        ctx_chat_client,
        gen_chat_client,
        ctx_completion_client,
        gen_completion_client,
    ):
        self.ctx_chat_client = ctx_chat_client
        self.gen_chat_client = gen_chat_client
        self.ctx_completion_client = ctx_completion_client
        self.gen_completion_client = gen_completion_client
        logger.info("INITIALIZED ROUTER")
        self.chat_processor = ChatProcessor("disagg_router", None)

    async def generate(self, request, ctx_client, gen_client, request_type):
        # Send request to context serve
        # These settings are needed to satisfy request checks.
        request.skip_special_tokens = False
        request.add_special_tokens = False
        request.spaces_between_special_tokens = False

        logger.debug(f"Received request {request}")

        gen_req = copy.deepcopy(request)

        request.max_tokens = 1
        request.disaggregated_params = DisaggregatedParams(request_type="context_only")
        logger.debug(f"[router] Sending request to context server: {request}")
        ctx_resp = [
            resp
            async for resp in await ctx_client.round_robin(request.model_dump_json())
        ]
        if len(ctx_resp) > 1:
            raise ValueError(
                "Context server returned more than one response. This is currently not supported in disaggregated server."
            )

        ctx_resp_obj = DisaggregatedResponse.parse_raw(ctx_resp[0].data())
        logger.debug(f"[router] Got response from context server: {ctx_resp_obj}")
        if request.streaming:
            # TODO: Return the first token and the rest of the tokens
            # are returned in the generation server.
            pass

        gen_req.disaggregated_params = ctx_resp_obj.disaggregated_params
        gen_req.disaggregated_params.request_type = "generation_only"

        logger.debug(f"[router] Sending request to generation server: {gen_req}")
        async for response in await gen_client.round_robin(gen_req.model_dump_json()):
            logger.debug(f"[router] Got response from generation server: {response}")
            yield response

    @triton_endpoint(CompletionRequest, CompletionStreamResponse)
    async def generate_completion(self, request):
        yield self.generate(
            request,
            self.ctx_completion_client,
            self.gen_completion_client,
            RequestType.COMPLETION,
        )

    @triton_endpoint(ChatCompletionRequest, ChatCompletionStreamResponse)
    async def generate_chat(self, request):
        yield self.generate(
            request, self.ctx_chat_client, self.gen_chat_client, RequestType.CHAT
        )


@triton_worker()
async def worker(runtime: DistributedRuntime):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("triton-init").component("router")
    await component.create_service()

    ctx_completion_client = (
        await runtime.namespace("triton-init")
        .component("tensorrt-llm-ctx")
        .endpoint("completions")
        .client()
    )
    gen_completion_client = (
        await runtime.namespace("triton-init")
        .component("tensorrt-llm-gen")
        .endpoint("completions")
        .client()
    )

    completions_endpoint = component.endpoint("completions")
    router = Router(None, None, ctx_completion_client, gen_completion_client)
    await asyncio.gather(
        completions_endpoint.serve_endpoint(router.generate_completion),
    )


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
