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
import json
from typing import AsyncIterator

import uvloop
from common.base_engine import ChatProcessorMixin
from common.parser import LLMAPIConfig, parse_tensorrt_llm_args
from common.protocol import (
    DisaggChatCompletionRequest,
    DisaggChatCompletionStreamResponse,
    DisaggCompletionStreamResponse,
    RoutingStrategy,
    Tokens,
    WorkerId,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import CompletionRequest, DisaggregatedParams

from triton_distributed.llm import KvRouter
from triton_distributed.runtime import (
    DistributedRuntime,
    triton_endpoint,
    triton_worker,
)

logger.set_level("debug")


class KVRouterWrapper:
    def __init__(self, kv_router: KvRouter, routing_strategy: RoutingStrategy):
        self.kv_router = kv_router
        self.routing_strategy = routing_strategy

    @triton_endpoint(Tokens, WorkerId)
    async def generate(self, request) -> AsyncIterator[WorkerId]:
        lora_id = 0
        worker_id = None
        if self.routing_strategy == RoutingStrategy.PREFIX:
            try:
                worker_id = await self.kv_router.schedule(request.tokens, lora_id)
            except Exception as e:
                logger.exception(f"Error during worker selection: {e}")
                worker_id = ""

            logger.info(f"Scheduling to worker_id: {worker_id}")

            yield str(worker_id)

        else:
            raise NotImplementedError(
                f"Routing strategy {self.routing_strategy} not implemented"
            )


class Router(ChatProcessorMixin):
    def __init__(
        self,
        ctx_chat_client,
        gen_chat_client,
        ctx_completion_client,
        gen_completion_client,
        kv_router_wrapper: KVRouterWrapper,
        engine_config: LLMAPIConfig,
    ):
        self.ctx_chat_client = ctx_chat_client
        self.gen_chat_client = gen_chat_client
        self.ctx_completion_client = ctx_completion_client
        self.gen_completion_client = gen_completion_client
        self.kv_router_wrapper = kv_router_wrapper
        logger.info("INITIALIZED ROUTER")

        # allows to use tokenizer
        super().__init__(engine_config)

    async def _get_ctx_resp(self, request, ctx_client):
        logger.debug(f"Received request {request}")

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
        logger.debug(
            f"[router] received response from context server: {ctx_resp[0].data()}"
        )
        return ctx_resp[0].data()

    # TODO (shreyasm): The only reason we cant further combine the two methods below is
    # because the disagg params are in different locations.
    # Disagg params should be in under the choices field in the response object.
    # This is the case for completions but not for chat.

    @triton_endpoint(CompletionRequest, DisaggCompletionStreamResponse)
    async def generate_completion(self, request):
        # These settings are needed to satisfy request checks.
        request.skip_special_tokens = False
        request.add_special_tokens = False
        request.spaces_between_special_tokens = False

        gen_req = copy.deepcopy(request)

        ctx_resp = await self._get_ctx_resp(request, self.ctx_completion_client)
        ctx_resp_obj = DisaggCompletionStreamResponse.model_validate(ctx_resp)

        gen_req.disaggregated_params = DisaggregatedParams.model_validate(
            ctx_resp_obj.choices[0].disaggregated_params
        )
        gen_req.disaggregated_params.request_type = "generation_only"

        if request.stream:
            yield json.loads(
                ctx_resp_obj.model_dump_json(
                    exclude_unset=True, exclude={"disaggregated_params"}
                )
            )

        logger.debug(f"[router] Sending request to generation server: {gen_req}")
        async for response in await self.gen_completion_client.round_robin(
            gen_req.model_dump_json()
        ):
            gen_resp_obj = DisaggCompletionStreamResponse.model_validate(
                response.data()
            )
            yield json.loads(gen_resp_obj.model_dump_json(exclude_unset=True))

    @triton_endpoint(DisaggChatCompletionRequest, DisaggChatCompletionStreamResponse)
    async def generate_chat(self, request):
        # These settings are needed to satisfy request checks.
        request.skip_special_tokens = False
        request.add_special_tokens = False
        request.spaces_between_special_tokens = False

        gen_req = copy.deepcopy(request)

        ctx_resp = await self._get_ctx_resp(request, self.ctx_chat_client)
        ctx_resp_obj = DisaggChatCompletionStreamResponse.model_validate_json(ctx_resp)

        gen_req.disaggregated_params = DisaggregatedParams.model_validate(
            ctx_resp_obj.disaggregated_params
        )
        gen_req.disaggregated_params.request_type = "generation_only"

        if request.stream:
            yield json.loads(
                ctx_resp_obj.model_dump_json(
                    exclude_unset=True, exclude={"disaggregated_params"}
                )
            )

        logger.debug(f"[router] Sending request to generation server: {gen_req}")
        async for response in await self.gen_chat_client.round_robin(
            gen_req.model_dump_json()
        ):
            gen_resp_obj = DisaggChatCompletionStreamResponse.model_validate(
                response.data()
            )
            yield json.loads(gen_resp_obj.model_dump_json(exclude_unset=True))


@triton_worker()
async def worker(runtime: DistributedRuntime, args, engine_config):
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
    ctx_chat_client = (
        await runtime.namespace("triton-init")
        .component("tensorrt-llm-ctx")
        .endpoint("chat/completions")
        .client()
    )
    gen_chat_client = (
        await runtime.namespace("triton-init")
        .component("tensorrt-llm-gen")
        .endpoint("chat/completions")
        .client()
    )

    # Only listen to context server for now
    kv_listener = runtime.namespace("triton-init").component("tensorrt-llm-ctx")
    await kv_listener.create_service()

    kv_router = KvRouter(runtime, kv_listener)

    completions_endpoint = component.endpoint("completions")
    chat_endpoint = component.endpoint("chat/completions")

    kv_router_wrapper = KVRouterWrapper(kv_router, args.routing_strategy)
    router = Router(
        ctx_chat_client,
        gen_chat_client,
        ctx_completion_client,
        gen_completion_client,
        kv_router_wrapper,
        engine_config,
    )
    await asyncio.gather(
        completions_endpoint.serve_endpoint(router.generate_completion),
        chat_endpoint.serve_endpoint(router.generate_chat),
    )


if __name__ == "__main__":
    uvloop.install()
    args, engine_config = parse_tensorrt_llm_args()

    asyncio.run(worker(args, engine_config))
