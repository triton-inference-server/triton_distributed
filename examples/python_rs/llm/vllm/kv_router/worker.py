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
import os
import uuid
from typing import Optional

import uvloop
import vllm
from common.parser import parse_vllm_args
from common.protocol import Request, Response, TokenizedRequest
from triton_distributed_rs import (
    DistributedRuntime,
    KvRouter,
    KvMetricsPublisher,
    triton_endpoint,
    triton_worker,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.logger import logger as vllm_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.engine.metrics_types import Stats, StatLoggerBase, SupportsMetricsInfo

vllm_logger.info(f"VLLM_KV_CAPI_PATH: {os.environ['VLLM_KV_CAPI_PATH']}")


class KvStatLogger(StatLoggerBase):
    def __init__(self, vllm_scheduler: vllm.core.scheduler.Scheduler, metrics_publisher: KvMetricsPublisher):
        # Must query initialized scheduler for max infos
        self.request_total_slots = vllm_scheduler.scheduler_config.max_num_seqs
        self.kv_total_blocks = vllm_scheduler.block_manager.num_total_gpu_blocks

        # KV metrics
        self.metrics_publisher = metrics_publisher
        self.metrics_publisher.publish(
            0,
            self.request_total_slots,
            0,
            self.kv_total_blocks,
        )


    def log(self, stats: Stats) -> None:
        self.metrics_publisher.publish(
            stats.num_running_sys,
            self.request_total_slots,
            int(stats.gpu_cache_usage_sys * self.kv_total_blocks),
            self.kv_total_blocks,
        )

    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        pass

class VllmEngine:
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine_args: AsyncEngineArgs, router: KvRouter, metrics_publisher: KvMetricsPublisher):
        self.engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        # Attach logger for continuous metrics publishing
        self.stat_logger = KvStatLogger(self.engine.engine.scheduler[0], metrics_publisher)
        self.engine.add_logger("kv_metrics", self.stat_logger)
        self.router = router
        self.tokenizer: Optional[AnyTokenizer] = None

    # Pattern to initialize async object as python __init__ is not async
    async def init(self):
        self.tokenizer = await self.engine.get_tokenizer()
        return self

    @triton_endpoint(TokenizedRequest, Response)
    async def generate_from_tokens(self, request):
        tokens_prompt = TokensPrompt(prompt_token_ids=request.tokens)

        sampling_params = vllm.SamplingParams(**request.sampling_params)
        request_id = str(uuid.uuid4())
        async for response in self.engine.generate(
            tokens_prompt, sampling_params, request_id
        ):
            yield response.outputs[0].text

    @triton_endpoint(Request, Response)
    async def generate_from_prompt(self, request):
        sampling_params = vllm.SamplingParams(**request.sampling_params)
        request_id = str(uuid.uuid4())
        async for response in self.engine.generate(
            request.prompt, sampling_params, request_id
        ):
            yield response.outputs[0].text

    @triton_endpoint(Request, Response)
    async def preprocess(self, request):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Must run init().")
        tokens = self.tokenizer.encode(request.prompt)

        engine_generator = await self.router.generate(
            TokenizedRequest(tokens=tokens, **request.model_dump()).model_dump_json()
        )

        async for resp in engine_generator:
            yield resp.data()


@triton_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    metrics_publisher = KvMetricsPublisher()
    worker_component = runtime.namespace("triton-init").component("vllm")
    await metrics_publisher.create_service(worker_component)

    preprocess_component = runtime.namespace("triton-init").component("preprocess")
    await preprocess_component.create_service()

    router_client = (
        await runtime.namespace("triton-init")
        .component("router")
        .endpoint("generate")
        .client()
    )

    worker_from_tokens_endpoint = worker_component.endpoint("generate_from_tokens")
    # worker_from_prompt_endpoint = worker_component.endpoint("generate")
    preprocess_endpoint = preprocess_component.endpoint("generate")

    # TODO Hack until we unify lease_id and worker_id
    VLLM_WORKER_ID = worker_from_tokens_endpoint.lease_id()
    os.environ["VLLM_WORKER_ID"] = str(VLLM_WORKER_ID)
    vllm_logger.info(f"Generate endpoint ID: {VLLM_WORKER_ID}")

    VLLM_KV_NAMESPACE = "triton-init"
    os.environ["VLLM_KV_NAMESPACE"] = str(VLLM_KV_NAMESPACE)

    VLLM_KV_COMPONENT = "vllm"
    os.environ["VLLM_KV_COMPONENT"] = str(VLLM_KV_COMPONENT)

    vllm_engine = VllmEngine(engine_args, router_client, metrics_publisher)
    vllm_engine = await vllm_engine.init()

    await asyncio.gather(
        worker_from_tokens_endpoint.serve_endpoint(vllm_engine.generate_from_tokens),
        # worker_from_prompt_endpoint.serve_endpoint(vllm_engine.generate_from_prompt),
        preprocess_endpoint.serve_endpoint(vllm_engine.preprocess),
    )


if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()
    asyncio.run(worker(engine_args))
