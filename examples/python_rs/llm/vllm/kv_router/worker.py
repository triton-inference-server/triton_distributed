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
from common.protocol import Request, Response, TokenizedRequest, MyRequestOutput
from triton_distributed_rs import (
    DistributedRuntime,
    KvRouter,
    triton_endpoint,
    triton_worker,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.logger import logger as vllm_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

vllm_logger.info(f"VLLM_KV_CAPI_PATH: {os.environ['VLLM_KV_CAPI_PATH']}")


class VllmEngine:
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine_args: AsyncEngineArgs, router: KvRouter):
        self.engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        self.router = router
        self.tokenizer: Optional[AnyTokenizer] = None

    # Pattern to initialize async object as python __init__ is not async
    async def init(self):
        self.tokenizer = await self.engine.get_tokenizer()
        return self

    @triton_endpoint(TokenizedRequest, MyRequestOutput)
    async def generate_from_tokens(self, request):
        vllm_logger.info(f"Got request: {request}")
        tokens_prompt = TokensPrompt(prompt_token_ids=request.tokens)

        sampling_params = vllm.SamplingParams(**request.sampling_params)
        request_id = str(uuid.uuid4())

        vllm_logger.info(
            f"Running generate with engine_prompt: {tokens_prompt}, sampling_params: {sampling_params}, request_id: {request_id}"
        )

        response = self.engine.generate(tokens_prompt, sampling_params, request_id)


        async for resp in response:
            my_request_output = MyRequestOutput(
                request_id=request_id,
                prompt=resp.prompt,
                prompt_token_ids=resp.prompt_token_ids,
                prompt_logprobs=resp.prompt_logprobs,
                outputs=resp.outputs,
                finished=resp.finished
            )

            vllm_logger.info(f"MyRequestOutput: {my_request_output.json()}")
            yield my_request_output.json()
            
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
    worker_component = runtime.namespace("triton-init").component("vllm")
    await worker_component.create_service()

    preprocess_component = runtime.namespace("triton-init").component("preprocess")
    await preprocess_component.create_service()

    router_client = (
        await runtime.namespace("triton-init")
        .component("router")
        .endpoint("generate")
        .client()
    )

    worker_from_tokens_endpoint = worker_component.endpoint("generate_from_tokens")
    worker_from_prompt_endpoint = worker_component.endpoint("generate")
    preprocess_endpoint = preprocess_component.endpoint("generate")

    # TODO Hack until we unify lease_id and worker_id
    VLLM_WORKER_ID = uuid.UUID(int=worker_from_tokens_endpoint.lease_id())
    os.environ["VLLM_WORKER_ID"] = str(VLLM_WORKER_ID)
    vllm_logger.info(f"Generate endpoint ID: {VLLM_WORKER_ID}")

    vllm_engine = VllmEngine(engine_args, router_client)
    vllm_engine = await vllm_engine.init()

    await asyncio.gather(
        worker_from_tokens_endpoint.serve_endpoint(vllm_engine.generate_from_tokens),
        worker_from_prompt_endpoint.serve_endpoint(vllm_engine.generate_from_prompt),
        preprocess_endpoint.serve_endpoint(vllm_engine.preprocess),
    )


if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()
    asyncio.run(worker(engine_args))
