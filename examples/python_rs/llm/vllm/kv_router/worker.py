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
from typing import AsyncIterator

import uvloop
import vllm
from common.parser import parse_vllm_args
from common.protocol import MyRequestOutput, vLLMGenerateRequest
from triton_distributed_rs import (
    DistributedRuntime,
    KvRouter,
    triton_endpoint,
    triton_worker,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import logger as vllm_logger
from vllm.sampling_params import RequestOutputKind
vllm_logger.info(f"VLLM_KV_CAPI_PATH: {os.environ['VLLM_KV_CAPI_PATH']}")


class VllmEngine:
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine_args: AsyncEngineArgs):
        self.engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        
            
    @triton_endpoint(vLLMGenerateRequest, MyRequestOutput)
    async def generate(self, request) -> AsyncIterator[MyRequestOutput]:

        sampling_params = request.sampling_params
        # rust HTTP requires Delta streaming
        sampling_params.output_kind = RequestOutputKind.DELTA

        async for response in self.engine.generate(
            request.engine_prompt, sampling_params, request.request_id
        ):
            vllm_logger.info(f"Response: {response}")
            my_request_output = MyRequestOutput(
                request_id=response.request_id,
                prompt=response.prompt,
                prompt_token_ids=response.prompt_token_ids,
                prompt_logprobs=response.prompt_logprobs,
                outputs=response.outputs,
                finished=response.finished,
            )
            vllm_logger.info(f"MyRequestOutput: {my_request_output.model_dump_json()}")
            yield my_request_output.model_dump_json()


@triton_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    worker_component = runtime.namespace("triton-init").component("vllm")
    await worker_component.create_service()

    # preprocess_component = runtime.namespace("triton-init").component("preprocess")
    # await preprocess_component.create_service()

    # router_client = (
    #     await runtime.namespace("triton-init")
    #     .component("router")
    #     .endpoint("generate")
    #     .client()
    # )

    # worker_from_tokens_endpoint = worker_component.endpoint("generate_from_tokens")
    worker_endpoint = worker_component.endpoint("generate")
    # preprocess_endpoint = preprocess_component.endpoint("generate")

    # TODO Hack until we unify lease_id and worker_id
    VLLM_WORKER_ID = uuid.UUID(int=worker_endpoint.lease_id())
    os.environ["VLLM_WORKER_ID"] = str(VLLM_WORKER_ID)
    vllm_logger.info(f"Generate endpoint ID: {VLLM_WORKER_ID}")

    vllm_engine = VllmEngine(engine_args)
    # vllm_engine = await vllm_engine.init()

    await asyncio.gather(
        # worker_from_tokens_endpoint.serve_endpoint(vllm_engine.generate_from_tokens),
        worker_endpoint.serve_endpoint(vllm_engine.generate),
        # preprocess_endpoint.serve_endpoint(vllm_engine.preprocess),
    )


if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()
    asyncio.run(worker(engine_args))
