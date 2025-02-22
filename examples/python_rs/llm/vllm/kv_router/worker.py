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
from common.base_engine import BaseVllmEngine
from common.parser import parse_vllm_args
from common.protocol import MyRequestOutput, vLLMGenerateRequest
from triton_distributed_rs import DistributedRuntime, triton_endpoint, triton_worker
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import logger as vllm_logger
from vllm.sampling_params import RequestOutputKind

vllm_logger.info(f"VLLM_KV_CAPI_PATH: {os.environ['VLLM_KV_CAPI_PATH']}")


class VllmEngine(BaseVllmEngine):
    """
    vLLM Inference Engine
    """

    def __init__(self, engine_args: AsyncEngineArgs):
        self.engine_args = engine_args
        super().__init__(engine_args)

    @triton_endpoint(vLLMGenerateRequest, MyRequestOutput)
    async def generate(self, request) -> AsyncIterator:
        assert (
            self.engine_client is not None
        ), "engine_client was not initialized, must call initialize() first"

        sampling_params = request.sampling_params
        # rust HTTP requires Delta streaming
        sampling_params.output_kind = RequestOutputKind.DELTA

        async for response in self.engine_client.generate(
            request.engine_prompt, sampling_params, request.request_id
        ):
            # MyRequestOutput takes care of serializing the response as
            # vLLM's RequestOutput is not serializable by default
            yield MyRequestOutput(
                request_id=response.request_id,
                prompt=response.prompt,
                prompt_token_ids=response.prompt_token_ids,
                prompt_logprobs=response.prompt_logprobs,
                outputs=response.outputs,
                finished=response.finished,
            ).model_dump_json()


@triton_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    """
    Serve the triton-init.vllm.generate endpoint.
    """
    worker_component = runtime.namespace("triton-init").component("vllm")
    await worker_component.create_service()

    worker_endpoint = worker_component.endpoint("generate")

    VLLM_WORKER_ID = worker_endpoint.lease_id()
    os.environ["VLLM_WORKER_ID"] = str(VLLM_WORKER_ID)
    vllm_logger.info(f"Generate endpoint ID: {VLLM_WORKER_ID}")

    VLLM_KV_NAMESPACE = "router"
    os.environ["VLLM_KV_NAMESPACE"] = str(VLLM_KV_NAMESPACE)

    VLLM_KV_COMPONENT = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    os.environ["VLLM_KV_COMPONENT"] = str(VLLM_KV_COMPONENT)

    vllm_engine = VllmEngine(engine_args)
    await vllm_engine.initialize()

    await worker_endpoint.serve_endpoint(vllm_engine.generate)


if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()
    asyncio.run(worker(engine_args))
