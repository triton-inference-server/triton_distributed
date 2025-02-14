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

import uvloop
import vllm
from common.parser import parse_vllm_args
from common.protocol import PrefillRequest, PrefillResponse
from triton_distributed_rs import DistributedRuntime, triton_endpoint, triton_worker
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import logger as vllm_logger


class VllmPrefillEngine:
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine_args: AsyncEngineArgs):
        assert (
            engine_args.kv_transfer_config.is_kv_producer
        ), "Prefill worker must be a KV producer"
        self.engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)

    @triton_endpoint(PrefillRequest, PrefillResponse)
    async def generate(self, request):
        vllm_logger.info(f"Received prefill request: {request}")
        sampling_params = vllm.SamplingParams(**request.sampling_params)
        async for response in self.engine.generate(
            request.prompt, sampling_params, request.request_id
        ):
            vllm_logger.debug(f"Generated response: {response}")
            yield True


@triton_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("triton-init").component("prefill")
    await component.create_service()

    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(VllmPrefillEngine(engine_args).generate)


if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()
    asyncio.run(worker(engine_args))
