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
import uuid

import uvloop
import vllm
from protocol import PrefillRequest, Request, Response
from triton_distributed_rs import DistributedRuntime, triton_endpoint, triton_worker
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import logger as vllm_logger
from vllm.utils import FlexibleArgumentParser


class VllmDecodeEngine:
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine_args: AsyncEngineArgs, prefill):
        assert (
            engine_args.kv_transfer_config.is_kv_consumer
        ), "Decode worker must be a KV consumer"
        self.engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        self.prefill = prefill

    @triton_endpoint(Request, Response)
    async def generate(self, request):
        vllm_logger.info(f"Received request: {request}")
        sampling_params = vllm.SamplingParams(**request.sampling_params)
        request_id = str(uuid.uuid4())

        prefill_sampling_params = {**request.sampling_params}
        prefill_sampling_params["max_tokens"] = 1
        prefill_request = PrefillRequest(
            prompt=request.prompt,
            sampling_params=prefill_sampling_params,
            request_id=request_id,
        )
        prefill_generator = await self.prefill.generate(
            prefill_request.model_dump_json()
        )
        prefill_response = [resp async for resp in prefill_generator]
        assert len(prefill_response) == 1, "Prefill response should be a single boolean"
        prefill_response = prefill_response[0]
        print(prefill_response, prefill_response.data(), type(prefill_response.data()))
        assert prefill_response.data() is True, "Prefill should have been successful"
        vllm_logger.debug(f"Prefill response: {prefill_response}")

        async for response in self.engine.generate(
            request.prompt, sampling_params, request_id
        ):
            vllm_logger.debug(f"Generated response: {response}")
            yield response.outputs[0].text


@triton_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("triton-init").component("vllm")
    await component.create_service()

    prefill = (
        await runtime.namespace("triton-init")
        .component("prefill")
        .endpoint("generate")
        .client()
    )

    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(VllmDecodeEngine(engine_args, prefill).generate)


if __name__ == "__main__":
    uvloop.install()
    parser = FlexibleArgumentParser()
    parser = AsyncEngineArgs.add_cli_args(parser)
    engine_args = AsyncEngineArgs.from_cli_args(parser.parse_args())
    asyncio.run(worker(engine_args))
