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
import socket
import uuid

import msgspec
import uvloop
from common.base_engine import BaseVllmEngine
from common.parser import parse_vllm_args
from common.protocol import PrefillRequest
from triton_distributed_rs import DistributedRuntime, triton_worker
from triton_distributed_rs.prefill_queue import PrefillQueue
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import logger as vllm_logger


class VllmDecodeEngine(BaseVllmEngine):
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine_args: AsyncEngineArgs):
        assert (
            engine_args.kv_transfer_config.is_kv_consumer
        ), "Decode worker must be a KV consumer"
        if engine_args.enable_chunked_prefill is not False:
            vllm_logger.info(
                "Chunked prefill is not supported in disaggregated mode, disabling it"
            )
            engine_args.enable_chunked_prefill = False
        super().__init__(engine_args)

        self.kv_transfer_config = engine_args.create_engine_config().kv_transfer_config
        self.kv_rank = self.kv_transfer_config.kv_rank
        self.nats_server = os.getenv("NATS_SERVER", "nats://localhost:4222")

    async def generate(self, raw_request):
        if self.engine_client is None:
            await self.initialize()

        vllm_logger.debug(f"Got raw request: {raw_request}")
        (
            request,
            conversation,
            request_prompt,
            engine_prompt,
            sampling_params,
        ) = await self._parse_raw_request(raw_request)

        # TODO: pass decode info through a separate request param
        request_id = f"{uuid.uuid4()}___decode_hostname_{socket.gethostname()}___decode_kv_rank_{self.kv_rank}"

        prefill_sampling_params = {**msgspec.to_builtins(sampling_params)}
        prefill_sampling_params["max_tokens"] = 1
        prefill_sampling_params["min_tokens"] = 1
        prefill_request = PrefillRequest(
            prompt=request_prompt,  # TODO: we should use engine prompt to avoid extra tokenization
            sampling_params=prefill_sampling_params,
            request_id=request_id,
        )
        vllm_logger.debug(f"Prefill request: {prefill_request}")

        async with PrefillQueue.get_instance(nats_server=self.nats_server) as queue:
            await queue.enqueue_task(prefill_request.model_dump_json())

        vllm_logger.debug(
            f"Running generate with engine_prompt: {engine_prompt}, sampling_params: {sampling_params}, request_id: {request_id}"
        )
        if self.engine_client is None:
            raise RuntimeError("Engine client not initialized")
        else:
            generator = self.engine_client.generate(
                engine_prompt, sampling_params, request_id
            )

        async for response in await self._stream_response(
            request, generator, request_id, conversation
        ):
            vllm_logger.debug(f"Generated response: {response}")
            yield response


@triton_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("triton-init").component("vllm")
    await component.create_service()

    async with VllmDecodeEngine(engine_args) as decode_engine:
        endpoint = component.endpoint("generate")
        await endpoint.serve_endpoint(decode_engine.generate)


if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()
    asyncio.run(worker(engine_args))
