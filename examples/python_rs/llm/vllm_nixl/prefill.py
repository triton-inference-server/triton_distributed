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

import msgspec
import uvloop
from common import NixlMetadataStore, parse_vllm_args, temp_metadata_file
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs.data import TokensPrompt
from vllm.remote_prefill import RemotePrefillParams, RemotePrefillRequest

from triton_distributed.runtime import DistributedRuntime, triton_worker


class RequestHandler:
    def __init__(self, engine_client, metadata_store):
        self.engine_client = engine_client
        self._metadata_store = metadata_store
        print("RequestHandler initialized")
        self._loaded_metadata = {}

    async def generate(self, raw_request: str):
        try:
            request: RemotePrefillRequest = msgspec.json.decode(
                raw_request.encode("utf-8"), type=RemotePrefillRequest
            )

            sampling_params = request.sampling_params
            sampling_params.max_tokens = 1
            sampling_params.min_tokens = 1

            remote_prefill_params = RemotePrefillParams(
                is_remote_decode=True,
                decode_block_ids=request.block_ids,
                decode_engine_id=request.engine_id,
            )
            # get meta data

            if request.engine_id not in self._loaded_metadata:
                remote_metadata = self._metadata_store.get(request.engine_id)
                x = await self.engine_client.add_remote_nixl_metadata(remote_metadata)
                print(f"loaded metadata for {request.engine_id} into engine client {self.engine_client.nixl_metadata.engine_id}")
                self._loaded_metadata[request.engine_id] = remote_metadata

            async for _ in self.engine_client.generate(
                request_id=request.request_id,
                prompt=TokensPrompt(prompt_token_ids=request.prompt_token_ids),
                sampling_params=sampling_params,
                remote_prefill_params=remote_prefill_params,
            ):
                yield
        except Exception as e:
            print(e)
            yield str(e)


@triton_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    component = runtime.namespace("test-nixl").component("prefill")
    await component.create_service()

    endpoint = component.endpoint("generate")

    metadata_store = NixlMetadataStore("test-nixl")

    async with build_async_engine_client_from_engine_args(engine_args) as engine_client:

        metadata = engine_client.nixl_metadata

        metadata_store.put(metadata.engine_id, metadata)

        await endpoint.serve_endpoint(
            RequestHandler(engine_client, metadata_store).generate
        )


if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()

    engine_args.tensor_parallel_size = 2
    engine_args.max_model_len = 1000
    engine_args.gpu_memory_utilization = 0.7

    if engine_args.enable_chunked_prefill is not False:
        print("Chunked prefill is not supported yet, setting to False")
        engine_args.enable_chunked_prefill = False

    if engine_args.pipeline_parallel_size != 1:
        print("Pipeline parallel size is not supported yet, setting to 1")
        engine_args.pipeline_parallel_size = 1

    asyncio.run(worker(engine_args))
