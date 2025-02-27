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
        self._loaded = False

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
            print("got request", request)

            # get meta data

            remote_metadata = self._metadata_store.get(request.engine_id)

            print("got metadata")
            #        print(remote_metadata,flush = True)
            print("got metadata_2")

            if not self._loaded:
                x = await self.engine_client.add_remote_nixl_metadata(remote_metadata)
                self._loaded = True
                print("loaded into engine client", x)

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
    print("hello!", flush=True)

    endpoint = component.endpoint("generate")

    metadata_store = NixlMetadataStore("test-nixl")

    async with build_async_engine_client_from_engine_args(engine_args) as engine_client:
        # This should be replaced with etcd
        metadata = engine_client.nixl_metadata

        metadata_store.put(metadata.engine_id, metadata)

        with temp_metadata_file(metadata.engine_id, metadata):
            # print(f"Waiting for remote metadata for engine {metadata.engine_id}")
            # remote_metadata = []
            # while not remote_metadata:
            #     await asyncio.sleep(1)
            #     remote_metadata = find_remote_metadata(metadata.engine_id)

            # print(f"Found {len(remote_metadata)} remote metadata for engine {metadata.engine_id}")
            # for remote_metadata in remote_metadata:
            #     await engine_client.add_remote_nixl_metadata(remote_metadata)
            await endpoint.serve_endpoint(
                RequestHandler(engine_client, metadata_store).generate
            )


if __name__ == "__main__":
    uvloop.install()
    engine_args = parse_vllm_args()

    engine_args.tensor_parallel_size = 1
    engine_args.max_model_len = 10
    engine_args.gpu_memory_utilization = 0.4

    if engine_args.enable_chunked_prefill is not False:
        print("Chunked prefill is not supported yet, setting to False")
        engine_args.enable_chunked_prefill = False

    if engine_args.pipeline_parallel_size != 1:
        print("Pipeline parallel size is not supported yet, setting to 1")
        engine_args.pipeline_parallel_size = 1

    asyncio.run(worker(engine_args))
