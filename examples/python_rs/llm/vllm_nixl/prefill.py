import asyncio

import msgspec
import uvloop
from common import NixlMetadataStore, temp_metadata_file
from triton_distributed_rs import DistributedRuntime, triton_worker
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs.data import TokensPrompt
from vllm.remote_prefill import RemotePrefillParams, RemotePrefillRequest


class RequestHandler:
    def __init__(self, engine_client, metadata_store):
        self.engine_client = engine_client
        print("RequestHandler initialized")
        self._metadata_store = metadata_store
        self._loaded = False

    async def generate(self, raw_request: str):
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

    engine_args = AsyncEngineArgs(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        enforce_eager=True,
        kv_transfer_config=KVTransferConfig(kv_connector="TritonNixlConnector"),
        enable_chunked_prefill=False,  # TODO add support for chunked prefill
        disable_async_output_proc=True,  # TODO add support for async output processing
        preemption_mode="swap",  # TODO add support for recompute
        pipeline_parallel_size=1,  # TODO add support for pipeline parallel > 1
        tensor_parallel_size=1,
        max_model_len=10,
        gpu_memory_utilization=0.4,
    )

    asyncio.run(worker(engine_args))
