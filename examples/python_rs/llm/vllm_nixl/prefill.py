import zmq
from vllm import LLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.config import KVTransferConfig
from vllm.inputs.data import TokensPrompt
from vllm.remote_prefill import RemotePrefillRequest, RemotePrefillParams, RemotePrefillResponse
from vllm.distributed.device_communicators.nixl import NixlMetadata
from vllm.entrypoints.openai.api_server import build_async_engine_client_from_engine_args
import msgspec
import asyncio
import uvloop


from triton_distributed_rs import DistributedRuntime, triton_worker


class RequestHandler:
    def __init__(self, engine_client):
        self.engine_client = engine_client
        print("RequestHandler initialized")
        

    async def generate(self, raw_request: str):
        print("Got request")
        request: RemotePrefillRequest = msgspec.json.decode(raw_request.encode("utf-8"), type=RemotePrefillRequest)
        print(f"Request: {request}")

        sampling_params = request.sampling_params
        sampling_params.max_tokens = 1
        sampling_params.min_tokens = 1
        
        remote_prefill_params = RemotePrefillParams(
            is_remote_decode=True,
            decode_block_ids=request.block_ids,
            decode_engine_id=request.engine_id,
        )

        # response = RemotePrefillResponse(
        #     request_id=request.request_id,
        #     first_token_id=435,
        # )

        # json_response = msgspec.json.encode(response).decode("utf-8")
        # yield json_response

        async for output in self.engine_client.generate(
            request_id=request.request_id,
            prompt=TokensPrompt(prompt_token_ids=request.prompt_token_ids),
            sampling_params=sampling_params,
            remote_prefill_params=remote_prefill_params,
        ):
            print(f"Output: {output.outputs[0].text}")
            remote_prefill_response = RemotePrefillResponse(
                request_id=output.request_id,
                first_token_id=output.outputs[0].token_ids[0],
            )
            json_response = msgspec.json.encode(remote_prefill_response).decode("utf-8")
            yield json_response


# This is only used for metadata exchange between prefill and decode
# Should be replaced with etcd
def init_zmq(hostname="localhost", port=5432):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PAIR)
    socket.bind(f"tcp://{hostname}:{port}")
    return socket


@triton_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    socket = init_zmq()

    component = runtime.namespace("test-nixl").component("prefill")
    await component.create_service()

    endpoint = component.endpoint("generate")

    async with build_async_engine_client_from_engine_args(engine_args) as engine_client:

        # This should be replaced with etcd
        print("[socket.recv] Receiving metadata from decode")
        msg = socket.recv()
        decode_meta = msgspec.msgpack.decode(msg, type=NixlMetadata)
        await engine_client.add_remote_nixl_metadata(decode_meta)
        print(f"Added remote metadata")

        await endpoint.serve_endpoint(RequestHandler(engine_client).generate)

if __name__ == "__main__":
    uvloop.install()

    engine_args = AsyncEngineArgs(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        enforce_eager=True,
        kv_transfer_config=KVTransferConfig(kv_connector="TritonNixlConnector"),
        enable_chunked_prefill=False, # TODO add support for chunked prefill
        disable_async_output_proc=True, # TODO add support for async output processing
        preemption_mode="swap", # TODO add support for recompute
        pipeline_parallel_size=1, # TODO add support for pipeline parallel > 1
        gpu_memory_utilization=0.25, # for dev to speed up mem registration
        max_model_len=100, # for dev to reduce required memory
        tensor_parallel_size=1,
    )

    asyncio.run(worker(engine_args))

