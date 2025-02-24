import zmq
import msgspec
import asyncio
import uvloop
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.config import KVTransferConfig
from vllm.remote_prefill import RemotePrefillRequest, RemotePrefillParams, RemotePrefillResponse
from vllm.entrypoints.openai.api_server import build_async_engine_client_from_engine_args

from triton_distributed_rs import DistributedRuntime, triton_worker

from protocol import Request

class RequestHandler:
    def __init__(self, engine_client, socket):
        self.engine_client = engine_client
        self.socket = socket

        # Send metadata to prefill
        metadata = engine_client.nixl_metadata
        assert metadata is not None
        print("[socket.send] Sending metadata to prefill")
        encoded_metadata = msgspec.msgpack.encode(metadata)
        self.socket.send(encoded_metadata)
        print("RequestHandler initialized")

    def get_remote_prefill_request_callback(self):
        def callback(request: RemotePrefillRequest) -> RemotePrefillResponse:
            self.socket.send(msgspec.msgpack.encode(request))
            response = msgspec.msgpack.decode(self.socket.recv(), type=RemotePrefillResponse)
            # response = RemotePrefillResponse(
            #     request_id=request.request_id,
            #     first_token_id=435,
            # )
            return response
        return callback

    async def generate(self, raw_request):
        print("Got request")
        request: Request = msgspec.json.decode(raw_request.encode("utf-8"), type=Request)
        print(f"Request: {request}")

        if request.do_remote_prefill:
            remote_prefill_params = RemotePrefillParams(
                is_remote_prefill=True,
                remote_prefill_request_callback=self.get_remote_prefill_request_callback(),
            )
        else:
            remote_prefill_params = None

        async for output in self.engine_client.generate(
            request_id=request.request_id,
            prompt=request.prompt,
            sampling_params=request.sampling_params,
            remote_prefill_params=remote_prefill_params,
        ):
            yield output.outputs[0].text


# This is only used for metadata exchange between prefill and decode
# Should be replaced with etcd
def init_zmq(hostname="localhost", port=5432):
    global _ctx, _socket
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PAIR)
    socket.connect(f"tcp://{hostname}:{port}")

    return socket


@triton_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    socket = init_zmq()

    component = runtime.namespace("test-nixl").component("vllm")
    await component.create_service()

    endpoint = component.endpoint("generate")

    async with build_async_engine_client_from_engine_args(engine_args) as engine_client:

        await endpoint.serve_endpoint(RequestHandler(engine_client, socket).generate)

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

