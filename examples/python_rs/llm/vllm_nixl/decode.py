import zmq
import msgspec
import json
import asyncio
import uvloop
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.config import KVTransferConfig
from vllm.remote_prefill import RemotePrefillRequest, RemotePrefillParams, RemotePrefillResponse
from vllm.entrypoints.openai.api_server import build_async_engine_client_from_engine_args
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.engine.multiprocessing.client import EngineClient
from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath

from triton_distributed_rs import DistributedRuntime, triton_worker, triton_endpoint

from protocol import Request

class RequestHandler:
    def __init__(self, engine_client: EngineClient, prefill_client):
        self.engine_client = engine_client
        self.prefill_client = prefill_client
        self.openai_serving_chat = None
        self.initialized = False
        print("RequestHandler initialized")

    async def init(self):
        models = OpenAIServingModels(
            engine_client=self.engine_client,
            model_config=await self.engine_client.get_model_config(),
            base_model_paths=[BaseModelPath(
                name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                model_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            )]
        )
        self.openai_serving_chat = OpenAIServingChat(
            engine_client=self.engine_client,
            model_config=await self.engine_client.get_model_config(),
            models=models,
            request_logger=None,
            response_role="assistant",
            chat_template=None,
            chat_template_content_format="auto",
        )
        self.initialized = True

    def get_remote_prefill_request_callback(self):
        async def callback(request: RemotePrefillRequest) -> RemotePrefillResponse:
            json_request = msgspec.json.encode(request).decode("utf-8")
            stream = await self.prefill_client.generate(json_request)
            async for response in stream:
                return msgspec.json.decode(response.data().encode("utf-8"), type=RemotePrefillResponse)
        return callback

    @triton_endpoint(ChatCompletionRequest, ChatCompletionStreamResponse)
    async def generate(self, request):

        if not self.initialized:
            await self.init()

        do_remote_prefill = True # TODO: this should be decided by the algorithm

        if do_remote_prefill:
            remote_prefill_params = RemotePrefillParams(
                is_remote_prefill=True,
                remote_prefill_request_callback=self.get_remote_prefill_request_callback(),
            )
        else:
            remote_prefill_params = None

        async for raw_response in await self.openai_serving_chat.create_chat_completion(
            request,
            remote_prefill_params=remote_prefill_params,
        ):
            if raw_response.startswith("data: [DONE]"):
                break
            response = json.loads(raw_response.lstrip("data: "))
            yield response
        


# This is only used for metadata exchange between prefill and decode
# Should be replaced with etcd
def init_zmq(hostname="localhost", port=5432):
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

    prefill_client = await runtime.namespace("test-nixl").component("prefill").endpoint("generate").client()

    async with build_async_engine_client_from_engine_args(engine_args) as engine_client:

        # This should be replaced with etcd
        metadata = engine_client.nixl_metadata
        assert metadata is not None
        print("[socket.send] Sending metadata to prefill")
        encoded_metadata = msgspec.msgpack.encode(metadata)
        socket.send(encoded_metadata)
        print("[socket.send] Sent metadata to prefill")

        await endpoint.serve_endpoint(RequestHandler(engine_client, prefill_client).generate)

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
        tensor_parallel_size=2,
    )

    asyncio.run(worker(engine_args))

