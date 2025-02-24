import zmq
import msgspec
import time
import asyncio
from vllm import LLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.config import KVTransferConfig
from vllm.remote_prefill import RemotePrefillRequest, RemotePrefillParams, RemotePrefillResponse
from vllm.entrypoints.openai.api_server import build_async_engine_client_from_engine_args

_ctx = None
_socket = None


def init_zmq(hostname="localhost", port=5432):
    global _ctx, _socket
    _ctx = zmq.Context()
    _socket = _ctx.socket(zmq.PAIR)
    _socket.connect(f"tcp://{hostname}:{port}")


def get_remote_prefill_request_callback(socket):
    def callback(request: RemotePrefillRequest) -> RemotePrefillResponse:
        print(f"Main callback: Sending remote prefill request: {request}")
        socket.send(msgspec.msgpack.encode(request))
        response = msgspec.msgpack.decode(socket.recv(), type=RemotePrefillResponse)
        print(f"Main callback: Received remote prefill response: {response}")
        return response
    return callback


async def main():
    init_zmq()

    args = AsyncEngineArgs(
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



    async with build_async_engine_client_from_engine_args(args) as engine_client:

        # Send metadata to prefill
        metadata = engine_client.nixl_metadata
        assert metadata is not None
        print("[socket.send] Sending metadata to prefill")
        encoded_metadata = msgspec.msgpack.encode(metadata)
        _socket.send(encoded_metadata)
        print(f"Sent metadata to prefill")
        
        async for output in engine_client.generate(
            request_id="0",
            prompt="A B C D E",
            sampling_params=SamplingParams(max_tokens=5, temperature=0.0),
            remote_prefill_params=RemotePrefillParams(
                is_remote_prefill=True,
                remote_prefill_request_callback=get_remote_prefill_request_callback(_socket),
            )
        ):
            print(f"Output: {output.outputs[0].text}")

if __name__ == "__main__":
    asyncio.run(main())

