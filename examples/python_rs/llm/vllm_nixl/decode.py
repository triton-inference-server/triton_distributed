import zmq
import msgspec
import time
import json
from vllm import LLMEngine
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.config import KVTransferConfig
from vllm.remote_prefill import RemotePrefillRequest, RemotePrefillParams, RemotePrefillResponse

_ctx = None
_socket = None


def init_zmq(hostname="localhost", port=5432):
    global _ctx, _socket
    _ctx = zmq.Context()
    _socket = _ctx.socket(zmq.PAIR)
    _socket.connect(f"tcp://{hostname}:{port}")


def get_remote_prefill_request_callback(socket):
    def callback(request: RemotePrefillRequest) -> RemotePrefillResponse:
        socket.send(msgspec.msgpack.encode(request))
        response = msgspec.msgpack.decode(socket.recv(), type=RemotePrefillResponse)
        return response
    return callback


def main():
    init_zmq()

    args = EngineArgs(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        enforce_eager=True,
        kv_transfer_config=KVTransferConfig(
            kv_connector="TritonNixlConnector",
            kv_role="kv_consumer",
        ),
        enable_chunked_prefill=False, # TODO add support for chunked prefill
        disable_async_output_proc=True, # TODO add support for async output processing
        preemption_mode="swap", # TODO add support for recompute
        pipeline_parallel_size=1, # TODO add support for pipeline parallel > 1
        gpu_memory_utilization=0.25, # for dev to speed up mem registration
        max_model_len=100, # for dev to reduce required memory
        tensor_parallel_size=2,
    )
    vllm_config = args.create_engine_config()
    executor_class = LLMEngine._get_executor_cls(vllm_config)
    
    engine = LLMEngine(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_stats=False,
    )


    print("Engine created")

    # Send metadata to prefill
    metadata = engine.get_nixl_metadata()
    print("[socket.send] Sending metadata to prefill")
    encoded_metadata = msgspec.msgpack.encode(metadata)
    _socket.send(encoded_metadata)
    print(f"Sent metadata to prefill")

    engine.add_request(
        request_id="0",
        prompt="A B C D E",
        params=SamplingParams(max_tokens=15, temperature=0.0),
        remote_prefill_params=RemotePrefillParams(
            is_remote_prefill=True,
            remote_prefill_request_callback=get_remote_prefill_request_callback(_socket),
        )
    )

    engine.add_request(
        request_id="1",
        prompt="1 2 3 4 5",
        params=SamplingParams(max_tokens=15, temperature=0.0),
    )

    iteration = 0
    while True:

        try:
            print(f"Iteration {iteration}")
            iteration += 1

            request_outputs = engine.step()

            num_outputs = len(request_outputs)
            print(f"Num outputs: {num_outputs}")
            if num_outputs > 0:
                for output in request_outputs:
                    print(f"Output {output.request_id}")
                    print(output.outputs[0].text)

            time.sleep(1)   
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()

