import zmq
from vllm import LLMEngine
from vllm.engine.arg_utils import EngineArgs
from vllm.config import KVTransferConfig
from vllm.inputs.data import TokensPrompt
from vllm.remote_prefill import RemotePrefillRequest, RemotePrefillParams, RemotePrefillResponse
import msgspec
import time

_ctx = None
_socket = None


def init_zmq(hostname="localhost", port=5432):
    global _ctx, _socket
    _ctx = zmq.Context()
    _socket = _ctx.socket(zmq.PAIR)
    _socket.bind(f"tcp://{hostname}:{port}")


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
    
    # Recv metadata from decode
    print("[socket.recv] Receiving metadata from decode")
    msg = _socket.recv()
    # TODO add to NixlMetadata msgpack.Struct
    decode_meta = msgspec.msgpack.decode(msg, type=tuple[str, list[bytes], list[dict[tuple[int, int], bytes]]])
    print(f"Received metadata from decode")

    remote_agent_names = engine.add_remote_nixl_metadata(*decode_meta)
    print(f"Added remote agent: {remote_agent_names}")

    iteration = 0
    while True:

        try:
            print(f"Iteration {iteration}")
            iteration += 1

            try:
                print(f"[socket.recv] Prefill waiting for request from decode")
                msg = _socket.recv(zmq.NOBLOCK)
                remote_prefill_request = msgspec.msgpack.decode(msg, type=RemotePrefillRequest)

                print(f"Received remote prefill request from decode")

                # Sampling params can be adjusted inside the engine
                sampling_params = remote_prefill_request.sampling_params
                sampling_params.max_tokens = 1
                sampling_params.min_tokens = 1
                engine.add_request(
                    request_id=remote_prefill_request.request_id,
                    prompt=TokensPrompt(prompt_token_ids=remote_prefill_request.prompt_token_ids),
                    params=sampling_params,
                    remote_prefill_params=RemotePrefillParams(
                        is_remote_decode=True,
                        decode_block_ids=remote_prefill_request.block_ids,
                        decode_engine_id=remote_prefill_request.engine_id,
                    )
                )
            except zmq.Again:
                print("No request from decode")

            print(f"Prefilling")
            prefill_outputs = engine.step()
            num_prefill_outputs = len(prefill_outputs)
            print(f"Prefilled {num_prefill_outputs} requests")

            if num_prefill_outputs > 0:
                assert num_prefill_outputs == 1
                output = prefill_outputs[0]
                print(f"Output {output.request_id}")
                print(output.outputs[0].text)
                remote_prefill_response = RemotePrefillResponse(
                    request_id=output.request_id,
                    first_token_id=output.outputs[0].token_ids[0],
                )
                print(f"[socket.send] Prefill sending response to decode")
                _socket.send(msgspec.msgpack.encode(remote_prefill_response))

            time.sleep(1)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()

