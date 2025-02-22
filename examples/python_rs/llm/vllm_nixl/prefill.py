import zmq
from vllm import LLMEngine
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import RemotePrefillParams
from vllm.config import KVTransferConfig
from vllm.inputs.data import TokensPrompt
from vllm.outputs import RemotePrefillRequest
import msgspec
import json
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
    )
    vllm_config = args.create_engine_config()
    executor_class = LLMEngine._get_executor_cls(vllm_config)
    
    engine = LLMEngine(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_stats=False,
    )

    agent_metadata = engine.nixl_connector.get_agent_metadata()
    print(f"Agent metadata len: {len(agent_metadata)}")

    print("Engine created")
    
    # Recv metadata from decode
    print("Receiving metadata from decode")
    decode_meta = _socket.recv()
    print(f"Received metadata from decode")

    remote_agent_name = engine.nixl_connector.add_remote_agent(decode_meta)
    print(f"Added remote agent: {remote_agent_name}")

    iteration = 0
    while True:

        try:
            print(f"Iteration {iteration}")
            iteration += 1

            remote_prefill_requests = msgspec.msgpack.decode(_socket.recv(), type=list[RemotePrefillRequest])

            print(f"Received {len(remote_prefill_requests)} remote prefill requests from decode")

            for remote_prefill_request in remote_prefill_requests:
                sampling_params = remote_prefill_request.sampling_params
                sampling_params.max_tokens = 1
                sampling_params.min_tokens = 1
                engine.add_request(
                    request_id=remote_prefill_request.request_id,
                    prompt=TokensPrompt(prompt_token_ids=remote_prefill_request.prompt_token_ids),
                    params=sampling_params,
                    remote_prefill_params=RemotePrefillParams(
                        is_remote_decode=True,
                        decode_mem_desc=remote_prefill_request.memory_desc,
                        decode_agent_name=remote_prefill_request.agent_name,
                    )
                )

            prefill_outputs = engine.step()
            num_prefill_outputs = len(prefill_outputs)
            print(f"Prefilled {num_prefill_outputs} requests")


            remote_prefill_outputs = {}
            if num_prefill_outputs > 0:
                for output in prefill_outputs:
                    print(f"Output {output.request_id}")
                    print(output.outputs[0].text)
                    remote_prefill_outputs[output.request_id] = output.outputs[0].token_ids[0]

            _socket.send(json.dumps(remote_prefill_outputs).encode())
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()

