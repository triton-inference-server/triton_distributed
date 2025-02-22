import zmq
import time
from vllm import LLMEngine
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams, RemotePrefillParams
from vllm.config import KVTransferConfig

_ctx = None
_socket = None


def init_zmq(hostname="localhost", port=5432):
    global _ctx, _socket
    _ctx = zmq.Context()
    _socket = _ctx.socket(zmq.PAIR)
    _socket.connect(f"tcp://{hostname}:{port}")


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

    # Send metadata to prefill
    print("Sending metadata to prefill")
    _socket.send(agent_metadata)
    print(f"Sent metadata to prefill")

    engine.add_request(
        request_id="0",
        prompt="The capital of France is",
        params=SamplingParams(max_tokens=5),
        remote_prefill_params=RemotePrefillParams(
            is_remote_prefill=True,
        )
    )

    engine.add_request(
        request_id="1",
        prompt="The capital of Germany is",
        params=SamplingParams(max_tokens=5),
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
                print(request_outputs[0].outputs[0].text)

            time.sleep(1)   
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()

