import zmq
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

if __name__ == "__main__":
    main()

