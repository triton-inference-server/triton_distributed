import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import logger as vllm_logger

from compoundai import depends, nova_endpoint, service, api
# from nova_init.decorators import nova_endpoint, nova_service, nova_depends

@service(
    nova={
        "enabled": True,
        "namespace": "triton-init",
    },
)
class Prefill:
    def __init__(self):
        engine_args = AsyncEngineArgs(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            max_model_len=100,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            kv_transfer_config={
                "kv_connector": "PyNcclConnector",
                "kv_role": "kv_producer", 
                "kv_rank": 0,
                "kv_parallel_size": 2
            }
        )
        # TODO: hacked right now
        assert (
            engine_args.kv_transfer_config["kv_role"] == "kv_producer"
        ), "Prefill worker must be a KV producer"
        self.engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)

    @nova_endpoint
    async def generate(self, request):
        print("prefill started")
        vllm_logger.info(f"Received prefill request: {request}")
        print("prefill request received", request)
        sampling_params = vllm.SamplingParams(**request["sampling_params"])
        async for response in self.engine.generate(
            request["prompt"], sampling_params, request["request_id"]
        ):
            vllm_logger.debug(f"Generated response: {response}")
            yield True
