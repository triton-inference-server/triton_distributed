import vllm
from vllm.engine.arg_utils import AsyncEngineArgs, KVTransferConfig
from vllm.logger import logger as vllm_logger

import os
from compoundai import nova_endpoint, service

@service(
    nova={
        "enabled": True,
        "namespace": "triton-init",
    },
)
class Prefill:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        engine_args = AsyncEngineArgs(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            max_model_len=100,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            tensor_parallel_size=1,
            kv_transfer_config=KVTransferConfig(
                kv_connector="PyNcclConnector",
                kv_role="kv_producer", 
                kv_rank=0,
                kv_parallel_size=2
            )
        )
        assert (
            engine_args.kv_transfer_config.kv_role == "kv_producer"
        ), "Prefill worker must be a KV producer"
        self.engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        print(self.engine.__repr__())

    @nova_endpoint
    async def generate(self, request):
        vllm_logger.info(f"Received prefill request: {request}")
        sampling_params = vllm.SamplingParams(**request["sampling_params"])
        async for response in self.engine.generate(
            request["prompt"], sampling_params, request["request_id"]
        ):
            vllm_logger.debug(f"Generated response: {response}")
            yield True
