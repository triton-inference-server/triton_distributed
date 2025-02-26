import vllm
from vllm.engine.arg_utils import AsyncEngineArgs, KVTransferConfig
from common.base_engine import BaseVllmEngine
from vllm.logger import logger as vllm_logger

import os
from compoundai import nova_endpoint, service, server_context, async_onstart
from common.protocol import PrefillRequest, PrefillResponse

@service(
    nova={
        "enabled": True,
        "namespace": "triton-init",
    },
)
class Prefill(BaseVllmEngine):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        engine_args = AsyncEngineArgs(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            max_model_len=100,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            tensor_parallel_size=1,
            kv_transfer_config=KVTransferConfig(
                kv_connector="TritonNcclConnector",
                kv_role="kv_producer", 
                kv_rank=0,
                kv_parallel_size=2
            )
        )
        assert (
            engine_args.kv_transfer_config.is_kv_producer
        ), "Prefill worker must be a KV producer"
            
        super().__init__(engine_args)
        self.kv_transfer_config = engine_args.kv_transfer_config
        self.kv_rank = self.kv_transfer_config.kv_rank

    @async_onstart
    async def init_engine(self):
        await self.initialize()
        print("Prefill engine initialized")

    @nova_endpoint()
    async def generate(self, request: PrefillRequest):
        if self.engine_client is None:
            await self.initialize()

        vllm_logger.info(f"Received prefill request: {request}")
        sampling_params = vllm.SamplingParams(**request.sampling_params)
        vllm_logger.debug(f"Sampling params: {sampling_params}")
        if self.engine_client is None:
            raise RuntimeError("Engine client not initialized")
        else:
            async for response in self.engine_client.generate(
                request.prompt, sampling_params, request.request_id
            ):
                vllm_logger.debug(f"Generated response: {response}")
                yield True
