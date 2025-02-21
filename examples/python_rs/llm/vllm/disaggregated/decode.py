import uuid
import random
import msgspec

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs, KVTransferConfig
from vllm.logger import logger as vllm_logger

# from nova_init.decorators import nova_endpoint, nova_service, nova_depends
import os
from disaggregated.prefill import Prefill
from compoundai import depends, nova_endpoint, service, api

from vllm.engine.arg_utils import AsyncEngineArgs
from common.base_engine import BaseVllmEngine
from common.protocol import PrefillRequest
from vllm.entrypoints.openai.protocol import ChatCompletionRequest

@service(
    nova={
        "enabled": True,
        "namespace": "triton-init",
    },
)
class Decode(BaseVllmEngine):
    prefill = depends(Prefill)

    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        engine_args = AsyncEngineArgs(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            max_model_len=100,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            tensor_parallel_size=1,
            kv_transfer_config=KVTransferConfig(
                kv_connector="PyNcclConnector",
                kv_role="kv_consumer",
                kv_rank=1,
                kv_parallel_size=2
            ),
        )
        assert (
            engine_args.kv_transfer_config.kv_role == "kv_consumer"
        ), "Decode worker must be a KV consumer"
        print("kicking off decode")
        super().__init__(engine_args)
        # self.prefills = []
        # self.num_prefill_workers = (
        #     self.engine.engine.vllm_config.kv_transfer_config.kv_producers_parallel_size
        # )
        self.kv_rank = self.engine.engine.vllm_config.kv_transfer_config.kv_rank
        print("decode engine initialized")
    
    def add_prefill(self, prefill):
        self.prefills.append(prefill)

    @nova_endpoint() 
    async def generate(self, raw_request: ChatCompletionRequest):
        vllm_logger.info(f"Received request: {raw_request}")
        (
            request,
            conversation,
            request_prompt,
            engine_prompt,
            sampling_params,
        ) = await self._parse_raw_request(raw_request)
        # prefill_rank = random.choice(range(self.num_prefill_workers))
        request_id = f"{uuid.uuid4()}___prefill_kv_rank_0___decode_kv_rank_{self.kv_rank}"

        prefill_sampling_params = {**msgspec.to_builtins(sampling_params)}
        prefill_sampling_params["max_tokens"] = 1
        prefill_request = PrefillRequest(
            prompt=request_prompt,
            sampling_params=prefill_sampling_params,
            request_id=request_id,
        )
        vllm_logger.debug(f"Prefill request: {prefill_request}")

        prefill_generator = self.prefill.generate(
            prefill_request.model_dump_json(),
        )
        prefill_resp = [resp async for resp in prefill_generator]
        vllm_logger.debug(f"Prefill response: {prefill_resp}")

        vllm_logger.debug(f"Running generate with engine_prompt: {engine_prompt}, sampling_params: {sampling_params}, request_id: {request_id}")
        generator = self.engine.generate(engine_prompt, sampling_params, request_id)
        async for response in await self._stream_response(
            request, generator, request_id, conversation
        ):
            vllm_logger.debug(f"Generated response: {response}")
            yield response