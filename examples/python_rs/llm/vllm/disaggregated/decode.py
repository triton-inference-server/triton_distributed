import uuid
import msgspec
import socket

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs, KVTransferConfig
from vllm.logger import logger as vllm_logger

# from nova_init.decorators import nova_endpoint, nova_service, nova_depends
import os
from disaggregated.prefill import Prefill
from compoundai import depends, nova_endpoint, service, api, async_onstart

from vllm.engine.arg_utils import AsyncEngineArgs
from common.base_engine import BaseVllmEngine
from common.chat_processor import ProcessMixIn
from common.protocol import PrefillRequest
from vllm.entrypoints.openai.protocol import ChatCompletionRequest

@service(
    nova={
        "enabled": True,
        "namespace": "triton-init",
    },
)
class Decode(BaseVllmEngine, ProcessMixIn):
    prefill = depends(Prefill) 

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
                kv_connector="TritonNcclConnector",
                kv_role="kv_consumer",
                kv_rank=1,
                kv_parallel_size=2
            ),
        )
        assert (
            engine_args.kv_transfer_config.kv_role == "kv_consumer"
        ), "Decode worker must be a KV consumer"
        
        super().__init__(engine_args)
        self.kv_transfer_config = engine_args.kv_transfer_config
        self.kv_rank = self.kv_transfer_config.kv_rank

    @async_onstart
    async def init_engine(self):
        await self.initialize()
    
    @nova_endpoint() 
    async def generate(self, raw_request: ChatCompletionRequest):
        if self.engine_client is None:
            await self.initialize()
            
        vllm_logger.info(f"Received request: {raw_request}")
        (
            request,
            conversation,
            request_prompt,
            engine_prompt,
            sampling_params,
        ) = await self._parse_raw_request(raw_request)

        # TODO: pass decode info through a separate request param
        request_id = f"{uuid.uuid4()}___decode_hostname_{socket.gethostname()}___decode_kv_rank_{self.kv_rank}"

        prefill_sampling_params = {**msgspec.to_builtins(sampling_params)}
        prefill_sampling_params["max_tokens"] = 1
        prefill_sampling_params["min_tokens"] = 1
        prefill_request = PrefillRequest(
            prompt=request_prompt, # TODO: we should use engine prompt to avoid extra tokenization
            sampling_params=prefill_sampling_params,
            request_id=request_id,
        )
        vllm_logger.debug(f"Prefill request: {prefill_request}")
        prefill_output = self.prefill.generate(
            prefill_request.model_dump_json(),
        )
        async for _ in prefill_output:
            pass

        vllm_logger.debug(
            f"Running generate with engine_prompt: {engine_prompt}, sampling_params: {sampling_params}, request_id: {request_id}"
        )
        if self.engine_client is None:
            raise RuntimeError("Engine client not initialized")
        else:
            generator = self.engine_client.generate(engine_prompt, sampling_params, request_id)

        async for response in await self._stream_response(
            request, generator, request_id, conversation
        ):
            vllm_logger.debug(f"Generated response: {response}")
            yield response
