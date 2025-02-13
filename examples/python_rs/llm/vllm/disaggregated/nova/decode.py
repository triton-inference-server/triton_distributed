import uuid

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import logger as vllm_logger

# from nova_init.decorators import nova_endpoint, nova_service, nova_depends
from prefill import Prefill
from compoundai import depends, nova_endpoint, service, api

@service(
    nova={
        "enabled": True,
        "namespace": "triton-init",
    }
)
class Decode:
    prefill = depends(Prefill)

    def __init__(self, engine_args: AsyncEngineArgs):
        engine_args = AsyncEngineArgs(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            max_model_len=100,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            kv_transfer_config={
                "kv_connector": "PyNcclConnector",
                "kv_role": "kv_consumer",
                "kv_rank": 1,
                "kv_parallel_size": 2
            }
        )
        assert (
            engine_args.kv_transfer_config.is_kv_consumer
        ), "Decode worker must be a KV consumer"
        self.engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        print(self.engine.check_health())

    @nova_endpoint # should have req/resp types
    async def generate(self, request):
        vllm_logger.info(f"Received request: {request}")
        print("received request")
        sampling_params = vllm.SamplingParams(**request["sampling_params"])
        print("sampling params")
        request_id = str(uuid.uuid4())
        print("request id")

        prefill_sampling_params = {**request["sampling_params"]}
        prefill_sampling_params["max_tokens"] = 1
        prefill_request = {
            "prompt": request["prompt"],
            "sampling_params": prefill_sampling_params,
            "request_id": request_id,
        }
        print("prefill request abt to be sent")
        prefill_generator = await self.prefill.generate(
            prefill_request
        )
        print("prefill generator abt to be awaited")
        prefill_response = [resp async for resp in prefill_generator]
        assert len(prefill_response) == 1, "Prefill response should be a single boolean"
        prefill_response = prefill_response[0]
        vllm_logger.debug(f"Prefill response: {prefill_response}")

        async for response in self.engine.generate(
            request["prompt"], sampling_params, request_id
        ):
            vllm_logger.debug(f"Generated response: {response}")
            yield response.outputs[0].text


    