import asyncio
import uuid

import uvloop
import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser
from vllm.logger import logger as vllm_logger

from triton_distributed_rs import DistributedRuntime, triton_endpoint, triton_worker, Client
from protocol import Request, Response

class VllmDecodeEngine:
    """
    Request handler for the generate endpoint
    """
    def __init__(self, engine_args: AsyncEngineArgs, prefill: Client):
        assert engine_args.kv_transfer_config.is_kv_consumer, "Decode worker must be a KV consumer"
        self.engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        self.prefill = prefill
        
    @triton_endpoint(Request, Response)
    async def generate(self, request):
        vllm_logger.info(f"Received request: {request}")
        sampling_params = vllm.SamplingParams(**request.sampling_params)
        request_id = str(uuid.uuid4())

        prefill_generator = await self.prefill.generate(request.prompt, sampling_params, request_id)
        prefill_response = await prefill_generator.next()
        vllm_logger.info(f"Prefill response: {prefill_response}")

        async for response in self.engine.generate(request.prompt, sampling_params, request_id):
            vllm_logger.info(f"Generated response: {response}")
            yield response.outputs[0].text


@triton_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("triton-init").component("vllm")
    await component.create_service()

    prefill = await runtime.namespace("triton-init").component("prefill").endpoint("generate").client()

    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(VllmDecodeEngine(engine_args, prefill).generate)


if __name__ == "__main__":
    uvloop.install()
    parser = FlexibleArgumentParser()
    parser = AsyncEngineArgs.add_cli_args(parser)
    engine_args = AsyncEngineArgs.from_cli_args(parser.parse_args())
    asyncio.run(worker(engine_args))
