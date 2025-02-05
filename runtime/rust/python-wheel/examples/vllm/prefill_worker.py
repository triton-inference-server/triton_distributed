import asyncio
import uuid

import uvloop
import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser
from vllm.logger import logger as vllm_logger

from triton_distributed_rs import DistributedRuntime, triton_endpoint, triton_worker
from protocol import PrefillRequest, PrefillResponse

class VllmPrefillEngine:
    """
    Request handler for the generate endpoint
    """
    def __init__(self, engine_args: AsyncEngineArgs):
        assert engine_args.kv_transfer_config.is_kv_producer, "Prefill worker must be a KV producer"
        self.engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        
    @triton_endpoint(PrefillRequest, PrefillResponse)
    async def generate(self, request):
        vllm_logger.info(f"Received prefill request: {request}")
        sampling_params = vllm.SamplingParams(**request.sampling_params)
        async for response in self.engine.generate(request.prompt, sampling_params, request.request_id):
            vllm_logger.debug(f"Generated response: {response}")
            yield True


@triton_worker()
async def worker(runtime: DistributedRuntime, engine_args: AsyncEngineArgs):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("triton-init").component("prefill")
    await component.create_service()

    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(VllmPrefillEngine(engine_args).generate)


if __name__ == "__main__":
    uvloop.install()
    parser = FlexibleArgumentParser()
    parser = AsyncEngineArgs.add_cli_args(parser)
    engine_args = AsyncEngineArgs.from_cli_args(parser.parse_args())
    asyncio.run(worker(engine_args))
