import asyncio
import uuid

import uvloop
import vllm
from protocol import Request, Response
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import FlexibleArgumentParser
from vllm.logger import logger as vllm_logger

from triton_distributed_rs import DistributedRuntime, triton_endpoint, triton_worker


class VllmEngine:
    """
    Request handler for the generate endpoint
    """
    def __init__(self, engine_args: AsyncEngineArgs):
        self.engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        
    @triton_endpoint(Request, Response)
    async def generate(self, request):
        vllm_logger.info(f"Received request: {request}")
        sampling_params = vllm.SamplingParams(**request.sampling_params)
        request_id = str(uuid.uuid4())
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

    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(VllmEngine(engine_args).generate)


if __name__ == "__main__":
    uvloop.install()
    parser = FlexibleArgumentParser()
    parser = AsyncEngineArgs.add_cli_args(parser)
    engine_args = AsyncEngineArgs.from_cli_args(parser.parse_args())
    asyncio.run(worker(engine_args))
