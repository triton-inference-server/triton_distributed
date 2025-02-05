import asyncio

import uvloop
from triton_distributed_rs import DistributedRuntime, triton_worker

uvloop.install()


class RequestHandler:
    def __init__(self, backend):
        self.backend = backend

    async def generate(self, request):
        request = f"{request}-mid"
        async for output in await self.backend.random(request):
            yield output.get("data")


@triton_worker()
async def worker(runtime: DistributedRuntime):
    # client to backend
    backend = (
        await runtime.namespace("examples/pipeline")
        .component("backend")
        .endpoint("generate")
        .client()
    )

    # create endpoint service for middle component
    component = runtime.namespace("examples/pipeline").component("middle")
    await component.create_service()

    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(RequestHandler(backend).generate)


asyncio.run(worker())
