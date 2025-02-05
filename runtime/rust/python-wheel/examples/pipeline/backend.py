import asyncio

import uvloop

from triton_distributed import DistributedRuntime, triton_worker

uvloop.install()


class RequestHandler:
    async def generate(self, request):
        request = f"{request}-back"
        for char in request:
            yield char


@triton_worker()
async def worker(runtime: DistributedRuntime):
    component = runtime.namespace("examples/pipeline").component("backend")
    await component.create_service()

    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(RequestHandler().generate)


asyncio.run(worker())
