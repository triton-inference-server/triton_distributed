import asyncio

import uvloop
from nova_distributed import DistributedRuntime, nova_worker

uvloop.install()


class RequestHandler:
    async def generate(self, request):
        request = f"{request}-back"
        for char in request:
            yield char


@nova_worker()
async def worker(runtime: DistributedRuntime):
    component = runtime.namespace("examples/pipeline").component("backend")
    await component.create_service()

    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(RequestHandler().generate)


asyncio.run(worker())
