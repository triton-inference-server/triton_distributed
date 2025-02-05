import asyncio

import uvloop
from triton_distributed_rs import DistributedRuntime, triton_worker

uvloop.install()


class RequestHandler:
    async def generate(self, request):
        for char in request:
            yield char
            yield char


@triton_worker()
async def worker(runtime: DistributedRuntime):
    component = runtime.namespace("examples/bls").component("bar")
    await component.create_service()

    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(RequestHandler().generate)


asyncio.run(worker())
