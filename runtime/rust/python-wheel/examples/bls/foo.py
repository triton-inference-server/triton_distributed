import asyncio
import uvloop

from nova_distributed import nova_worker, DistributedRuntime

uvloop.install()


class RequestHandler:
    async def generate(self, request):
        for char in request:
            yield char


@nova_worker()
async def worker(runtime: DistributedRuntime):
    component = runtime.namespace("examples/bls").component("foo")
    await component.create_service()

    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(RequestHandler().generate)


asyncio.run(worker())
