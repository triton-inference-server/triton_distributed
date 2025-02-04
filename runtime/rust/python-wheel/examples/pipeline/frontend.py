import asyncio
import uvloop

from nova_distributed import nova_worker, DistributedRuntime

uvloop.install()

class RequestHandler:
    def __init__(self, next):
        self.next = next

    async def generate(self, request):
        request = f"{request} front"
        async for output in await self.next.round_robin(request):
            yield output.get("data")


@nova_worker()
async def worker(runtime: DistributedRuntime):
    # client to the next component - in this case the middle component
    next = await runtime.namespace("examples/pipeline").component("middle").endpoint("generate").client()

    # create endpoint service for frontend component
    component = runtime.namespace("examples/pipeline").component("frontend")
    await component.create_service()
    
    endpoint = component.endpoint("generate")

    handler = RequestHandler(next)
    await endpoint.serve_endpoint(handler.generate)


asyncio.run(worker())
