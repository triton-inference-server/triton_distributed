import asyncio
import uvloop

from nova_distributed import nova_worker, nova_endpoint, DistributedRuntime
from protocol import Request, Response

uvloop.install()


class RequestHandler:
    """
    Request handler for the generate endpoint
    """
    @nova_endpoint(Request, Response)
    async def generate(self, request):
        for char in request.data:
            yield char


@nova_worker()
async def worker(runtime: DistributedRuntime):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("nova-init").component("backend")
    await component.create_service()
    
    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(RequestHandler().generate)


asyncio.run(worker())
