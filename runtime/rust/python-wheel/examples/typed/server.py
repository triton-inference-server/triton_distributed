import asyncio

import uvloop
from protocol import Request, Response

from triton_distributed import DistributedRuntime, triton_endpoint, triton_worker

uvloop.install()


class RequestHandler:
    """
    Request handler for the generate endpoint
    """

    @triton_endpoint(Request, Response)
    async def generate(self, request):
        for char in request.data:
            yield char


@triton_worker()
async def worker(runtime: DistributedRuntime):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("triton-init").component("backend")
    await component.create_service()

    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(RequestHandler().generate)


asyncio.run(worker())
