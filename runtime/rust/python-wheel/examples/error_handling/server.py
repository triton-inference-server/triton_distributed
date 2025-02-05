import asyncio

import uvloop
from triton_distributed_rs import DistributedRuntime, triton_worker


class RequestHandler:
    """
    Request handler for the generate endpoint
    """

    async def generate(self, request):
        print(f"Received request: {request}")
        for char in request:
            if char == "w":
                raise ValueError("w is not allowed")
            yield char


@triton_worker()
async def worker(runtime: DistributedRuntime):
    await init(runtime, "triton-init")


async def init(runtime: DistributedRuntime, ns: str):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace(ns).component("backend")
    await component.create_service()

    endpoint = component.endpoint("generate")
    print("Started server instance")
    await endpoint.serve_endpoint(RequestHandler().generate)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
