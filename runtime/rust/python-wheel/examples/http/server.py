import asyncio
import uvloop

from nova_distributed import nova_worker, DistributedRuntime

uvloop.install()

class LLMBackend:
    async def generate(self, request):
        print(f"Received request: {request}")
        yield request


@nova_worker()
async def worker(runtime: DistributedRuntime):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("my-model").component("llm-backend")
    await component.create_service()

    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(LLMBackend().generate)


asyncio.run(worker())
