import asyncio
import uvloop

from nova_distributed import nova_worker, DistributedRuntime

uvloop.install()


@nova_worker()
async def worker(runtime: DistributedRuntime):
    """
    # Pipeline Example

    This example demonstrates how to create a pipeline of components:
    - `frontend` call `middle` which calls `backend`
    - each component transforms the request before passing it to the backend
    """
    pipeline = (
        await runtime.namespace("examples/pipeline")
        .component("frontend")
        .endpoint("generate")
        .client()
    )

    async for char in await pipeline.round_robin("hello from"):
        print(char)


asyncio.run(worker())
