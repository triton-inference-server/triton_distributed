import asyncio

import uvloop
from triton_distributed_rs import DistributedRuntime, triton_worker

uvloop.install()


@triton_worker()
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
