import asyncio

import uvloop
from triton_distributed_rs import DistributedRuntime, triton_worker


@triton_worker()
async def worker(runtime: DistributedRuntime):
    await init(runtime, "triton-init")


async def init(runtime: DistributedRuntime, ns: str):
    """
    Instantiate a `backend` client and call the `generate` endpoint
    """
    # get endpoint
    endpoint = runtime.namespace(ns).component("backend").endpoint("generate")

    # create client
    client = await endpoint.client()

    # wait for an endpoint to be ready
    await client.wait_for_endpoints()

    # issue request
    stream = await client.generate("hello world")

    # process the stream
    async for char in stream:
        print(char)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
