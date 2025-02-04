import asyncio

import uvloop
from nova_distributed import DistributedRuntime, nova_worker


@nova_worker()
async def worker(runtime: DistributedRuntime):
    await init(runtime, "nova-init")


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

    error_count = 0
    try:
        # process response
        async for char in stream:
            print(char)
    except ValueError:
        error_count += 1
        pass
    finally:
        assert error_count == 1

    stream = await client.generate("hello earth", annotated=False)

    async for char in stream:
        print(char)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
