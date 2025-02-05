import asyncio

import uvloop

from triton_distributed import DistributedRuntime, triton_worker

uvloop.install()


@triton_worker()
async def worker(runtime: DistributedRuntime):
    foo = (
        await runtime.namespace("examples/bls")
        .component("foo")
        .endpoint("generate")
        .client()
    )
    bar = (
        await runtime.namespace("examples/bls")
        .component("bar")
        .endpoint("generate")
        .client()
    )

    # hello world showed us the client has a .generate, which uses the default load balancer
    # however, you can explicitly opt-in to client side load balancing by using the `round_robin`
    # or `random` methods on client. note - there is a direct method as well, but that is for a
    # router example
    async for char in await foo.round_robin("hello world"):
        # the responses are sse-style responses, so we extract the data key
        async for x in await bar.random(char.get("data")):
            print(x)


asyncio.run(worker())
