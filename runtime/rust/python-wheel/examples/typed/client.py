import asyncio

from protocol import Request

from triton_distributed import DistributedRuntime, triton_worker


@triton_worker()
async def worker(runtime: DistributedRuntime):
    """
    Instantiate a `backend` client and call the `generate` endpoint
    """
    # get endpoint
    endpoint = (
        runtime.namespace("triton-init").component("backend").endpoint("generate")
    )

    # create client
    client = await endpoint.client()

    # list the endpoints
    print(client.endpoint_ids())

    # issue request
    stream = await client.generate(Request(data="hello world").model_dump_json())

    # process response
    async for char in stream:
        print(char)


asyncio.run(worker())
