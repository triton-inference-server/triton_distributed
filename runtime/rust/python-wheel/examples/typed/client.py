import asyncio

from nova_distributed import DistributedRuntime, nova_worker
from protocol import Request


@nova_worker()
async def worker(runtime: DistributedRuntime):
    """
    Instantiate a `backend` client and call the `generate` endpoint
    """
    # get endpoint
    endpoint = runtime.namespace("nova-init").component("backend").endpoint("generate")

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
