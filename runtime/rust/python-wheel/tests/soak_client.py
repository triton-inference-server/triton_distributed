import argparse
import asyncio

# from triton_distributed.runtime import RemoteOperator as RemoteFunction
import time

import uvloop
from triton_distributed_rs import DistributedRuntime, triton_worker

# from triton_distributed.icp import NatsRequestPlane, UcpDataPlane


async def do_one(client):
    stream = await client.generate("hello world")
    async for char in stream:
        pass


@triton_worker()
async def main(runtime: DistributedRuntime, ns: str = "soak", request_count=5000):
    """
    Instantiate a `backend` client and call the `generate` endpoint
    """
    """
    Instantiate a `backend` client and call the `generate` endpoint
    """
    start_time = time.time()
    # get endpoint
    endpoint = runtime.namespace(ns).component("backend").endpoint("generate")

    # create client
    client = await endpoint.client()

    # wait for an endpoint to be ready
    await client.wait_for_endpoints()

    # issue 1000 concurrent requests
    # the task should issue the request and process the response
    tasks = []
    for i in range(request_count):
        tasks.append(asyncio.create_task(do_one(client)))

    await asyncio.gather(*tasks)

    # ensure all tasks are done and without errors
    error_count = 0
    for task in tasks:
        if task.exception():
            error_count += 1

    assert error_count == 0, f"expected 0 errors, got {error_count}"

    print(f"time: {time.time()-start_time}")


if __name__ == "__main__":
    uvloop.install()
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-count", type=int, default=5000)
    args = parser.parse_args()
    asyncio.run(main(request_count=args.request_count))
