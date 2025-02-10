import argparse
import asyncio
import time

from triton_distributed.icp import NatsRequestPlane, UcpDataPlane

# from triton_distributed import DistributedRuntime, triton_worker
from triton_distributed.runtime import RemoteOperator as RemoteFunction


async def do_one(client):
    stream = client.call("hello world", return_type=str)
    async for char in stream:
        pass


async def main(request_count):
    """
    Instantiate a `backend` client and call the `generate` endpoint
    """
    start_time = time.time()
    # get endpoint
    #    endpoint = runtime.namespace(ns).component("backend").endpoint("generate")

    # create client
    #   client = await endpoint.client()

    # wait for an endpoint to be ready
    #  await client.wait_for_endpoints()

    # issue 1000 concurrent requests
    # the task should issue the request and process the response

    request_plane = NatsRequestPlane()
    await request_plane.connect()

    data_plane = UcpDataPlane()
    data_plane.connect()

    client = RemoteFunction("generate", request_plane, data_plane)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-count", type=int, default=5000)

    args = parser.parse_args()

    asyncio.run(main(args.request_count))
