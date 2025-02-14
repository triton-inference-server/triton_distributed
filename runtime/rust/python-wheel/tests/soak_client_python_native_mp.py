import argparse
import asyncio
import multiprocessing
import time

import uvloop

from triton_distributed.icp import NatsRequestPlane, UcpDataPlane

# from triton_distributed import DistributedRuntime, triton_worker
from triton_distributed.runtime import RemoteOperator as RemoteFunction


async def do_one(client):
    stream = client.call("hello world", return_type=str)
    async for char in stream:
        pass
    # print(time.time_ns())


async def run_client(request_count, use_zmq_response_path):
    """
    Instantiate a `backend` client and call the `generate` endpoint
    """
    # get endpoint
    #    endpoint = runtime.namespace(ns).component("backend").endpoint("generate")

    # create client
    #   client = await endpoint.client()

    # wait for an endpoint to be ready
    #  await client.wait_for_endpoints()

    # issue 1000 concurrent requests
    # the task should issue the request and process the response

    request_plane = NatsRequestPlane(use_zmq_response_path=use_zmq_response_path)
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


def main(request_count, use_zmq_response_path):
    asyncio.run(run_client(request_count, use_zmq_response_path))


if __name__ == "__main__":
    uvloop.install()

    parser = argparse.ArgumentParser()
    parser.add_argument("--request-count", type=int, default=5000)
    parser.add_argument("--process-count", type=int, default=8)
    parser.add_argument("--use-zmq-response-path", action="store_true", default=False)
    args = parser.parse_args()

    request_count = args.request_count
    process_count = args.process_count
    use_zmq_response_path = args.use_zmq_response_path
    assert request_count % process_count == 0

    start_time = time.time()

    processes = []
    for i in range(process_count):
        processes.append(
            multiprocessing.Process(
                target=main,
                args=(request_count // process_count, use_zmq_response_path),
            )
        )
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    print(f"time: {time.time()-start_time}")
