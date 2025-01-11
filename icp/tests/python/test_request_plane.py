# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import asyncio
import shutil
import subprocess
import time
import uuid
from multiprocessing import Process, Queue

import pytest
import tdist
from tdist.icp.nats_request_plane import NatsRequestPlane
from tdist.icp.protos.icp_pb2 import ModelInferRequest, ModelInferResponse
from tdist.icp.request_plane import get_icp_component_id

NATS_PORT = 4222


def is_port_in_use(port: int) -> None:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


@pytest.fixture
def nats_server(request):
    command = [
        "/usr/local/bin/nats-server",
        "--jetstream",
        "--debug",
        "--trace",
        "--port",
        str(NATS_PORT),
    ]
    print(f"Running: [{' '.join(command)}]")

    # Raise more intuitive error to developer if port is already in-use.
    if is_port_in_use(NATS_PORT):
        raise RuntimeError(
            f"ERROR: NATS Port {NATS_PORT} already in use. Is a nats-server already running?"
        )

    shutil.rmtree("/tmp/nats", ignore_errors=True)

    with open("nats_server.stdout.log", "wt") as output_:
        with open("nats_server.stderr.log", "wt") as output_err:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=output_, stderr=output_err
            )
            time.sleep(1)
            yield process

            process.terminate()
            process.wait()

            shutil.rmtree("/tmp/nats", ignore_errors=True)


class ResponseHandler:
    def __init__(self, request_plane):
        self._request_plane = request_plane

    async def response_handler(self, response):
        print(response)
        request = ModelInferRequest()
        request.model_name = response.model_name
        request.model_version = response.model_version
        print("publishing request")
        acks = []
        for i in range(5):
            acks.append(
                self._request_plane.post_request(
                    request, response_handler=self.response_handler
                )
            )
        asyncio.gather(*acks)


@pytest.mark.timeout(30)
async def test_handler(nats_server):
    model_name = str(uuid.uuid1())
    model_version = "1"

    client_request_plane = NatsRequestPlane()
    await client_request_plane.connect()
    request = ModelInferRequest()
    request.model_name = model_name
    request.model_version = model_version
    await client_request_plane.post_request(
        request, response_handler=ResponseHandler(client_request_plane).response_handler
    )

    worker_request_plane = NatsRequestPlane()

    await worker_request_plane.connect()
    request_count = 10
    while request_count > 0:
        requests = await worker_request_plane.pull_requests(
            model_name, model_version, 100, 0.1
        )
        acks = []
        async for request in requests:
            request_count -= 1
            response = ModelInferResponse()
            tdist.icp.request_plane.set_icp_final_response(response, True)
            acks.append(worker_request_plane.post_response(request, response))
        print(request_count)
        await asyncio.gather(*acks)
        await asyncio.sleep(0.1)
    requests = await worker_request_plane.pull_requests(
        model_name, model_version, 100, 0.1
    )
    await worker_request_plane.close()
    await client_request_plane.close()


def run_request_generator(request_queue, response_queue, direct_requests=False):
    asyncio.run(
        request_generator(
            request_queue, response_queue, direct_requests=direct_requests
        )
    )


async def request_generator(request_queue, response_queue, direct_requests=False):
    # Generate requests and wait for responses
    # if direct_requests == True, then send all requests to the
    # worker that responds to the first request

    request_plane = NatsRequestPlane()
    await request_plane.connect()
    target_component_id = None
    while True:
        request = request_queue.get()
        if request is None:
            response_queue.put(None)
            break
        async for response in await request_plane.post_request(
            request, response_iterator=True, component_id=target_component_id
        ):
            if direct_requests:
                target_component_id = get_icp_component_id(response)
            print(response)
            response_queue.put(response)


def run_worker(model_name, model_version, batch_size, request_count, pull_timeout=0.1):
    asyncio.run(
        worker(model_name, model_version, batch_size, request_count, pull_timeout)
    )


async def worker(
    model_name, model_version, batch_size, request_count, pull_timeout=0.1
):
    request_plane = NatsRequestPlane()
    await request_plane.connect()
    while request_count:
        requests = await request_plane.pull_requests(
            model_name, model_version, batch_size, pull_timeout
        )
        acks = []

        async for request in requests:
            print(request)
            request_count -= 1
            response = ModelInferResponse()
            tdist.icp.request_plane.set_icp_final_response(response, True)
            acks.append(request_plane.post_response(request, responses=response))
        await asyncio.gather(*acks)


@pytest.mark.timeout(30)
async def test_iterator(nats_server):
    batch_size = 10
    request_count = 100
    model_name = str(uuid.uuid1())
    model_version = "1"
    request_queue = Queue()
    response_queue = Queue()
    generator_process = Process(
        target=run_request_generator, args=(request_queue, response_queue)
    )

    worker_process = Process(
        target=run_worker, args=(model_name, model_version, batch_size, request_count)
    )

    generator_process.start()
    worker_process.start()

    for index in range(request_count):
        request_queue.put(
            ModelInferRequest(
                model_name=model_name, model_version=model_version, id=str(index)
            )
        )
    request_queue.put(None)

    generator_process.join()
    worker_process.join()

    response_count = 0

    while True:
        response = response_queue.get()
        if response is None:
            break
        response_count += 1

    assert request_count == response_count


@pytest.mark.parametrize("pull_timeout,batch_size", [(0.1, 10), (None, 1)])
@pytest.mark.timeout(30)
async def test_direct_requests(nats_server, pull_timeout, batch_size):
    request_count = 100
    model_name = str(uuid.uuid1())
    model_version = "1"
    request_queue = Queue()
    response_queue = Queue()

    # Note with direct_requests == True
    # all requests should target a single worker
    # and all responses should be from a single worker

    generator_process = Process(
        target=run_request_generator,
        args=(request_queue, response_queue),
        kwargs={"direct_requests": True},
    )

    worker_process_1 = Process(
        target=run_worker,
        args=(model_name, model_version, batch_size, request_count, pull_timeout),
    )

    worker_process_2 = Process(
        target=run_worker,
        args=(model_name, model_version, batch_size, request_count, pull_timeout),
    )

    worker_process_1.start()
    worker_process_2.start()
    time.sleep(1)
    generator_process.start()

    for index in range(request_count):
        request_queue.put(
            ModelInferRequest(
                model_name=model_name, model_version=model_version, id=str(index)
            )
        )
    request_queue.put(None)

    generator_process.join()
    worker_process_1.terminate()
    worker_process_1.join()
    worker_process_2.terminate()
    worker_process_2.join()

    response_count = 0

    responders = set()

    while True:
        response = response_queue.get()
        if response is None:
            break
        response_count += 1
        responders.add(get_icp_component_id(response))

    assert len(responders) == 1
    assert request_count == response_count
