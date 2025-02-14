# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio

# import gc
import random
import string
import time
from typing import AsyncIterator

import uvloop

from triton_distributed.icp import NatsServer
from triton_distributed.runtime import CallableOperator
from triton_distributed.runtime import OperatorConfig as FunctionConfig
from triton_distributed.runtime.mp_worker import MPWorker

# import uvloop
# from triton_distributed_rs import DistributedRuntime, triton_worker

# Soak Test
#
# This was a failure case for the distributed runtime. If the Rust Tokio
# runtime is started with a small number of threads, it will starve the
# the GIL + asyncio event loop can starve timeout the ingress handler.
#
# There may still be some blocking operations in the ingress handler that
# could still eventually be a problem.


def worker():
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    #    component = runtime.namespace(ns).component("backend")
    #    await component.create_service()

    #    endpoint = component.endpoint("generate")

    generate_engine = FunctionConfig(
        name="generate",
        implementation=CallableOperator,
        parameters={"callable_object": RequestHandler().generate},
        max_inflight_requests=10000,
    )

    print("Started server instance")

    # Worker(
    #     operators=[generate_engine],
    #     log_level=1,
    # ).start()

    MPWorker(
        operators=[generate_engine],
        log_level=1,
    ).start()


#    await endpoint.serve_endpoint(RequestHandler().generate)


class RequestHandler:
    """
    Request handler for the generate endpoint
    """

    async def generate(self, request: str) -> AsyncIterator[str]:
        # await asyncio.sleep(2)
        # first_sent = False
        for char in request:
            await asyncio.sleep(0.1)
            yield char
            # if not first_sent:
            # print(f"\t\t\t\t{time.time_ns()}")
            # first_sent = True
        # print(f"\t\t\t\t\t\t\t\t{time.time_ns()}")


def random_string(length=10):
    chars = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
    return "".join(random.choices(chars, k=length))


if __name__ == "__main__":
    # gc.disable()
    uvloop.install()
    request_plane_server = NatsServer(log_dir=None)
    time.sleep(2)
    worker()
