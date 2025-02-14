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

import time
import uuid
from typing import AsyncIterator

import uvloop
import vllm
from common.metrics import RequestMetric
from common.parser import parse_vllm_args
from common.protocol import Request, Response

# from triton_distributed_rs import DistributedRuntime, triton_endpoint, triton_worker
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import logger as vllm_logger

from triton_distributed.icp import NatsEventPlane, NatsServer
from triton_distributed.runtime import CallableOperator
from triton_distributed.runtime import OperatorConfig as FunctionConfig
from triton_distributed.runtime import Worker


class VllmEngine:
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine_args: AsyncEngineArgs, component_id):
        self.engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        self._component_id = component_id
        self._event_plane = NatsEventPlane(
            component_id=self._component_id, server_uri="nats://localhost:4222"
        )

    async def generate(self, request: Request) -> AsyncIterator[Response]:
        vllm_logger.debug(f"Received request: {request}")
        sampling_params = vllm.SamplingParams(**request.sampling_params)
        await self._event_plane.connect()
        request_id = str(uuid.uuid4())
        response_token_count = 0
        async for response in self.engine.generate(
            request.prompt, sampling_params, request_id
        ):
            vllm_logger.debug(f"Generated response: {response}")
            yield Response(response.outputs[0].text)
            response_token_count += 1

        await self._event_plane.publish(
            payload=RequestMetric(len(request.prompt), response_token_count),
            event_topic=["request_count"],
        )


def worker(engine_args: AsyncEngineArgs):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    # component = runtime.namespace("triton-init").component("vllm")
    # await component.create_service()

    # endpoint = component.endpoint("generate")
    # await endpoint.serve_endpoint(VllmEngine(engine_args).generate)

    component_id = uuid.uuid4()

    vllm_engine = FunctionConfig(
        name="vllm_generate",
        implementation=CallableOperator,
        parameters={"callable_object": VllmEngine(engine_args, component_id).generate},
        max_inflight_requests=10000,
    )

    Worker(
        operators=[vllm_engine],
        log_level=1,
        request_plane_args=([], {"component_id": component_id}),
    ).start()


if __name__ == "__main__":
    uvloop.install()
    request_plane_server = NatsServer(log_dir=None)
    time.sleep(2)
    engine_args = parse_vllm_args()
    worker(engine_args)
