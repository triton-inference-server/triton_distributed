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

import vllm
from common.parser import parse_vllm_args
from common.protocol import Request, Response

# from triton_distributed_rs import DistributedRuntime, triton_endpoint, triton_worker
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import logger as vllm_logger

from triton_distributed.icp import NatsServer
from triton_distributed.runtime import CallableOperator
from triton_distributed.runtime import OperatorConfig as FunctionConfig
from triton_distributed.runtime import Worker


class VllmEngine:
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine_args: AsyncEngineArgs):
        self.engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)

    async def generate(self, request: Request) -> AsyncIterator[Response]:
        vllm_logger.debug(f"Received request: {request}")
        sampling_params = vllm.SamplingParams(**request.sampling_params)
        request_id = str(uuid.uuid4())
        async for response in self.engine.generate(
            request.prompt, sampling_params, request_id
        ):
            vllm_logger.debug(f"Generated response: {response}")
            yield Response(response.outputs[0].text)


def worker(engine_args: AsyncEngineArgs):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    # component = runtime.namespace("triton-init").component("vllm")
    # await component.create_service()

    # endpoint = component.endpoint("generate")
    # await endpoint.serve_endpoint(VllmEngine(engine_args).generate)
    vllm_engine = FunctionConfig(
        name="vllm_generate",
        implementation=CallableOperator,
        parameters={"callable_object": VllmEngine(engine_args).generate},
        max_inflight_requests=10000,
    )

    Worker(operators=[vllm_engine], log_level=1).start()


if __name__ == "__main__":
    #    uvloop.install()
    request_plane_server = NatsServer(log_dir=None)
    time.sleep(2)
    engine_args = parse_vllm_args()
    worker(engine_args)
