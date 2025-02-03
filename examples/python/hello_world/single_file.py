# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import shutil
import sys
from pathlib import Path

import cupy
import numpy
from tqdm import tqdm
from tritonserver import MemoryType

from triton_distributed.icp.nats_request_plane import NatsRequestPlane
from triton_distributed.icp.ucp_data_plane import UcpDataPlane
from triton_distributed.runtime import (
    Deployment,
    Operator,
    OperatorConfig,
    RemoteInferenceRequest,
    RemoteOperator,
    TritonCoreOperator,
    WorkerConfig,
)


class EncodeDecodeOperator(Operator):
    def __init__(
        self,
        name,
        version,
        triton_core,
        request_plane,
        data_plane,
        parameters,
        repository,
        logger,
    ):
        self._encoder = RemoteOperator("encoder", request_plane, data_plane)
        self._decoder = RemoteOperator("decoder", request_plane, data_plane)
        self._logger = logger

    async def execute(self, requests: list[RemoteInferenceRequest]):
        for request in requests:
            self._logger.info("got request!")
            encoded_responses = await self._encoder.async_infer(
                inputs={"input": request.inputs["input"]}
            )

            async for encoded_response in encoded_responses:
                input_copies = int(
                    numpy.from_dlpack(encoded_response.outputs["input_copies"])
                )
                decoded_responses = await self._decoder.async_infer(
                    inputs={"input": encoded_response.outputs["output"]},
                    parameters={"input_copies": input_copies},
                )

                async for decoded_response in decoded_responses:
                    await request.response_sender().send(
                        final=True,
                        outputs={"output": decoded_response.outputs["output"]},
                    )
                    del decoded_response


async def send_requests(nats_server_url, request_count=10):
    request_plane = NatsRequestPlane(nats_server_url)
    data_plane = UcpDataPlane()
    await request_plane.connect()
    data_plane.connect()

    remote_operator: RemoteOperator = RemoteOperator(
        "encoder_decoder", request_plane, data_plane
    )

    inputs = [
        numpy.array(numpy.random.randint(0, 100, 10000)).astype("int64")
        for _ in range(request_count)
    ]

    with tqdm(total=request_count, desc="Sending Requests", unit="request") as pbar:
        requests = [
            await remote_operator.async_infer(
                inputs={"input": inputs[index]}, request_id=str(index)
            )
            for index in range(request_count)
        ]

        for request in requests:
            async for response in request:
                for output_name, output_value in response.outputs.items():
                    if output_value.memory_type == MemoryType.CPU:
                        output = numpy.from_dlpack(output_value)
                        numpy.testing.assert_array_equal(
                            output, inputs[int(response.request_id)]
                        )
                    else:
                        output = cupy.from_dlpack(output_value)
                        cupy.testing.assert_array_equal(
                            output, inputs[int(response.request_id)]
                        )
                    del output_value

                pbar.set_description(
                    f"Finished Request: {response.request_id} Response From: {response.component_id} Error: {response.error}"
                )
                pbar.update(1)
                del response

    await request_plane.close()
    data_plane.close()


async def main():
    module_dir = Path(__file__).parent.absolute()

    log_dir = module_dir.joinpath("logs")

    if log_dir.is_dir():
        shutil.rmtree(log_dir)

    log_dir.mkdir(exist_ok=True)

    triton_core_models_dir = module_dir.joinpath("operators", "triton_core_models")

    encoder_op = OperatorConfig(
        name="encoder",
        repository=str(triton_core_models_dir),
        implementation=TritonCoreOperator,
        max_inflight_requests=1,
        parameters={
            "config": {
                "instance_group": [{"count": 1, "kind": "KIND_CPU"}],
                "parameters": {"delay": {"string_value": "0"}},
            }
        },
    )

    decoder_op = OperatorConfig(
        name="decoder",
        repository=str(triton_core_models_dir),
        implementation=TritonCoreOperator,
        max_inflight_requests=1,
        parameters={
            "config": {
                "instance_group": [{"count": 1, "kind": "KIND_GPU"}],
                "parameters": {"delay": {"string_value": "0"}},
            }
        },
    )

    encoder_decoder_op = OperatorConfig(
        name="encoder_decoder",
        implementation=EncodeDecodeOperator,
        max_inflight_requests=100,
    )

    encoder = WorkerConfig(
        operators=[encoder_op],
        name="encoder",
    )

    decoder = WorkerConfig(
        operators=[decoder_op],
        name="decoder",
    )

    encoder_decoder = WorkerConfig(
        operators=[encoder_decoder_op],
        name="encoder_decoder",
    )

    print("Starting Workers")

    # You can configure the number of instances of each
    # type of worker in a deployment

    num_instances = 1

    deployment = Deployment(
        [
            (encoder, num_instances),
            (decoder, num_instances),
            (encoder_decoder, num_instances),
        ],
        initialize_request_plane=True,
        log_dir=str(log_dir),
        log_level=1,
    )

    deployment.start()

    print("Sending Requests")

    await send_requests(deployment.request_plane_server.url)

    print("Stopping Workers")

    sys.exit(deployment.stop())


if __name__ == "__main__":
    asyncio.run(main())
