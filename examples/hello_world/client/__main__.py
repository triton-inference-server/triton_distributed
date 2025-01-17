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

import multiprocessing
import time

import cupy
import numpy
import tqdm
from triton_distributed.icp import NatsRequestPlane, UcpDataPlane
from triton_distributed.worker import RemoteOperator
from tritonserver import MemoryType

from .parser import parse_args


def _get_input_sizes(args):
    return numpy.maximum(
        0,
        numpy.round(
            numpy.random.normal(
                loc=args.input_size_mean,
                scale=args.input_size_stdev,
                size=args.requests_per_client,
            )
        ),
    ).astype(int)


async def client(client_index, args):
    request_count = args.requests_per_client
    request_plane = NatsRequestPlane(args.request_plane_uri)
    data_plane = UcpDataPlane()
    await request_plane.connect()
    data_plane.connect()

    remote_operator: RemoteOperator = RemoteOperator(
        args.operator, request_plane, data_plane
    )
    input_sizes = _get_input_sizes(args)

    inputs = [
        numpy.array(numpy.random.randint(0, 100, input_sizes[index]))
        for index in range(request_count)
    ]

    with tqdm(
        total=args.requests_per_client,
        desc=f"Client: {client_index}",
        unit="request",
        position=client_index,
    ) as pbar:
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
                    f"Client Received Response: {response.request_id} From: {response.component_id} Error: {response.error}"
                )
                pbar.update(1)
                del response

    await request_plane.close()
    data_plane.close()


def main(args):
    process_context = multiprocessing.get_context("spawn")
    processes = []
    start_time = time.time()
    for index in range(args.clients):
        processes.append(process_context.Process(target=client, args=(index, args)))
        processes[-1].start()

    for process in processes:
        process.join()
    end_time = time.time()
    print(
        f"Throughput: {(args.requests_per_client*args.clients)/(end_time-start_time)} Total Time: {end_time-start_time}"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
