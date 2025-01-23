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
import queue
import sys
import time
from functools import partial
from multiprocessing import Process

import cupy
import numpy
import pytest
import tritonclient.grpc as grpcclient
import ucp
from cupy_backends.cuda.api.runtime import CUDARuntimeError
from transformers import XLNetTokenizer
from tritonclient.utils import InferenceServerException
from tritonserver import Tensor

from triton_distributed.icp.nats_request_plane import NatsRequestPlane
from triton_distributed.icp.ucp_data_plane import UcpDataPlane
from triton_distributed.worker.deployment import Deployment
from triton_distributed.worker.logger import get_logger
from triton_distributed.worker.operator import OperatorConfig
from triton_distributed.worker.remote_operator import RemoteOperator
from triton_distributed.worker.triton_core_operator import TritonCoreOperator
from triton_distributed.worker.worker import WorkerConfig

NATS_PORT = 4223
MODEL_REPOSITORY = (
    "/workspace/worker/tests/python/integration/operators/triton_core_models"
)
OPERATORS_REPOSITORY = "/workspace/worker/tests/python/integration/operators"
TRITON_LOG_LEVEL = 6

logger = get_logger(__name__)

# Run cupy's cuda.is_available once to
# avoid the exception hitting runtime code.
try:
    if cupy.cuda.is_available():
        pass
    else:
        print("CUDA not available.")
except CUDARuntimeError:
    print("CUDA not available")

# TODO
# Decide if this should be
# pre merge, nightly, or weekly
pytestmark = pytest.mark.pre_merge


@pytest.fixture
def workers(request, log_dir):
    operator_configs = {}

    # Add configs for triton core operators
    triton_core_operators = ["preprocessing", "context", "generation", "postprocessing"]
    for operator_name in triton_core_operators:
        operator_configs[operator_name] = OperatorConfig(
            name=operator_name,
            implementation=TritonCoreOperator,
            version=1,
            max_inflight_requests=10,
            repository=MODEL_REPOSITORY,
        )

    # Add configs for other custom operators
    operator_name = "mock_disaggregated_serving"
    operator_configs[operator_name] = OperatorConfig(
        name=operator_name,
        implementation="mock_disaggregated_serving:MockDisaggregatedServing",
        version=1,
        max_inflight_requests=10,
        repository=OPERATORS_REPOSITORY,
    )

    worker_configs = []

    test_log_dir = log_dir / request.node.name
    test_log_dir.mkdir(parents=True, exist_ok=True)

    # We will instantiate a worker for each operator
    for name, operator_config in operator_configs.items():
        # Set the logging directory
        worker_log_dir = test_log_dir / name
        worker_configs.append(
            WorkerConfig(
                name=name,
                request_plane=NatsRequestPlane,
                data_plane=UcpDataPlane,
                request_plane_args=(
                    [],
                    {"request_plane_uri": f"nats://localhost:{NATS_PORT}"},
                ),
                log_level=TRITON_LOG_LEVEL,
                log_dir=str(worker_log_dir),
                operators=[operator_config],
            )
        )

    worker_deployment = Deployment(worker_configs)

    worker_deployment.start()
    yield worker_deployment
    worker_deployment.shutdown()


def _create_inputs(number):
    inputs = []
    outputs = []

    for _ in range(number):
        request_output_len = 10
        query_arr = numpy.array(["This is a sample prompt"], dtype=numpy.object_)
        request_output_len_arr = numpy.array([request_output_len], dtype=numpy.int32)
        input_ = {"query": query_arr, "request_output_len": request_output_len_arr}

        expected_output = numpy.repeat(query_arr, request_output_len)

        tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
        tokens = numpy.array(tokenizer.encode(query_arr[0]))
        expected_output = numpy.array(
            tokenizer.convert_ids_to_tokens((tokens.tolist()))
        )

        output_data_ = {"output": Tensor._from_object(expected_output)}

        inputs.append(input_)
        outputs.append(output_data_)
    return inputs, outputs


async def post_requests(num_requests):
    ucp.reset()

    data_plane = UcpDataPlane()
    data_plane.connect()

    request_plane = NatsRequestPlane(f"nats://localhost:{NATS_PORT}")
    await request_plane.connect()

    mock_disaggregated_serving_operator = RemoteOperator(
        "mock_disaggregated_serving", request_plane, data_plane
    )

    expected_results = {}

    inputs, outputs = _create_inputs(num_requests)
    begin = None
    token_latency = []
    timeout = True
    for i, input_dict in enumerate(inputs):
        request_id = str(i)
        request = mock_disaggregated_serving_operator.create_request(
            inputs=input_dict, request_id=request_id
        )

        begin = time.time()
        response_count = 0

        try:
            async for response in await mock_disaggregated_serving_operator.async_infer(
                inference_request=request
            ):
                token_latency.append(time.time() - begin)
                expected_results[request_id] = outputs[i]
                if not response.final:
                    for output_name, expected_value in expected_results[
                        response.request_id
                    ].items():
                        output = response.outputs[output_name]
                        output_value = output.to_bytes_array()
                        print(f"Final Output: {output_value}")
                        numpy.testing.assert_equal(
                            output_value, expected_value.to_bytes_array()
                        )
                    response_count += 1

            # 1 response from context and 10 responses from generation
            assert response_count == 11

        except Exception as e:
            print("Failed collecting responses:" + repr(e))
            del response
            print(f"Token latency: {token_latency}")
            data_plane.close(wait_for_release=timeout)
            await request_plane.close()
            raise e

    print(f"Token latency: {token_latency}")
    data_plane.close(wait_for_release=timeout)
    await request_plane.close()


def run(num_requests):
    sys.exit(asyncio.run(post_requests(num_requests=num_requests)))


@pytest.mark.skipif(
    "(not os.path.exists('/usr/local/bin/nats-server'))",
    reason="NATS.io not present or test is not configured to run with mock disaggregated serving",
)
def test_mock_disaggregated_serving(request, nats_server, workers):
    # Using a separate process to use data plane across multiple tests.
    p = Process(target=run, args=(1,))
    p.start()
    p.join()
    assert p.exitcode == 0


class UserData:
    def __init__(self):
        self._completed_requests: queue.Queue[
            grpcclient.Result | InferenceServerException
        ] = queue.Queue()


# Define the callback function. Note the last two parameters should be
# result and error. InferenceServerClient would povide the results of an
# inference as grpcclient.InferResult in result. For successful
# inference, error will be None, otherwise it will be an object of
# tritonclientutils.InferenceServerException holding the error details
def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


async def send_kserve_requests(num_requests):
    inputs_dict, outputs_dicts = _create_inputs(num_requests)
    inputs = []
    inputs.append(grpcclient.InferInput("query", [1], "BYTES"))
    inputs.append(grpcclient.InferInput("request_output_len", [1], "INT32"))

    user_data = UserData()

    with grpcclient.InferenceServerClient("localhost:8001") as client:
        client.start_stream(
            callback=partial(callback, user_data),
        )
        for i, input_dict in enumerate(inputs_dict):
            inputs[0].set_data_from_numpy(input_dict["query"])
            inputs[1].set_data_from_numpy(input_dict["request_output_len"])

            client.async_stream_infer(
                model_name="mock_disaggregated_serving", inputs=inputs
            )

        recv_count = 0
        while recv_count < 10:
            data_item = user_data._completed_requests.get()
            recv_count += 1
            if isinstance(data_item, InferenceServerException):
                raise data_item
            else:
                result = data_item.as_numpy("output")
                print("test \n")
                print(result)

    # Wait for the tensor clean-up
    time.sleep(5)


def run_kserve(num_requests):
    sys.exit(asyncio.run(send_kserve_requests(num_requests=num_requests)))


@pytest.mark.skipif(
    "(not os.path.exists('/usr/local/bin/nats-server'))",
    reason="NATS.io not present",
)
def test_mock_disaggregated_serving_kserve(request, nats_server, workers, api_server):
    # Using a separate process to use data plane across multiple tests.
    p = Process(target=run_kserve, args=(1,))
    p.start()
    p.join()
    assert p.exitcode == 0
