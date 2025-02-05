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

import numpy
import pytest
import ucp

from triton_distributed.icp.nats_request_plane import NatsRequestPlane
from triton_distributed.icp.ucp_data_plane import UcpDataPlane
from triton_distributed.runtime.deployment import Deployment
from triton_distributed.runtime.logger import get_logger
from triton_distributed.runtime.operator import OperatorConfig
from triton_distributed.runtime.remote_operator import RemoteOperator
from triton_distributed.runtime.worker import WorkerConfig

NATS_PORT = 4223
MODEL_REPOSITORY = (
    "/workspace/runtime/tests/python/integration/operators/triton_core_models"
)
OPERATORS_REPOSITORY = "/workspace/runtime/tests/python/integration/operators"
TRITON_LOG_LEVEL = 0

logger = get_logger(__name__)

# TODO
# Decide if this should be
# pre merge, nightly, or weekly
pytestmark = pytest.mark.pre_merge


@pytest.fixture
def workers(log_dir, request, number_workers=1):
    store_outputs_in_response = request.getfixturevalue("store_outputs_in_response")

    # Add configs for identity operator
    operator_name = "identity"
    operator_config = OperatorConfig(
        name=operator_name,
        implementation="identity:Identity",
        version=1,
        max_inflight_requests=10,
        parameters={"store_outputs_in_response": store_outputs_in_response},
        repository=OPERATORS_REPOSITORY,
    )

    worker_configs = []

    test_log_dir = log_dir / request.node.name
    test_log_dir.mkdir(parents=True, exist_ok=True)

    for i in range(number_workers):
        # Set the logging directory
        worker_log_dir = test_log_dir / (operator_name + "_" + str(i))
        worker_configs.append(
            WorkerConfig(
                name=operator_name,
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


def _create_inputs(number, tensor_size_in_kb):
    inputs = []
    outputs = []

    elem_cnt = int(tensor_size_in_kb * 1024 / 4)
    for _ in range(number):
        input_ = numpy.random.randint(low=1, high=100, size=[elem_cnt])

        expected_ = {}

        expected_["output"] = input_

        inputs.append(input_)
        outputs.append(expected_)
    return inputs, outputs


def run(
    aio_benchmark,
    store_inputs_in_request,
    store_outputs_in_response,
    tensor_size_in_kb,
    data_plane_tracker,
):
    if data_plane_tracker.is_first_run:
        ucp.reset()
        data_plane_tracker._data_plane = UcpDataPlane()
        data_plane_tracker._data_plane.connect()

    request_plane = NatsRequestPlane(f"nats://localhost:{NATS_PORT}")
    asyncio.get_event_loop().run_until_complete(request_plane.connect())

    identity_operator = RemoteOperator(
        "identity", request_plane, data_plane_tracker._data_plane
    )

    inputs, outputs = _create_inputs(1, tensor_size_in_kb)

    aio_benchmark(
        post_requests,
        identity_operator,
        inputs,
        outputs,
        store_inputs_in_request,
        store_outputs_in_response,
    )

    timeout = 5
    asyncio.get_event_loop().run_until_complete(request_plane.close())

    if data_plane_tracker.is_last_run:
        data_plane_tracker._data_plane.close(timeout)


async def post_requests(
    identity_model, inputs, outputs, store_inputs_in_request, store_outputs_in_response
):
    results = []
    expected_results = {}

    for i, input_ in enumerate(inputs):
        request_id = str(i)
        request = identity_model.create_request(
            inputs={"input": input_}, request_id=request_id
        )
        if store_inputs_in_request:
            request.store_inputs_in_request.add("input")
        results.append(identity_model.async_infer(request))
        expected_results[request_id] = outputs[i]

    for result in asyncio.as_completed(results):
        responses = await result
        async for response in responses:
            for output_name, expected_value in expected_results[
                response.request_id
            ].items():
                output = response.outputs[output_name]
                _ = numpy.from_dlpack(output.to_host())

                del output

            del response


@pytest.fixture(scope="module")
def data_plane_tracker():
    class Tracker:
        def __init__(self):
            self.total_runs = 0
            self.current_run = 0
            self._data_plane = None

        def increment_run(self):
            self.current_run += 1

        @property
        def is_first_run(self):
            return self.current_run == 1

        @property
        def is_last_run(self):
            return self.current_run == self.total_runs

    return Tracker()


# FIXME: NATS default size limit is 1 MB. However, even when the tensor_size_in_kb
# is set as 600, which corresponds to 0.6144 MB, we are hiting MaxPayloadError.
# Need to investigate why the limit is being hit.
@pytest.mark.skipif(
    "(not os.path.exists('/usr/local/bin/nats-server'))",
    reason="NATS.io not present or test is configured to run with mock disaggregated_serving",
)
@pytest.mark.parametrize(
    ["store_inputs_in_request", "store_outputs_in_response"],
    [(True, True), (False, False)],
)
@pytest.mark.parametrize(
    "tensor_size_in_kb",
    [10, 100, 500],
)
@pytest.mark.benchmark(min_rounds=100, max_time=1)
def test_identity(
    request,
    nats_server,
    workers,
    aio_benchmark,
    store_inputs_in_request,
    store_outputs_in_response,
    tensor_size_in_kb,
    data_plane_tracker,
):
    """
    This benchmark test checks the latency of a simple operator which returns input in its output
    without any processing.
    NOTE: We can not use benchmark fixture in the child process. Hence, we are required to use the
    same process for opening then data plane object as pytest.
    This means that the pytest main process cannot create another data plane object in any other
    tests. Hence, we will use a run tracker to open and close the data plane
    """
    if data_plane_tracker.total_runs == 0:
        data_plane_tracker.total_runs = 6  # Set this to the number of parameters
    data_plane_tracker.increment_run()
    run(
        aio_benchmark,
        store_inputs_in_request,
        store_outputs_in_response,
        tensor_size_in_kb,
        data_plane_tracker,
    )
