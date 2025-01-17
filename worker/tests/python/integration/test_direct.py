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
import logging
import sys
import uuid
from multiprocessing import Process

import cupy
import numpy
import pytest
import ucp
from cupy_backends.cuda.api.runtime import CUDARuntimeError
from triton_distributed.icp.nats_request_plane import NatsRequestPlane
from triton_distributed.icp.ucp_data_plane import UcpDataPlane
from triton_distributed.worker.deployment import Deployment
from triton_distributed.worker.log_formatter import LOGGER_NAME
from triton_distributed.worker.operator import OperatorConfig
from triton_distributed.worker.remote_operator import RemoteOperator
from triton_distributed.worker.worker import WorkerConfig

NATS_PORT = 4223
MODEL_REPOSITORY = (
    "/workspace/worker/tests/python/integration/operators/triton_core_models"
)
OPERATORS_REPOSITORY = "/workspace/worker/tests/python/integration/operators"
TRITON_LOG_FILE = "triton.log"
TRITON_LOG_LEVEL = 6

logger = logging.getLogger(LOGGER_NAME)

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
def workers(request, log_dir, number_workers=10):
    # Add configs for identity operator
    operator_name = "identity"
    operator_config = OperatorConfig(
        name=operator_name,
        implementation="identity:Identity",
        version=1,
        max_inflight_requests=10,
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
                request_plane=NatsRequestPlane,
                data_plane=UcpDataPlane,
                request_plane_args=(
                    [],
                    {"request_plane_uri": f"nats://localhost:{NATS_PORT}"},
                ),
                log_level=TRITON_LOG_LEVEL,
                log_dir=str(worker_log_dir),
                triton_log_path=str(worker_log_dir / TRITON_LOG_FILE),
                operators=[operator_config],
            )
        )

    worker_deployment = Deployment(worker_configs)

    worker_deployment.start()
    yield worker_deployment
    worker_deployment.shutdown()


async def post_requests(num_requests, num_targets):
    """
    Posts requests until the number of
    workers that respond is equal to the number of targets
    after that - only sends requests to one of the targets
    """
    ucp.reset()
    timeout = 5

    data_plane = UcpDataPlane()
    data_plane.connect()

    request_plane = NatsRequestPlane(f"nats://localhost:{NATS_PORT}")
    await request_plane.connect()

    identity_operator = RemoteOperator("identity", 1, request_plane, data_plane)

    target_components = set()
    target_component_list: list[uuid.UUID] = []
    responding_components = set()

    for index in range(num_requests):
        request = identity_operator.create_request(
            inputs={"input": [index]},
        )
        target_component = None

        if target_component_list:
            # we have the list of targets
            # only send to workers in that list
            target_index = index % len(target_component_list)
            target_component = target_component_list[target_index]
            identity_operator.component_id = target_component

        async for response in await identity_operator.async_infer(request):
            responding_component = response.component_id
            numpy.testing.assert_equal(
                numpy.from_dlpack(response.outputs["output"]), request.inputs["input"]
            )
            responding_components.add(responding_component)

            if not target_component_list:
                # add to list of acceptable targets
                target_components.add(responding_component)

        if len(target_components) >= num_targets:
            # finalize list
            target_component_list = list(target_components)

    timeout = 5
    data_plane.close(timeout)
    await request_plane.close()
    assert target_components == responding_components


def run(num_requests, num_targets=5):
    sys.exit(
        asyncio.run(
            post_requests(
                num_requests=num_requests,
                num_targets=num_targets,
            )
        )
    )


@pytest.mark.skipif(
    "(not os.path.exists('/usr/local/bin/nats-server'))",
    reason="NATS.io not present",
)
@pytest.mark.timeout(30)
def test_direct(request, nats_server, workers):
    # Using a separate process to use data plane across multiple tests.
    p = Process(target=run, args=(50,))
    p.start()
    p.join()
    assert p.exitcode == 0
