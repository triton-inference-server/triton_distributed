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
import time

from triton_distributed.worker import (
    Deployment,
    OperatorConfig,
    TritonCoreOperator,
    WorkerConfig,
)


async def main():
    #    nats_server = NatsServer()
    time.sleep(1)

    encoder_op = OperatorConfig(
        name="encoder",
        repository="/workspace/examples/hello_world/operators/triton_core_models",
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
        repository="/workspace/examples/hello_world/operators/triton_core_models",
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
        implementation="EncodeDecodeOperator",
        max_inflight_requests=100,
        repository="/workspace/examples/hello_world/operators",
    )

    encoder = WorkerConfig(
        #        request_plane_args=([nats_server.url], {}),
        #        log_level=6,
        operators=[encoder_op],
        name="encoder",
        metrics_port=8060,
        #       log_dir="logs",
    )

    decoder = WorkerConfig(
        #      request_plane_args=([nats_server.url], {}),
        #     log_level=6,
        operators=[decoder_op],
        name="decoder",
        metrics_port=8061,
        #    log_dir="logs",
    )

    encoder_decoder = WorkerConfig(
        #   request_plane_args=([nats_server.url], {}),
        #  log_level=6,
        operators=[encoder_decoder_op],
        name="encoder_decoder",
        metrics_port=8062,
        # log_dir="logs",
    )

    print("Starting Workers")

    deployment = Deployment(
        [(encoder, 5), decoder, (encoder_decoder, 6)], initialize_request_plane=True
    )

    deployment.start()

    print("Sending Requests")

    #    await send_requests(nats_server.url)

    print("Stopping Workers")

    deployment.stop()


if __name__ == "__main__":
    asyncio.run(main())
