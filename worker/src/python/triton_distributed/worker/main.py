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

import sys

from triton_distributed.icp.nats_request_plane import NatsRequestPlane
from triton_distributed.icp.ucp_data_plane import UcpDataPlane
from triton_distributed.worker.parser import Parser
from triton_distributed.worker.worker import Worker


def main(args=None):
    args, cli_parser = Parser.parse_args(args)

    # TODO: Revisit the worklow args. To simplify.
    worker = Worker(
        request_plane=NatsRequestPlane(args.request_plane_uri),
        data_plane=UcpDataPlane(),
        log_level=args.log_level,
        operators=cli_parser.operator_configs,
        metrics_port=args.metrics_port,
        log_dir=args.log_dir,
        name=args.name,
        triton_log_path=args.triton_log_path,
    )

    worker.start()


if __name__ == "__main__":
    sys.exit(main())
