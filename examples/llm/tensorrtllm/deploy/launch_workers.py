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

import os
import signal
import subprocess
import sys
from pathlib import Path

from llm.tensorrtllm.deploy.parser import parse_args

deployment = None


def handler(signum, frame):
    exit_code = 0
    if deployment:
        print("Stopping Workers")
        exit_code = deployment.stop()
    print(f"Workers Stopped Exit Code {exit_code}")
    sys.exit(exit_code)


signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
for sig in signals:
    try:
        signal.signal(sig, handler)
    except Exception:
        pass


def _launch_workers(args):
    if args.context_worker_count == 1 or args.generate_worker_count == 1:
        processes = []

        if args.initialize_request_plane:
            processes.append(_launch_nats_server(args))

        WORKER_LOG_DIR = str(Path(args.log_dir) / "workers")
        command = [
            "mpiexec",
            "--allow-run-as-root",
            "--oversubscribe",
            "--output-filename",
            WORKER_LOG_DIR,
            "--display-map",
            "--verbose",
        ]

        aggregate_gpus = args.context_worker_count + args.generate_worker_count

        for index in range(args.context_worker_count):
            starting_gpu = index * aggregate_gpus
            command.extend(_context_cmd(args, starting_gpu))
            command.append(":")

        for index in range(args.generate_worker_count):
            starting_gpu = index * aggregate_gpus + args.context_worker_count
            command.extend(_generate_cmd(args, starting_gpu))
            command.append(":")

        if args.disaggregated_serving:
            starting_gpu = 0
            command.extend(_disaggregated_serving_cmd(args, starting_gpu))
            command.append(":")

        command = command[0:-1]

        print(" ".join(command))

        if args.dry_run:
            return

        env = os.environ.copy()
        return subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL)
    else:
        raise ValueError("Only supporting 1 worker each for now")


def _context_cmd(args, starting_gpu):
    command = [
        "-np",
        "1",
        "-x",
        f"CUDA_VISIBLE_DEVICES={starting_gpu}",
        "python3",
        "-m",
        "llm.tensorrtllm.deploy",
        "--context-worker-count",
        "1",
        "--worker-name",
        "llama",
        "--initialize-request-plane",
        "--request-plane-uri",
        f"{os.getenv('HOSTNAME')}:{args.nats_port}",
    ]

    return command


def _generate_cmd(args, starting_gpu):
    command = [
        "-np",
        "1",
        "-x",
        f"CUDA_VISIBLE_DEVICES={starting_gpu}",
        "python3",
        "-m",
        "llm.tensorrtllm.deploy",
        "--generate-worker-count",
        "1",
        "--worker-name",
        "llama",
        "--request-plane-uri",
        f"{os.getenv('HOSTNAME')}:{args.nats_port}",
    ]

    return command


def _disaggregated_serving_cmd(args, starting_gpu):
    command = [
        "-np",
        "1",
        "-x",
        f"CUDA_VISIBLE_DEVICES={starting_gpu}",
        "python3",
        "-m",
        "llm.tensorrtllm.deploy",
        "--disaggregated-serving",
        "--starting-metrics-port",
        "50002",
        "--worker-name",
        "llama",
        "--request-plane-uri",
        f"{os.getenv('HOSTNAME')}:{args.nats_port}",
    ]

    return command


def _launch_nats_server(args):
    command = [
        "/usr/local/bin/nats-server",
        "--jetstream",
        "--port",
        args.nats_port,
    ]

    if args.dry_run:
        print(" ".join(command))
        return

    env = os.environ.copy()
    process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL)
    return process


if __name__ == "__main__":
    args = parse_args()
    _launch_workers(args)
