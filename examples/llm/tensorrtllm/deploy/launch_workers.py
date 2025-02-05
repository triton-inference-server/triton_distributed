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
import shutil
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


def _launch_mpi_workers(args):
    if (
        args.context_worker_count == 1
        or args.generate_worker_count >= 1
        or args.aggregate_worker_count == 1
    ):
        command = [
            "mpiexec",
            "--allow-run-as-root",
            "--oversubscribe",
            "--display-map",
            "--verbose",
        ]

        if args.log_dir:
            WORKER_LOG_DIR = str(Path(args.log_dir) / "workers")
            command += ["--output-filename", WORKER_LOG_DIR]

        aggregate_gpus = args.context_worker_count + args.generate_worker_count

        for index in range(args.context_worker_count):
            starting_gpu = index * aggregate_gpus
            command.extend(_context_cmd(args, starting_gpu))
            command.append(":")

        # [FIXME] the starting gpu is not count correctly,
        # i.e. if context_worker_count is 1, generate is 4,
        # starting GPUs are 1, 5, 9, 13, but should be 1, 2, 3, 4
        for index in range(args.generate_worker_count):
            starting_gpu = index * aggregate_gpus + args.context_worker_count
            command.extend(_generate_cmd(args, starting_gpu, index))
            command.append(":")

        for index in range(args.aggregate_worker_count):
            starting_gpu = index * aggregate_gpus + args.context_worker_count
            command.extend(_aggregate_cmd(args, starting_gpu))
            command.append(":")

        command = command[0:-1]
        print(" ".join(command))

        if args.dry_run:
            return

        env = os.environ.copy()
        return subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL)
    else:
        raise ValueError("Only supporting 1 worker each for now")


def _launch_disagg_model(args):
    if not args.disaggregated_serving:
        return

    starting_gpu = 0
    env = os.environ.copy()
    command = _disaggregated_serving_cmd(args, starting_gpu)
    print(" ".join(command))

    if args.dry_run:
        return

    return subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL)

def _launch_kv_aware_model(args):
    if not args.kv_aware_routing:
        return

    starting_gpu = 0
    env = os.environ.copy()
    command = _kv_aware_routing_cmd(args, starting_gpu)
    print(" ".join(command))

    if args.dry_run:
        return

    return subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL)


def _launch_workers(args):
    # Launch nats-server if requested by user for convenience, otherwise
    # it can be started separately beforehand.
    if args.initialize_request_plane:
        _launch_nats_server(args)
        # [FIXME] not really related to request plane
        _launch_etcd(args)

    # Launch TRT-LLM models via mpiexec in the same MPI WORLD
    _launch_mpi_workers(args)

    # [FIXME] below should be "one of" or merged together
    # Launch disaggregated serving "workflow" model to interface
    # client-facing requests with Triton Distributed deployment.
    _launch_disagg_model(args)

    # Launch KV aware routing "workflow" model to interface
    # client-facing requests with Triton Distributed deployment.
    _launch_kv_aware_model(args)


def _context_cmd(args, starting_gpu):
    # Hard-coded worker name for internal communication,
    # see tensorrtllm.deploy script
    worker_name = "context"
    command = [
        "-np",
        "1",
        # FIXME: May need to double check this CUDA_VISIBLE_DEVICES
        # and trtllm gpu_device_id/participant_id interaction
        # "-x",
        # f"CUDA_VISIBLE_DEVICES={starting_gpu}",
        "python3",
        "-m",
        "llm.tensorrtllm.deploy",
        "--worker-type",
        "context",
        "--worker-name",
        worker_name,
        "--model",
        args.model,
        "--gpu-device-id",
        f"{starting_gpu}",
        "--metrics-port",
        "50000",
        "--initialize-request-plane",
        "--request-plane-uri",
        f"{os.getenv('HOSTNAME')}:{args.nats_port}",
    ]

    return command


def _generate_cmd(args, starting_gpu, index):
    # Hard-coded worker name for internal communication
    # see tensorrtllm.deploy script
    worker_name = "generate"
    command = [
        "-np",
        "1",
        # FIXME: May need to double check this CUDA_VISIBLE_DEVICES
        # and trtllm gpu_device_id/participant_id interaction
        # "-x",
        # f"CUDA_VISIBLE_DEVICES={starting_gpu}",
        "python3",
        "-m",
        "llm.tensorrtllm.deploy",
        "--worker-type",
        "generate",
        "--worker-name",
        worker_name,
        "--model",
        args.model,
        "--gpu-device-id",
        f"{starting_gpu}",
        "--metrics-port",
        str(50001 + index * 10),
        "--request-plane-uri",
        f"{os.getenv('HOSTNAME')}:{args.nats_port}",
    ]

    return command


def _aggregate_cmd(args, starting_gpu):
    # Hard-coded worker name for internal communication
    # see tensorrtllm.deploy script
    worker_name = "aggregate"
    command = [
        "-np",
        "1",
        # FIXME: May need to double check this CUDA_VISIBLE_DEVICES
        # and trtllm gpu_device_id/participant_id interaction
        # "-x",
        # f"CUDA_VISIBLE_DEVICES={starting_gpu}",
        "python3",
        "-m",
        "llm.tensorrtllm.deploy",
        "--worker-type",
        "aggregate",
        "--worker-name",
        worker_name,
        "--model",
        args.model,
        "--gpu-device-id",
        f"{starting_gpu}",
        "--metrics-port",
        "50001",
        "--request-plane-uri",
        f"{os.getenv('HOSTNAME')}:{args.nats_port}",
    ]

    return command


def _disaggregated_serving_cmd(args, starting_gpu):
    # NOTE: This worker gets the args --worker-name because it will
    # receive the API-serving facing requests, and internally handle
    # the disaggregation. So this worker name should match the one
    # registered to the API Server.
    command = [
        # FIXME: Does this model need a GPU assigned to it?
        # "-x",
        # f"CUDA_VISIBLE_DEVICES={starting_gpu}",
        "python3",
        "-m",
        "llm.tensorrtllm.deploy",
        "--worker-type",
        "disaggregated-serving",
        "--metrics-port",
        "50002",
        "--model",
        args.model,
        "--worker-name",
        args.worker_name,
        "--request-plane-uri",
        f"{os.getenv('HOSTNAME')}:{args.nats_port}",
    ]

    return command

def _kv_aware_routing_cmd(args, starting_gpu):
    # NOTE: This worker gets the args --worker-name because it will
    # receive the API-serving facing requests, and internally handle
    # the disaggregation. So this worker name should match the one
    # registered to the API Server.
    command = [
        # FIXME: Does this model need a GPU assigned to it?
        # "-x",
        # f"CUDA_VISIBLE_DEVICES={starting_gpu}",
        "python3",
        "-m",
        "llm.tensorrtllm.deploy",
        "--worker-type",
        "kv-aware-routing",
        "--metrics-port",
        "50002",
        "--model",
        args.model,
        "--worker-name",
        args.worker_name,
        "--request-plane-uri",
        f"{os.getenv('HOSTNAME')}:{args.nats_port}",
    ]

    return command


def _launch_nats_server(args, clear_store=True):
    # FIXME: Use NatsServer object defined in icp package
    store_dir = "/tmp/nats_store"
    if clear_store:
        shutil.rmtree(store_dir, ignore_errors=True)

    command = [
        "/usr/local/bin/nats-server",
        "--jetstream",
        "--port",
        str(args.nats_port),
        "--store_dir",
        store_dir,
    ]

    print(" ".join(command))
    if args.dry_run:
        return

    env = os.environ.copy()
    return subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL)

def _launch_etcd(args):
    command = [
        "/usr/local/bin/etcd",
    ]

    print(" ".join(command))
    if args.dry_run:
        return

    env = os.environ.copy()
    return subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL)


if __name__ == "__main__":
    args = parse_args()
    _launch_workers(args)
