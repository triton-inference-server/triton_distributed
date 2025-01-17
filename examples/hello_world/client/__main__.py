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
import signal
import sys
import time

from .client import _start_client
from .parser import parse_args

processes = None


def handler(signum, frame):
    exit_code = 0
    if processes:
        print("Stopping Clients")
        for process in processes:
            process.terminate()
            process.kill()
            process.join()

            exit_code += process.exitcode
    print(f"Clients Stopped Exit Code {exit_code}")
    sys.exit(exit_code)


signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
for sig in signals:
    try:
        signal.signal(sig, handler)
    except Exception:
        pass


def main(args):
    global processes
    process_context = multiprocessing.get_context("spawn")
    args.lock = process_context.Lock()
    processes = []
    start_time = time.time()
    for index in range(args.clients):
        processes.append(
            process_context.Process(target=_start_client, args=(index, args))
        )
        processes[-1].start()

    for process in processes:
        process.join()
    end_time = time.time()
    print(
        f"Throughput: {(args.requests_per_client*args.clients)/(end_time-start_time)} Total Time: {end_time-start_time}"
    )
    exit_code = 0
    for process in processes:
        exit_code += process.exitcode
    print(f"Clients Stopped Exit Code {exit_code}")
    return exit_code


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))
