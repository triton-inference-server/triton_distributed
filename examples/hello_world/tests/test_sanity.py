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

import subprocess

import pytest

# TODO
# Decide if this should be
# pre merge, nightly, or weekly
pytestmark = pytest.mark.pre_merge


def test_sanity():
    deployment_command = [
        "python3",
        "-m",
        "hello_world.deploy",
        "--initialize-request-plane",
    ]

    deployment_process = subprocess.Popen(
        deployment_command,
        stdin=subprocess.DEVNULL,
    )

    client_command = [
        "python3",
        "-m",
        "hello_world.client",
        "--requests-per-client",
        "10",
    ]

    client_process = subprocess.Popen(
        client_command,
        stdin=subprocess.DEVNULL,
    )
    try:
        client_process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        print("Client timed out!")
        client_process.terminate()
        client_process.wait()

    client_process.terminate()
    client_process.kill()
    client_process.wait()
    deployment_process.terminate()
    deployment_process.wait()
    assert client_process.returncode == 0, "Error in clients!"
    assert deployment_process.returncode == 0, "Error starting deployment!"
