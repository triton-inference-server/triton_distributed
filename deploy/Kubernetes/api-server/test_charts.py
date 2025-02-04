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
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "chart, test",
    [
        ("OpenAI", "basic"),
        ("OpenAI", "basic_error"),
        ("OpenAI", "kubernetes_correct"),
        ("OpenAI", "kubernetes_invalid"),
    ],
)
def test_chart(chart, test):
    cmd_args = ["git", "rev-parse", "--show-toplevel"]

    repository_root_path = subprocess.check_output(cmd_args).decode("utf-8")
    repository_root_path = repository_root_path.strip()

    test_chart_path = os.path.join(
        repository_root_path,
        "deploy",
        "Kubernetes",
        "api-server",
        "tests",
        chart,
        "run.ps1",
    )

    print()
    print(f"Run {test_chart_path}")

    cmd_args = [
        "pwsh",
        "-c",
        test_chart_path,
        "test",
        "--test",
        test,
        "-v:detailed",
    ]

    assert subprocess.run(cmd_args).returncode == 0


if __name__ == "__main__":
    print(
        "Error: This script is not indented to executed direct. "
        "Instead use `pytest worker_tests.py` to execute it.",
        file=sys.stderr,
        flush=True,
    )
    exit(1)
