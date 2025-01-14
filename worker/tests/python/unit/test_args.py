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

import pytest
from triton_distributed.worker.parser import Parser

"""
Tests for parsing the arguments by command line parser
"""


@pytest.fixture
def default_values():
    # Add default values for the command-line interface
    return {
        "request_plane_uri": "nats://localhost:4222",
        "log_level": 0,
        # TODO: Add the default options for the worker executable here
    }


def test_parse_args_default(default_values):
    # Tests for default values
    args, parser = Parser.parse_args([])
    assert args.request_plane_uri == default_values["request_plane_uri"]
    assert args.log_level == default_values["log_level"]
    if args.operators:
        raise Exception(f"Expected no operators by default, got {args.operators}")

    if args.operator_configs:
        raise Exception(
            f"Expected no operators by default, got {args.operator_configs}"
        )


@pytest.mark.parametrize(
    "valid_request_plane_uri",
    [
        "https://example.com",
        # Add valid request plane uri values
    ],
)
def test_parse_args_valid_request_plane_uri(valid_request_plane_uri):
    # Tests with valid values for request plane uri
    args, _ = Parser.parse_args(["--request-plane-uri", valid_request_plane_uri])
    assert args.request_plane_uri == valid_request_plane_uri


def clean_argument_list(args_list):
    return [x for x in args_list if x is not None]


@pytest.mark.parametrize(
    "first_arg, second_arg, third_arg",
    [
        ("name:abc", "version:1", "max_inflight_requests:5"),
        ("name:abc", "max_inflight_requests:5", None),
        ("name:abc", "version:1", None),
        ("name:abc", None, None),
        # Add valid cases
    ],
)
def test_parse_args_valid_model(first_arg, second_arg, third_arg, tmp_path):
    model_repo_path = tmp_path / "model_repo"
    model_repo_path.mkdir()
    d = model_repo_path / "abc"
    d.mkdir()
    # Tests with valid arguments
    input_args = ["--operator"]
    model_args = clean_argument_list(
        [
            first_arg,
            second_arg,
            third_arg,
            f"repository:{model_repo_path}",
            "module:worker.triton_core_operator:TritonCoreOperator",
        ]
    )
    print(model_args)
    input_args = input_args + model_args
    args, _ = Parser.parse_args(input_args)
    assert args.operators[0] == model_args


def test_parse_args_invalid_operator(capsys):
    # Tests with  invalid arguments
    with pytest.raises(SystemExit):
        Parser.parse_args(["--operator"])
    captured = capsys.readouterr()
    assert "expected at least one argument" in captured.err


@pytest.mark.parametrize(
    "first_arg, second_arg, third_arg",
    [
        ("name:abc", "version:1", "max_inflight_requests:5"),
        ("name:abc", "max_inflight_requests:5", None),
        ("name:abc", "version:1", None),
        # TODO: Revisit can be uncommented once the operator module can be inferred automatically.
        # ("abc", None, None),
        # Add valid cases
    ],
)
def test_parse_args_valid_operator(first_arg, second_arg, third_arg, tmp_path):
    repo_path = tmp_path / "worker_repo"
    repo_path.mkdir()
    d = repo_path / "abc"
    d.mkdir()
    # Tests with valid arguments
    input_args = ["--operator"]
    operator_args = clean_argument_list([first_arg, second_arg, third_arg])
    input_args = input_args + operator_args + ["module:dummyworkflow:Workflow"]
    args, _ = Parser.parse_args(input_args)
    assert args.operators[0] == operator_args + ["module:dummyworkflow:Workflow"]
