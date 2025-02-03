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

import argparse
import json
import os

from triton_distributed.worker.worker import OperatorConfig

# Default values
DEFAULT_REQUEST_PLANE_URI = "nats://localhost:4222"
DEFAULT_LOG_LEVEL = 0

# Property keys
NAME = "name"
VERSION = "version"
MAX_INFLIGHT_REQUESTS = "max_inflight_requests"
PARAMETERS = "parameters"
MODULE = "module"
REPOSITORY = "repository"
IMPLEMENTATION = "implementation"


class InvalidArgumentError(Exception):
    pass


def _parse_name_and_properties(args, valid_properties):
    kind = "operator"
    args_dict = {}
    if len(args) == 1:
        args_dict[NAME] = args[0]
    else:
        for arg in args:
            values = arg.split(":")
            if values[0] not in valid_properties:
                raise InvalidArgumentError(
                    f"Unexpected property found for `--{kind}` found. Expected one of {valid_properties}, found {values[0]}"
                )
            args_dict[values[0]] = ":".join(values[1:])
            if values[0] == PARAMETERS:
                parameter_file_path = args_dict[values[0]]
                if not os.path.exists(parameter_file_path):
                    args_dict[values[0]] = json.loads(args_dict[values[0]])
                else:
                    with open(parameter_file_path, "r") as f:
                        args_dict[values[0]] = json.load(f)
        if NAME not in args_dict.keys():
            raise InvalidArgumentError(
                f"`name` is a required property for `--{kind}`. Missing `name:<{kind}_name>`, in {args}"
            )

    return args_dict


def _validate_operator_args(operator_args):
    valid_properties = [
        NAME,
        VERSION,
        MAX_INFLIGHT_REQUESTS,
        MODULE,
        PARAMETERS,
        REPOSITORY,
    ]
    properties = _parse_name_and_properties(operator_args, valid_properties)

    for int_property in [VERSION, MAX_INFLIGHT_REQUESTS]:
        if int_property in properties.keys():
            try:
                int(properties[int_property])
            except ValueError:
                raise InvalidArgumentError(
                    f"Unexpected value provided for `{int_property}` for operator `{properties[NAME]}`. Expected an integer, Got  {properties[int_property]}"
                )

    if MODULE not in properties.keys():
        raise InvalidArgumentError(
            f"{MODULE} property not provided for operator `{properties[NAME]}`. This is a required property."
        )

    properties[IMPLEMENTATION] = properties[MODULE]
    properties.pop(MODULE)

    return properties


class Parser:
    @classmethod
    def _validate_args(cls, args):
        operator_configs: list[OperatorConfig] = []
        for operator_args in args.operators:
            operator_properties = _validate_operator_args(operator_args)
            operator_configs.append(OperatorConfig(**operator_properties))
        args.operator_configs = operator_configs

        # TODO: Add validation for request plane URI

    @classmethod
    def parse_args(cls, args=None):
        parser = argparse.ArgumentParser(description="Triton Worker Component")
        parser.add_argument(
            "-c",
            "--request-plane-uri",
            type=str,
            default=DEFAULT_REQUEST_PLANE_URI,
            help="Request plane URI for the worker",
        )
        parser.add_argument(
            "-l",
            "--log-level",
            type=int,
            default=DEFAULT_LOG_LEVEL,
            help="The logging level for Triton. The verbose logging can be enabled by specifying a value >= 1.",
        )

        parser.add_argument(
            "--log-dir",
            type=str,
            default=None,
            help="log dir folder",
        )

        parser.add_argument(
            "--triton-log-path",
            type=str,
            default=None,
            help="triton log path",
        )
        parser.add_argument(
            "--name",
            type=str,
            default=None,
            help="worker name",
        )
        parser.add_argument(
            "-op",
            "--operator",
            type=str,
            action="append",
            nargs="+",
            default=[],
            dest="operators",
            help="The operator to be hosted in the worker. The option can accept a single argument for the model name to load. Alternatively, it can also accept optional arguments in format `name:<model_name> version:<model_version>(optional) batch_size:<batch_size>(optional)`",
        )

        parser.add_argument(
            "--metrics-port",
            type=int,
            default=0,
            help="enable prometheus metrics for worker",
        )

        """
        TODO: Add more options as per requirements
        """

        args = parser.parse_args(args)
        try:
            Parser._validate_args(args)
        except Exception as err:
            parser.error(f"Failed to validate arguments {err=}, {type(err)=}")

        return args, cls
