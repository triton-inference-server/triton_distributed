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
import argparse

ALLOWED_CI_OPTIONS = {
    # Boolean flags
    'run_jet_tests': {'type': 'bool'},
    'nightly_benchmark': {'type': 'bool'},
    'run_vllm': {'type': 'bool'},
    # Options with values
    'ci_default_branch': {'type': 'str'},
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--vllm-filter", type=str, required=True)
    parser.add_argument("--commit-message", type=str, required=True)
    return parser.parse_args()


def extract_options_from_commit(commit_message):
    """Extract CI options from commit message.
    Supports two formats:
    1. Boolean flags: [option]
    2. Valued options: [option=value]
    Example: [run_jet] [ci_default_branch=test123]
    """
    options = {}
    # Look for text between square brackets
    for part in commit_message.split('['):
        if ']' in part:
            content = part.split(']')[0].strip()
            if not content:  # Ignore empty brackets
                continue

            # Check if it's a key=value pair
            if '=' in content:
                key, value = content.split('=', 1)
                key = key.strip()
            else:
                key = content.strip()
                value = True

            # Normalize key: convert hyphens to underscores for consistent lookup
            normalized_key = key.replace('-', '_')

            if normalized_key not in ALLOWED_CI_OPTIONS:
                raise ValueError(f"Invalid CI option: '{key}'.\n\nAllowed options are:\n{sorted(ALLOWED_CI_OPTIONS)}")

            # Validate option type
            option_type = ALLOWED_CI_OPTIONS[normalized_key]['type']
            if option_type == 'bool' and value is not True:
                raise ValueError(f"Option '{key}' should not have a value")
            elif option_type == 'str' and value is True:
                raise ValueError(f"Option '{key}' requires a value")

            options[normalized_key] = value
    return options

if __name__ == "__main__":
    args = parse_args()
    try:
        ci_options = extract_options_from_commit(args.commit_message)
        print(f"Command line args: {args}")
        print(f"CI options from commit message: {ci_options}")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
