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
import os

import requests

ALLOWED_CI_OPTIONS = {
    # Boolean flags
    "RUN_JET_TESTS": {"type": "bool"},
    "NIGHTLY_BENCHMARK": {"type": "bool"},
    "RUN_VLLM": {"type": "bool"},
    # Options with values
    "CI_DEFAULT_BRANCH": {"type": "str"},  # placeholder for future use
}


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--vllm-filter", type=str, required=True)
    parser.add_argument("--commit-message", type=str, required=True)
    return parser.parse_args()


def extract_options_from_commit(commit_message):
    """
    Extract CI options from commit message.
    Supports two formats:
    1. Boolean flags: [option]
    2. Valued options: [option=value]
    """
    options = {}
    # Look for text between square brackets
    for part in commit_message.split("["):
        if "]" in part:
            content = part.split("]")[0].strip()
            if not content:  # Ignore empty brackets
                continue

            # Check if it's a key=value pair
            if "=" in content:
                key, value = content.split("=", 1)
                key = key.strip()
            else:
                key = content.strip()
                value = True

            # Normalize key: convert hyphens to underscores and uppercase for consistent lookup
            normalized_key = key.replace("-", "_").upper()

            if normalized_key not in ALLOWED_CI_OPTIONS:
                print(f"Warning: Ignoring invalid CI option: '{key}'. Not in allowed options list {ALLOWED_CI_OPTIONS}.")
                continue

            # Validate option type
            option_type = ALLOWED_CI_OPTIONS[normalized_key]["type"]
            if option_type == "bool" and value is not True:
                raise ValueError(f"Option '{key}' should not have a value")
            elif option_type == "str" and value is True:
                raise ValueError(f"Option '{key}' requires a value")

            options[normalized_key] = value
    return options


def add_path_filter(ci_options, vllm_filter):
    """
    Add path filter result to CI options
    """
    if vllm_filter == 'true':
        print(f"Detected changes in VLLM path filter")
        ci_options["RUN_VLLM"] = True
    return ci_options


def run_ci(ref, ci_options):
    """
    Trigger CI pipeline using the extracted options.
    """
    # Get secrets from environment variables
    pipeline_token = os.environ.get("PIPELINE_TOKEN")
    pipeline_url = os.environ.get("PIPELINE_URL")

    if not pipeline_token or not pipeline_url:
        raise ValueError(
            "Missing required environment variables: PIPELINE_TOKEN and/or PIPELINE_URL"
        )

    # Convert boolean values to strings
    variables = {}
    for key, value in ci_options.items():
        variables[key] = str(value).lower() if isinstance(value, bool) else str(value)

    # Prepare request data as JSON
    json_data = {
        "token": pipeline_token,
        "ref": ref,
        "variables": variables,
        "description": f"Triggered from GitHub Actions - {ref}"
    }

    # Send the request as JSON
    headers = {"Content-Type": "application/json"}
    response = requests.post(pipeline_url, json=json_data, headers=headers)

    response.raise_for_status()
    print(f"CI pipeline triggered successfully: {response.text}")


if __name__ == "__main__":
    args = parse_args()
    try:
        ci_options = extract_options_from_commit(args.commit_message)
        ci_options = add_path_filter(ci_options, args.vllm_filter)
        print(f"Command line args: {args}")
        print(f"CI options from commit message: {ci_options}")
        run_ci(args.ref, ci_options)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error triggering CI pipeline: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)
