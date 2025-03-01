# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict
from tensorrt_llm.llmapi import KvCacheConfig, PyTorchConfig


class LLMAPIConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    model_name: str
    model_path: str | Optional = None
    pytorch_backend_config: PyTorchConfig | Optional = None
    kv_cache_config: KvCacheConfig | Optional = None


def _get_llm_args(engine_config):
    # Only do model validation checks and leave other checks to LLMAPI
    if "model" not in engine_config:
        raise ValueError("Model name is required in the TRT-LLM engine config.")

    if engine_config.get("model_path", ""):
        if os.path.exists(engine_config.get("model_path", "")):
            engine_config["model_path"] = Path(engine_config["model_path"])
        else:
            raise ValueError(f"Model path {engine_config['model_path']} does not exist")
    # We can initialize the sub configs needed
    if "pytorch_backend_config" in engine_config:
        engine_config["pytorch_backend_config"] = PyTorchConfig(
            **engine_config["pytorch_backend_config"]
        )

    if "kv_cache_config" in engine_config:
        engine_config["kv_cache_config"] = KvCacheConfig(
            **engine_config["kv_cache_config"]
        )

    return LLMAPIConfig(**engine_config)


def _init_engine_args(engine_args_filepath):
    """Initialize engine arguments from config file."""
    if not os.path.isfile(engine_args_filepath):
        raise ValueError(
            "'YAML file containing TRT-LLM engine args must be provided in when launching the worker."
        )

    try:
        with open(engine_args_filepath) as file:
            trtllm_engine_config = yaml.load(file)
    except yaml.YAMLError as e:
        raise RuntimeError(f"Failed to parse engine config: {e}")

    return _get_llm_args(trtllm_engine_config)


def parse_tensorrt_llm_args() -> Tuple[Any, Tuple[Dict[str, Any], Dict[str, Any]]]:
    parser = argparse.ArgumentParser(description="A TensorRT-LLM Worker parser")
    parser.add_argument(
        "--engine_args", type=str, required=True, help="Path to the engine args file"
    )
    parser.add_argument(
        "--llmapi-disaggregated-config",
        "-c",
        type=str,
        help="Path to the llmapi disaggregated config file",
        default=None,
    )
    args = parser.parse_args()
    return (args, _init_engine_args(args.engine_args))
