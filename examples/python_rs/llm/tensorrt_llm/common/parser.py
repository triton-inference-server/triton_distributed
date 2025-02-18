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
import json
import os
from typing import Any, Dict, Tuple

# Define the expected keys for each config
# TODO: Add more keys as needed
PYTORCH_CONFIG_KEYS = {
    "use_cuda_graph",
    "cuda_graph_batch_sizes",
    "cuda_graph_max_batch_size",
    "cuda_graph_padding_enabled",
    "enable_overlap_scheduler",
    "kv_cache_dtype",
    "torch_compile_enabled",
    "torch_compile_fullgraph",
    "torch_compile_inductor_enabled",
}

LLM_ENGINE_KEYS = {
    "model",
    "tokenizer",
    "tokenizer_model",
    "skip_tokenizer_init",
    "trust_remote_code",
    "tensor_parallel_size",
    "dtype",
    "revision",
    "tokenizer_revision",
    "speculative_model",
    "enable_chunked_prefill",
    "disagg_config",
}


def _get_llm_args(args_dict):
    pytorch_config_args = {
        k: v for k, v in args_dict.items() if k in PYTORCH_CONFIG_KEYS and v is not None
    }
    llm_engine_args = {
        k: v for k, v in args_dict.items() if k in LLM_ENGINE_KEYS and v is not None
    }
    if "model" not in llm_engine_args:
        raise ValueError("Model name is required in the TRT-LLM engine config.")

    return (pytorch_config_args, llm_engine_args)


def _init_engine_args(engine_args_filepath):
    """Initialize engine arguments from config file."""
    if not os.path.isfile(engine_args_filepath):
        raise ValueError(
            f"'{engine_args_filepath}' containing TRT-LLM engine args must be provided in when launching the worker"
        )

    try:
        with open(engine_args_filepath) as file:
            trtllm_engine_config = json.load(file)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse engine config: {e}")

    return _get_llm_args(trtllm_engine_config)


def parse_tensorrt_llm_args() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    parser = argparse.ArgumentParser(description="A TensorRT-LLM Worker parser")
    parser.add_argument(
        "--engine_args", type=str, required=True, help="Path to the engine args file"
    )
    args = parser.parse_args()
    return _init_engine_args(args.engine_args)
