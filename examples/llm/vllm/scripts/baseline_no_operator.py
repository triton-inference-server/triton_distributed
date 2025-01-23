#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging

from operators.vllm_disaggregated.args_utils import parse_args

# from triton_distributed.icp.ucp_data_plane import get_ucp_data_plane_singleton
from ..operators.vllm_disaggregated.pipelines import SingleComputePipeline
from ..operators.vllm_disaggregated.stage_executor import PiplineStageExecutor

LOGGER = logging.getLogger(__name__)


# TODO this is a temporary workaround to avoid deadlocks in the UCP data plane
# get_ucp_data_plane_singleton(keep_endpoints_open=True).connect()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format=args.log_format)

    LOGGER.info(
        f"Start single compute pipeline {args.model_name} tp: {args.baseline_tp_size}"
    )
    stage = SingleComputePipeline(
        model=args.model_name,
        tensor_parallel_size=args.baseline_tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enforce_eager=args.enforce_eager,
        ignore_eos=args.ignore_eos,
        max_num_seqs=args.max_num_seqs,
        disable_async_output_proc=args.disable_async_output_proc,
        disable_log_stats=args.disable_log_stats,
    )
    # asyncio.run(handle_requests(args, stage, "baseline"))

    executor = PiplineStageExecutor(args, stage, "baseline")
    asyncio.run(executor.handle_pipelined_requests())
