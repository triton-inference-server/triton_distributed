import argparse
from dataclasses import field
from typing import Any, Optional

import json
import numpy as np

from triton_distributed.icp.data_plane import DataPlane
from triton_distributed.icp.request_plane import RequestPlane
from triton_distributed.worker import Operator, RemoteInferenceRequest

from .vllm_disaggregated.pipelines import (
    GenerateStage,
    PrefillStage,
    SingleComputePipeline,
)
from .vllm_disaggregated.stage_executor import PiplineStageExecutor


class VllmContextOperator(Operator):
    def __init__(
        self,
        name: str,
        version: int,
        triton_core,
        request_plane: RequestPlane,
        data_plane: DataPlane,
        parameters: Optional[dict[str, str | int | bool | bytes]] = field(
            default_factory=dict
        ),
        repository: Optional[str] = None,
        logger: Optional[Any] = None,
    ):
        args = argparse.Namespace(**parameters)  # type: ignore
        stage = PrefillStage(
            model=args.model_name,
            tensor_parallel_size=args.context_tp_size,
            generate_tensor_parallel_size=args.generate_tp_size,
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
        self.executor = PiplineStageExecutor(
            args, request_plane, stage, "prefill", "generate"
        )

    async def execute(self, requests: list[RemoteInferenceRequest]) -> None:
        for request in requests:
            stage_request = _prepare_inputs(request)
            return_result = request.response_sender().send
            await self.executor.process_request(stage_request, return_result)


class VllmGenerateOperator(Operator):
    def __init__(
        self,
        name: str,
        version: int,
        triton_core,
        request_plane: RequestPlane,
        data_plane: DataPlane,
        parameters: Optional[dict[str, str | int | bool | bytes]] = field(
            default_factory=dict
        ),
        repository: Optional[str] = None,
        logger: Optional[Any] = None,
    ):
        args = argparse.Namespace(**parameters)  # type: ignore
        args.worker_name = "generate"
        stage = GenerateStage(
            model=args.model_name,
            tensor_parallel_size=args.generate_tp_size,
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
        self.executor = PiplineStageExecutor(args, request_plane, stage, "generate")

    async def execute(self, requests: list[RemoteInferenceRequest]) -> None:
        for request in requests:
            stage_request = _prepare_inputs(request)
            return_result = request.response_sender().send
            await self.executor.process_request(stage_request, return_result)


class VllmBaselineOperator(Operator):
    def __init__(
        self,
        name: str,
        version: int,
        triton_core,
        request_plane: RequestPlane,
        data_plane: DataPlane,
        parameters: Optional[dict[str, str | int | bool | bytes]] = field(
            default_factory=dict
        ),
        repository: Optional[str] = None,
        logger: Optional[Any] = None,
    ):
        args = argparse.Namespace(**parameters)  # type: ignore
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
        self.executor = PiplineStageExecutor(args, request_plane, stage, "baseline")

    async def execute(self, requests: list[RemoteInferenceRequest]) -> None:
        for request in requests:
            stage_request = _prepare_inputs(request)
            return_result = request.response_sender().send
            await self.executor.process_request(stage_request, return_result)


def _prepare_inputs(request: RemoteInferenceRequest):
    inputs, parameters = {}, {}
    for input_name, input_data in request.inputs.items():
        local_tensor = input_data.local_tensor
        numpy_tensor = np.from_dlpack(local_tensor)
        input_data.__del__()
        inputs[input_name] = numpy_tensor
    for key, value in request.parameters.items():
        if isinstance(value, str) and value.startswith("JSON:"):
            parameters[key] = json.loads(value[5:])
        else:
            parameters[key] = value
    return {
        "inputs": inputs,
        "parameters": parameters,
    }