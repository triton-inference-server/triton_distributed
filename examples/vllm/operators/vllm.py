import argparse
from dataclasses import field
from typing import Optional, Any

from .vllm_disaggregated.stage_executor import PiplineStageExecutor
from .vllm_disaggregated.pipelines import PrefillStage, GenerateStage, SingleComputePipeline
from triton_distributed.icp.data_plane import DataPlane
from triton_distributed.icp.request_plane import RequestPlane
from triton_distributed.worker import Operator, RemoteInferenceRequest


class VllmContextOperator(Operator):

    def __init__(self, name: str, version: int, triton_core, request_plane: RequestPlane, data_plane: DataPlane,
                 parameters: Optional[dict[str, str | int | bool | bytes]] = field(
                     default_factory=dict
                 ), repository: Optional[str] = None, logger: Optional[Any] = None):
        args = argparse.Namespace(**parameters)
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
        self.executor = PiplineStageExecutor(args, stage, "prefill", "generate")

    async def execute(self, requests: list[RemoteInferenceRequest]) -> None:
        await self.executor.process_requests(requests, self.executor.process_request)


class VllmGenerateOperator(Operator):

    def __init__(self, name: str, version: int, triton_core, request_plane: RequestPlane, data_plane: DataPlane,
                 parameters: Optional[dict[str, str | int | bool | bytes]] = field(
                     default_factory=dict
                 ), repository: Optional[str] = None, logger: Optional[Any] = None):
        args = argparse.Namespace(**parameters)
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
        self.executor = PiplineStageExecutor(args, stage, "generate")

    async def execute(self, requests: list[RemoteInferenceRequest]) -> None:
        await self.executor.process_requests(requests)

class VllmBaselineOperator(Operator):

    def __init__(self, name: str, version: int, triton_core, request_plane: RequestPlane, data_plane: DataPlane,
                 parameters: Optional[dict[str, str | int | bool | bytes]] = field(
                     default_factory=dict
                 ), repository: Optional[str] = None, logger: Optional[Any] = None):
        args = argparse.Namespace(**parameters)
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
        self.executor = PiplineStageExecutor(args, stage, "baseline")

    async def execute(self, requests: list[RemoteInferenceRequest]) -> None:
        await self.executor.process_requests(requests)
