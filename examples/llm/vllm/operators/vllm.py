import argparse
import json
from dataclasses import field
from typing import Any, List, Optional

import numpy as np

from triton_distributed.icp.data_plane import DataPlane
from triton_distributed.icp.request_plane import RequestPlane
from triton_distributed.worker import Operator, RemoteInferenceRequest, RemoteOperator

from .vllm_disaggregated.pipelines import (
    AggregatedPipeline,
    GenerateStage,
    PrefillStage,
)


class BaseVllmOperator(Operator):
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
        self.name = name
        self.request_plane = request_plane
        self.logger = logger

    @staticmethod
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
        return inputs, parameters


class VllmContextOperator(BaseVllmOperator):
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
        super().__init__(
            name,
            version,
            triton_core,
            request_plane,
            data_plane,
            parameters,
            repository,
            logger,
        )
        args = argparse.Namespace(**parameters)  # type: ignore
        self._prefill_stage = PrefillStage(
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

        self._generate_operator = RemoteOperator("generate", request_plane, data_plane)

    async def execute(self, requests: List[RemoteInferenceRequest]) -> None:
        for request in requests:
            inputs, parameters = self._prepare_inputs(request)
            try:
                self.logger.info("Processing request")
                responses = list(
                    [
                        response
                        async for response in self._prefill_stage(
                            {
                                "inputs": inputs,
                                "parameters": parameters,
                            }
                        )
                    ]
                )
                self.logger.info("Prefill finished")
                assert len(responses) == 1
                response = responses[0]
                self.logger.info("Processing generate")
                generate_response = await self._generate_operator.async_infer(
                    inputs=response["outputs"],
                    parameters={**request.parameters, **response["parameters"]},
                )
                async for response in generate_response:
                    self.logger.info("Sending response")
                    await request.response_sender().send(
                        outputs=response.outputs,
                        parameters=response.parameters,
                        final=response.final,
                        error=response.error,
                    )
                    self.logger.info("Response send")
            except Exception as e:
                self.logger.error(f"Error processing request: {e}")
                await request.response_sender().send(error=e, final=True)


class VllmGenerateOperator(BaseVllmOperator):
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
        super().__init__(
            name,
            version,
            triton_core,
            request_plane,
            data_plane,
            parameters,
            repository,
            logger,
        )
        args = argparse.Namespace(**parameters)  # type: ignore
        args.worker_name = "generate"
        self.generate_stage = GenerateStage(
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

    async def execute(self, requests: List[RemoteInferenceRequest]) -> None:
        for request in requests:
            inputs, parameters = self._prepare_inputs(request)
            try:
                self.logger.debug("Processing request")
                async for response in self.generate_stage(
                    {
                        "inputs": inputs,
                        "parameters": parameters,
                    }
                ):
                    self.logger.debug("Sending response")
                    await request.response_sender().send(**response)
                    self.logger.debug("Response send")
            except Exception as e:
                self.logger.error(f"Error processing request: {e}")
                await request.response_sender().send(error=e, final=True)


class VllmBaselineOperator(BaseVllmOperator):
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
        super().__init__(
            name,
            version,
            triton_core,
            request_plane,
            data_plane,
            parameters,
            repository,
            logger,
        )
        args = argparse.Namespace(**parameters)  # type: ignore
        self.aggregated_pipeline = AggregatedPipeline(
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

    async def execute(self, requests: List[RemoteInferenceRequest]) -> None:
        for request in requests:
            inputs, parameters = self._prepare_inputs(request)
            try:
                self.logger.debug("Processing request")
                async for response in self.aggregated_pipeline(
                    {
                        "inputs": inputs,
                        "parameters": parameters,
                    }
                ):
                    self.logger.debug("Sending response")
                    await request.response_sender().send(**response)
                    self.logger.debug("Response send")
            except Exception as e:
                self.logger.error(f"Error processing request: {e}")
                await request.response_sender().send(error=e, final=True)
