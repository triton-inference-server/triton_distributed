import argparse
import os
from dataclasses import field
from typing import Optional, Any

from .vllm_disaggregated.stage_executor import PiplineStageExecutor
from .vllm_disaggregated.pipelines import PrefillStage, GenerateStage, SingleComputePipeline
from triton_distributed.icp.data_plane import DataPlane
from triton_distributed.icp.request_plane import RequestPlane
from triton_distributed.worker import Operator, RemoteInferenceRequest

class DummyOperator(Operator):

    def __init__(self, name: str, version: int, triton_core, request_plane: RequestPlane, data_plane: DataPlane,
                 parameters: Optional[dict[str, str | int | bool | bytes]] = field(
                     default_factory=dict
                 ), repository: Optional[str] = None, logger: Optional[Any] = None):
        print("DummyOperator init")
        #print("Os.env: ", os.environ)


    async def execute(self, requests: list[RemoteInferenceRequest]) -> None:
        print("DummyOperator execute")