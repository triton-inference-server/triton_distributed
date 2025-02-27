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

import os
import uuid

import bentoml
with bentoml.importing():
    # from triton_distributed_rs import DistributedRuntime, triton_endpoint, triton_worker
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.logger import logger as vllm_logger
    from vllm.sampling_params import RequestOutputKind
    from common.base_engine import BaseVllmEngine

    # from common.parser import parse_vllm_args
    from common.protocol import MyRequestOutput, vLLMGenerateRequest

from compoundai import async_onstart, nova_endpoint, service, tdist_context

vllm_logger.info(f"VLLM_KV_CAPI_PATH: {os.environ['VLLM_KV_CAPI_PATH']}")


lease_id = None


@service(
    nova={
        "enabled": True,
        "namespace": "triton-init",
    },
    resources={"gpu": 1}
)
class VllmEngine(BaseVllmEngine):
    """
    vLLM Inference Engine
    """

    def __init__(self):
        model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.engine_args = AsyncEngineArgs(
            model=model,
            gpu_memory_utilization=0.8,
            enable_prefix_caching=True,
            block_size=64,
            max_model_len=16384,
        )
        print(f"VLLM worker tdist_context: {tdist_context}")
        VLLM_WORKER_ID = uuid.UUID(int=tdist_context["endpoints"][0].lease_id())
        os.environ["VLLM_WORKER_ID"] = str(VLLM_WORKER_ID)
        print(f"VLLM worker id: {VLLM_WORKER_ID}")
        vllm_logger.info(f"Generate endpoint ID: {VLLM_WORKER_ID}")
        super().__init__(self.engine_args)

    @nova_endpoint()
    async def generate(self, request: vLLMGenerateRequest):
        if self.engine_client is None:
            await self.initialize()
            print("VLLM ENGINE INITIALIZED")
        assert self.engine_client is not None, "engine_client was not initialized"
        sampling_params = request.sampling_params
        # rust HTTP requires Delta streaming
        sampling_params.output_kind = RequestOutputKind.DELTA

        async for response in self.engine_client.generate(
            request.engine_prompt, sampling_params, request.request_id
        ):
            # MyRequestOutput takes care of serializing the response as
            # vLLM's RequestOutput is not serializable by default
            print("BEFORE VLLM ENGINE RESPONSE: ", response)
            resp = MyRequestOutput(
                request_id=response.request_id,
                prompt=response.prompt,
                prompt_token_ids=response.prompt_token_ids,
                prompt_logprobs=response.prompt_logprobs,
                outputs=response.outputs,
                finished=response.finished,
            ).model_dump_json()
            print("AFTER VLLM ENGINE RESPONSE: ", resp)
            yield resp
