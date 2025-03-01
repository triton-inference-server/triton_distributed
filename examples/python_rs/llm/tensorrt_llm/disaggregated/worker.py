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


import asyncio
import os
import signal
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Tuple

import uvloop
from common.disagg_processor import (
    ChatProcessor,
    CompletionsProcessor,
    parse_chat_message_content,
)
from common.parser import parse_tensorrt_llm_args
from common.protocol import DisaggChatCompletionRequest, DisaggregatedResponse
from mpi4py.futures import MPICommExecutor
from mpi4py.MPI import COMM_WORLD
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm._utils import set_mpi_comm
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.llmapi import DisaggregatedParams, KvCacheConfig, MpiCommSession
from tensorrt_llm.llmapi.disagg_utils import (
    CtxGenServerConfig,
    DisaggServerConfig,
    parse_disagg_config_file,
    split_world_comm,
)
from tensorrt_llm.logger import logger
from transformers import AutoTokenizer

from triton_distributed.runtime import (
    DistributedRuntime,
    triton_endpoint,
    triton_worker,
)

logger.set_level("info")


class TensorrtLLMEngine:
    """
    Request handler for the generate endpoint
    """

    def __init__(
        self,
        engine_args: Tuple[Dict[str, Any], Dict[str, Any]],
        disagg_config: DisaggServerConfig,
        instance_idx: int,
        sub_comm,
    ):
        self.pytorch_config_args, self.llm_engine_args = engine_args
        self.disagg_config = disagg_config
        self.instance_idx = instance_idx
        self.server_config: CtxGenServerConfig = disagg_config.server_configs[
            instance_idx
        ]
        self.mpi_session = MpiCommSession(sub_comm, n_workers=sub_comm.Get_size())
        self._init_engine()
        self.model = self.llm_engine_args["model"]
        if "tokenizer" in self.llm_engine_args.keys():
            tokenizer = self.llm_engine_args["tokenizer"]
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.completions_processor = CompletionsProcessor(self.model)

    def _init_engine(self):
        logger.info("Initializing engine")

        # Run the engine in a separate thread running the AsyncIO event loop.
        self._llm_engine: Optional[Any] = None
        self._llm_engine_start_cv = threading.Condition()
        self._llm_engine_shutdown_event = asyncio.Event()
        self._event_thread = threading.Thread(
            target=asyncio.run, args=(self._run_llm_engine(),)
        )
        self._event_thread.start()
        with self._llm_engine_start_cv:
            while self._llm_engine is None:
                self._llm_engine_start_cv.wait()

        # The 'threading.Thread()' will not raise the exception here should the engine
        # failed to start, so the exception is passed back via the engine variable.
        if isinstance(self._llm_engine, Exception):
            e = self._llm_engine
            logger.error(f"Failed to start engine: {e}")
            if self._event_thread is not None:
                self._event_thread.join()
                self._event_thread = None
            raise e

    async def _run_llm_engine(self):
        # Counter to keep track of ongoing request counts.
        self._ongoing_request_count = 0

        @asynccontextmanager
        async def async_llm_wrapper():
            # Create LLM in a thread to avoid blocking
            loop = asyncio.get_running_loop()
            try:
                pytorch_config = PyTorchConfig(**self.pytorch_config_args)
                # TODO: maybe add build config
                llm = await loop.run_in_executor(
                    None,
                    lambda: LLM(
                        **self.llm_engine_args,
                        tensor_parallel_size=self.server_config.other_args["tp_size"],
                        pipeline_parallel_size=self.server_config.other_args["pp_size"],
                        gpus_per_node=None,
                        trust_remote_code=True,
                        _mpi_session=self.mpi_session,
                        kv_cache_config=KvCacheConfig(
                            free_gpu_memory_fraction=self.server_config.other_args[
                                "gpu_fraction"
                            ]
                        ),
                        pytorch_backend_config=pytorch_config,
                        backend="pytorch",
                    ),
                )
                yield llm
            finally:
                if "llm" in locals():
                    # Run shutdown in a thread to avoid blocking
                    await loop.run_in_executor(None, llm.shutdown)

        try:
            async with async_llm_wrapper() as engine:
                # Capture the engine event loop and make it visible to other threads.
                self._event_loop = asyncio.get_running_loop()

                # Signal the engine is started and make it visible to other threads.
                with self._llm_engine_start_cv:
                    self._llm_engine = engine
                    self._llm_engine_start_cv.notify_all()

                logger.info("Engine loaded and ready to serve...")

                # Wait for the engine shutdown signal.
                await self._llm_engine_shutdown_event.wait()

                # Wait for the ongoing requests to complete.
                while self._ongoing_request_count > 0:
                    logger.info(
                        "Awaiting remaining {} requests".format(
                            self._ongoing_request_count
                        )
                    )
                    await asyncio.sleep(1)

                # Cancel all tasks in the event loop.
                for task in asyncio.all_tasks(loop=self._event_loop):
                    if task is not asyncio.current_task():
                        task.cancel()

        except Exception as e:
            # Signal and pass the exception back via the engine variable if the engine
            # failed to start. If the engine has started, re-raise the exception.
            with self._llm_engine_start_cv:
                if self._llm_engine is None:
                    self._llm_engine = e
                    self._llm_engine_start_cv.notify_all()
                    return
            raise e

        self._llm_engine = None
        logger.info("Shutdown complete")

    @triton_endpoint(DisaggChatCompletionRequest, DisaggregatedResponse)
    async def generate(self, raw_request):
        if self._llm_engine is None:
            raise RuntimeError("Engine not initialized")

        request = DisaggChatCompletionRequest(**raw_request.model_dump())
        chat_processor = ChatProcessor(self.model, self.tokenizer, request)

        self._ongoing_request_count += 1
        logger.debug(f"Received request: {request}")

        try:
            conversation = []
            for message in request.messages:
                conversation.extend(parse_chat_message_content(message))
            tool_dicts = (
                None
                if request.tools is None
                else [tool.model_dump() for tool in request.tools]
            )
            prompt: str = self.tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt,
                tools=tool_dicts,
                documents=request.documents,
                chat_template=request.chat_template,
                **(request.chat_template_kwargs or {}),
            )
            sampling_params = request.to_sampling_params()
            disaggregated_params = DisaggregatedParams(**request.disaggregated_params)

            # Opaque state is  described as an additional state needing to be exchanged
            # between context and gen instances
            if disaggregated_params.opaque_state is not None:
                disaggregated_params.opaque_state = (
                    disaggregated_params.opaque_state.encode("utf-8")
                    .decode("unicode_escape")
                    .encode("latin1")
                )

            final_result = None
            async for result in self._llm_engine.generate_async(
                prompt,
                sampling_params,
                streaming=request.stream,
                disaggregated_params=disaggregated_params,
            ):
                final_result = result
                logger.debug(f"Generated result: {result}")
                if self.server_config.type == "ctx":
                    disaggregated_response = chat_processor.get_chat_stream_response(
                        request.id,
                        result,
                        first_iteration=True,
                    )
                    disaggregated_response.disaggregated_params = result.outputs[
                        0
                    ].disaggregated_params
                    yield disaggregated_response.model_dump_json()
                else:
                    yield chat_processor.get_chat_stream_response(
                        request.id,
                        result,
                        first_iteration=False,
                    ).model_dump_json(
                        exclude_unset=True, exclude={"disaggregated_params"}
                    )

            if request.stream_options and request.stream_options.include_usage:
                yield chat_processor.create_final_stream_response(
                    request.id,
                    final_result,
                ).model_dump_json(exclude_unset=True, exclude={"disaggregated_params"})

        except CppExecutorError:
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            raise RuntimeError("Failed to generate: " + str(e))

        self._ongoing_request_count -= 1


@triton_worker()
async def worker(
    runtime: DistributedRuntime,
    engine_args: Tuple[Dict[str, Any], Dict[str, Any]],
    disagg_config: DisaggServerConfig,
    instance_idx: int,
    sub_comm,
):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    server_type = disagg_config.server_configs[instance_idx].type
    logger.info(f"Starting {server_type} server")

    component = runtime.namespace("triton-init").component(
        f"tensorrt-llm-{server_type}"
    )
    await component.create_service()

    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(
        TensorrtLLMEngine(engine_args, disagg_config, instance_idx, sub_comm).generate
    )


if __name__ == "__main__":
    uvloop.install()
    args, engine_args = parse_tensorrt_llm_args()

    if args.llmapi_disaggregated_config is None or not os.path.exists(
        args.llmapi_disaggregated_config
    ):
        raise ValueError(
            "llmapi_disaggregated_config file does not exist or not provided"
        )

    disagg_config: DisaggServerConfig = parse_disagg_config_file(
        args.llmapi_disaggregated_config
    )

    logger.info(f"Parsed disaggregated config: {disagg_config}")

    is_leader, instance_idx, sub_comm = split_world_comm(disagg_config.server_configs)
    os.environ["TRTLLM_USE_MPI_KVCACHE"] = "1"
    set_mpi_comm(sub_comm)

    logger.info(f"is_leader: {is_leader}, instance_idx: {instance_idx}")

    if is_leader:
        asyncio.run(worker(engine_args, disagg_config, instance_idx, sub_comm))
    else:
        with MPICommExecutor(sub_comm) as executor:
            if not is_leader and executor is not None:
                raise RuntimeError(f"rank{COMM_WORLD} should not have executor")
