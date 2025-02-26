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
import signal
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Tuple

import uvloop
from common.chat_processor import ChatProcessor, parse_chat_message_content
from common.parser import parse_tensorrt_llm_args
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
)
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

    def __init__(self, engine_args: Tuple[Dict[str, Any], Dict[str, Any]]):
        self.pytorch_config_args, self.llm_engine_args = engine_args
        self._init_engine()
        self.model = self.llm_engine_args["model"]
        if "tokenizer" in self.llm_engine_args.keys():
            tokenizer = self.llm_engine_args["tokenizer"]
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)

        self.chat_processor = ChatProcessor(self.model, self.tokenizer)

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
                llm = await loop.run_in_executor(
                    None,
                    lambda: LLM(
                        **self.llm_engine_args, pytorch_backend_config=pytorch_config
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

    @triton_endpoint(ChatCompletionRequest, ChatCompletionStreamResponse)
    async def generate(self, request):
        if self._llm_engine is None:
            raise RuntimeError("Engine not initialized")

        self._ongoing_request_count += 1
        logger.debug(f"Received request: {request}")
        request_id = str(uuid.uuid4())

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

            if self._llm_engine is not None:
                promise = self._llm_engine.generate_async(
                    prompt,
                    sampling_params,
                    streaming=request.stream,
                )
                if request.stream:
                    response_generator = self.chat_processor.stream_response(
                        request, request_id, conversation, promise
                    )
                    async for response in response_generator:
                        yield response
                else:
                    raise RuntimeError("Streaming is not supported")
            else:
                raise RuntimeError("Engine not initialized")
        except CppExecutorError:
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            raise RuntimeError("Failed to generate: " + str(e))
        finally:
            self._ongoing_request_count -= 1


@triton_worker()
async def worker(
    runtime: DistributedRuntime, engine_args: Tuple[Dict[str, Any], Dict[str, Any]]
):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("triton-init").component("tensorrt-llm")
    await component.create_service()

    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(TensorrtLLMEngine(engine_args).generate)


if __name__ == "__main__":
    uvloop.install()
    _, engine_args = parse_tensorrt_llm_args()
    asyncio.run(worker(engine_args))
