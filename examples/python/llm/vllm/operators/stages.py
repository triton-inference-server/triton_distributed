# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
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

import abc
import inspect
import os
import time
from typing import Any, AsyncGenerator, Dict, Optional

import numpy as np
import vllm.engine.arg_utils
import vllm.engine.async_llm_engine
import vllm.inputs.data

LOGGER = vllm.logger.init_logger(__name__)


# TODO ptarasiewicz remove after veryfing streaming works efficiently
# FIXME currently streaming all the tokens is not efficient
# with RETURN_EVERY_N so large we return only first token and whole sequence at the end
RETURN_EVERY_N = 1


class Stage(abc.ABC):
    @abc.abstractmethod
    async def __call__(
        self, input_payload: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        yield {}


class AggregatedStage(Stage):
    def __init__(
        self,
        **kwargs,
    ):
        self._ignore_eos = kwargs.pop("ignore_eos", False)
        engine_args = vllm.engine.arg_utils.AsyncEngineArgs(**kwargs)
        LOGGER.info(f"Creating engine with args: {engine_args}")
        self._engine = vllm.engine.async_llm_engine.AsyncLLMEngine.from_engine_args(
            engine_args
        )
        LOGGER.info(f"Created engine: {self._engine}")

    async def __call__(
        self, input_payload: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            vllm_input = input_payload["parameters"]["prompt"]
            sampling_params = vllm.SamplingParams(
                **input_payload["parameters"].get("sampling_params", {}),
                ignore_eos=self._ignore_eos,
            )
            LOGGER.debug(f"sampling_params: {sampling_params}")
            request_id = input_payload["parameters"].get("request_id", None)

            results_generator = self._engine.generate(
                vllm_input, sampling_params, request_id
            )
            LOGGER.debug("results_generator started")
            counter = 0
            async for result in results_generator:
                if counter % RETURN_EVERY_N == 0 or result.finished:
                    tokens_ids = np.stack(
                        [output_row.token_ids for output_row in result.outputs]
                    ).astype(np.int64)
                    LOGGER.debug(f"tokens_ids: {tokens_ids.shape}")
                    yield {
                        "outputs": {},
                        "error": None,
                        "final": result.finished,
                        "parameters": {
                            "text": result.outputs[0].text,
                        },
                    }
                counter += 1
            LOGGER.debug("results_generator finished")
        except Exception as e:
            LOGGER.error(f"Exception in SingleComputePipeline: {e}")
            yield {"outputs": {}, "error": str(e), "final": True}


class PrefillStage(Stage):
    def __init__(
        self,
        generate_tensor_parallel_size: Optional[int] = None,
        **kwargs,
    ):
        context_tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)
        generate_tensor_parallel_size = (
            generate_tensor_parallel_size or context_tensor_parallel_size
        )
        assert (
            generate_tensor_parallel_size % context_tensor_parallel_size == 0
        ), "generate_tensor_parallel_size must be multiple of context_tensor_parallel_size"
        LOGGER.debug(f"context_tensor_parallel_size: {context_tensor_parallel_size}")
        LOGGER.debug(f"generate_tensor_parallel_size: {generate_tensor_parallel_size}")
        os.environ["VLLM_DISAGG_STAGE"] = "PREFILL"
        os.environ["VLLM_CONTEXT_TP_SIZE"] = str(context_tensor_parallel_size)
        os.environ["VLLM_GENERATE_TP_SIZE"] = str(generate_tensor_parallel_size)
        LOGGER.info(f"Env VLLM_DISAGG_STAGE set to {os.environ['VLLM_DISAGG_STAGE']}")
        kwargs[
            "enforce_eager"
        ] = True  # Prefill stage must be eager because of variable ISL
        self._ignore_eos = kwargs.pop("ignore_eos", False)
        engine_args = vllm.engine.arg_utils.AsyncEngineArgs(**kwargs)
        LOGGER.info(f"Creating engine with args: {engine_args}")
        self._engine = vllm.engine.async_llm_engine.AsyncLLMEngine.from_engine_args(
            engine_args
        )
        LOGGER.info("Prefill stage initialized")

    async def __call__(
        self, input_payload: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            vllm_input = input_payload["parameters"]["prompt"]
            request_id = input_payload["parameters"].get("request_id", None)
            assert request_id is not None, "request_id is required for prefill"

            sampling_params = vllm.SamplingParams(
                **input_payload["parameters"].get("sampling_params", {}),
                ignore_eos=self._ignore_eos,
            )

            old_my_max_tokens = sampling_params.max_tokens
            old_my_min_tokens = sampling_params.min_tokens
            sampling_params.max_tokens = 1
            sampling_params.min_tokens = 1

            LOGGER.debug(f"sampling_params: {sampling_params}")

            start_time_ns = time.monotonic_ns()
            results_generator = self._engine.generate(
                vllm_input, sampling_params, request_id
            )
            LOGGER.debug("results_generator started")
            async for result in results_generator:
                taken_ms = (time.monotonic_ns() - start_time_ns) / 1_000_000
                LOGGER.info(
                    "==== Prefill completed kv cache taken %0.3fms ====", taken_ms
                )

                # TODO: needed to pass prompt, request_id, sampling_params to the next stage as there is no pipeline concept in online scenario
                sampling_params.max_tokens = old_my_max_tokens
                sampling_params.min_tokens = old_my_min_tokens
                sampling_params_init_names = inspect.signature(
                    vllm.SamplingParams
                ).parameters.keys()
                sampling_params = {
                    k: v
                    for k, v in sampling_params.__dict__.items()
                    if k in sampling_params_init_names
                }
                LOGGER.debug(
                    f"Yield response {input_payload['inputs'].keys()} parameters {input_payload['parameters']}"
                )
                yield {
                    "outputs": {},  # See line 195 for context
                    "error": None,
                    "parameters": {
                        "context_worker_id": os.environ["VLLM_WORKER_ID"],
                        "first_token": result.outputs[0].token_ids[0],
                        "seq_len": len(result.prompt_token_ids),
                    },
                    "final": True,
                }
            LOGGER.debug("Results generator for prefill finishes")
        except Exception as e:
            LOGGER.error(f"Exception in SingleComputePipeline: {e}")
            yield {"outputs": {}, "error": str(e), "final": True}


class GenerateStage(Stage):
    def __init__(
        self,
        **kwargs,
    ):
        os.environ["VLLM_DISAGG_STAGE"] = "GENERATE"
        LOGGER.info(f"Env VLLM_DISAGG_STAGE set to {os.environ['VLLM_DISAGG_STAGE']}")
        self._ignore_eos = kwargs.pop("ignore_eos", False)
        engine_args = vllm.engine.arg_utils.AsyncEngineArgs(**kwargs)
        LOGGER.info(f"Creating engine with args: {engine_args}")
        self._engine = vllm.engine.async_llm_engine.AsyncLLMEngine.from_engine_args(
            engine_args
        )
        LOGGER.info("Generation stage initialized")

    async def __call__(
        self, input_payload: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        seq_len = input_payload["parameters"]["seq_len"]
        LOGGER.debug(f"input sequence length: {seq_len}")
        # we can use any tokens because first token is already sampled by the context worker
        # and we just need the correct shape to allocate space in the kv cache
        vllm_input = vllm.inputs.data.TokensPrompt(prompt_token_ids=[0] * seq_len)
        sampling_params = vllm.SamplingParams(
            **input_payload["parameters"].get("sampling_params", {}),
            ignore_eos=self._ignore_eos,
        )
        LOGGER.debug(f"sampling_params: {sampling_params}")
        request_id = input_payload["parameters"].get("request_id", None)
        assert request_id is not None, "request_id is required for generate"
        context_worker_id = input_payload["parameters"]["context_worker_id"]
        new_request_id = f"{request_id}___{context_worker_id}"
        first_token = input_payload["parameters"]["first_token"]
        self._engine.engine.model_executor.driver_worker.model_runner.set_first_token(
            new_request_id, first_token
        )

        # TODO ptarasiewicz this is only temporary way to pass worker id to the engine
        # so that it can pull the correct kv cache
        results_generator = self._engine.generate(
            vllm_input,
            sampling_params,
            new_request_id,
        )
        LOGGER.debug("results_generator started")
        counter = 0
        async for result in results_generator:
            if counter % RETURN_EVERY_N == 0 or result.finished:
                yield {
                    "outputs": {},
                    "error": None,
                    "final": result.finished,
                    "parameters": {
                        "text": result.outputs[0].text,
                    },
                }
            counter += 1
        LOGGER.debug("results_generator finished for generate")
