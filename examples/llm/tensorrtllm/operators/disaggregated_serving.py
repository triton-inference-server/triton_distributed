# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging

import numpy

from triton_distributed.worker import (
    RemoteInferenceRequest,
    RemoteOperator,
    TritonCoreOperator,
)

logger = logging.getLogger(__name__)


class DisaggregatedServingOperator(TritonCoreOperator):
    def __init__(
        self,
        name,
        version,
        triton_core,
        request_plane,
        data_plane,
        parameters,
        repository,
        logger,
    ):
        self._prefill = RemoteOperator("context", request_plane, data_plane)
        self._decode = RemoteOperator("generate", request_plane, data_plane)

        self._repository = repository
        self._triton_core.register_model_repository(repository)
        self._preprocess_model = self._triton_core.load("preprocessing")
        self._postprocess_model = self._triton_core.load("postprocessing")
        self._logger = logger

    async def execute(self, requests: list[RemoteInferenceRequest]):
        logger.debug("Executing DisaggregatedServing Request")
        background_tasks = []
        for request in requests:
            task = asyncio.create_task(self._execute_request(request))
            background_tasks.append(task)

        try:
            results = await asyncio.gather(*background_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.exception(f"Running request execution failed: {result}")
                else:
                    logger.debug(f"Request execution completed with result: {result}")
        except Exception as e:
            logger.exception(f"Error during request execution: {e}")

    async def _execute_request(self, request: RemoteInferenceRequest):
        background_tasks = []
        prefill_inputs = {}

        """Preprocessing"""
        query = request.inputs["text_input"].to_bytes_array()
        streaming = request.parameters.get("streaming", False)
        input_ids, input_lengths = await self._preprocess(query)

        prefill_inputs["input_ids"] = input_ids
        prefill_inputs["input_lengths"] = input_lengths
        prefill_inputs["request_output_len"] = request.inputs["max_tokens"]

        """Prefill"""
        prefill_parameters = {}
        prefill_parameters["request_type"] = "context_only"
        logger.debug(
            f"Executing request on context worker with inputs: {prefill_inputs}"
        )

        response_sender = request.response_sender()

        async for prefill_response in await self._prefill.async_infer(
            inputs=prefill_inputs,
            parameters=prefill_parameters,
        ):
            logger.debug(f"Prefill response completed: {prefill_response}")
            output_ids = numpy.from_dlpack(prefill_response["output_ids"])
            logger.debug(f"Output IDs: {output_ids}")
            if streaming:
                tasks = asyncio.create_task(
                    self._send_llm_response(
                        prefill_response, response_sender, final=False
                    )
                )
                background_tasks.append(tasks)

            """Decode"""
            decode_parameters = {}
            decode_parameters["request_type"] = "generation_only"
            decode_inputs = {}

            decode_inputs["context_phase_params"] = prefill_response.outputs[
                "context_phase_params"
            ]

            decode_inputs["input_ids"] = input_ids
            decode_inputs["input_lengths"] = input_lengths
            decode_inputs["request_output_len"] = request.inputs["max_tokens"]

            async for decode_response in await self._decode.async_infer(
                inputs=decode_inputs,
                parameters=decode_parameters,
            ):
                logger.debug(f"Decode response completed: {decode_response}")
                background_tasks.append(
                    asyncio.create_task(
                        self._send_llm_response(
                            decode_response,
                            response_sender,
                            final=decode_response.final,
                        )
                    )
                )

        try:
            results = await asyncio.gather(*background_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.exception(
                        f"Sending response failed with exception: {result}"
                    )
                else:
                    logger.debug(f"Response sent successfully: {result}")
        except Exception as e:
            logger.exception(f"Error during response sending: {e}")

        for output in prefill_response.outputs:
            del output
        for output in decode_response.outputs:
            del output

    async def _preprocess(self, query):
        start_ids = None
        start_lengths = None
        async for preprocess_response in await self._preprocess_model.async_infer(
            inputs={"query": query}
        ):
            logger.debug(f"Preprocess response completed: {preprocess_response}")
            start_ids = numpy.from_dlpack(preprocess_response.outputs["start_ids"])
            start_lengths = numpy.from_dlpack(
                preprocess_response.outputs["start_lengths"]
            )

        return start_ids, start_lengths

    async def _postprocessing(self, tokens_batch, sequence_lengths):
        outputs = []
        async for postprocess_response in self._postprocess_model.async_infer(
            inputs={"tokens_batch": tokens_batch, "sequence_lengths": sequence_lengths}
        ):
            logger.debug(f"Received postprocess response: {postprocess_response}")
            output = postprocess_response.outputs["output"].to_string_array()
            outputs.append(output)

        return outputs

    async def send_llm_response(self, llm_response, response_sender, final):
        tokens_batch = numpy.from_dlpack(llm_response.outputs["output_ids"])
        logger.debug(f"Output ids length: {tokens_batch}")
        sequence_length = numpy.from_dlpack(llm_response.outputs["sequence_length"])
        output = self._postprocessing(tokens_batch, sequence_length)
        store_outputs_in_response = set()
        if self._store_outputs_in_response:
            store_outputs_in_response.add("text_output")
        await response_sender.send(
            outputs={"text_output": output},
            final=final,
            store_outputs_in_response=store_outputs_in_response,
        )
