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
import json

import numpy

from triton_distributed.worker import (
    RemoteInferenceRequest,
    RemoteOperator,
    TritonCoreOperator,
)

from nova_distributed import KvRouter, nova_worker, DistributedRuntime

class KvAwareRoutingOperator(TritonCoreOperator):
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
        self._router = None
        self._generate = RemoteOperator("generate", request_plane, data_plane)

        self._repository = repository
        self._triton_core = triton_core
        self._triton_core.register_model_repository(repository)
        self._preprocess_model = self._triton_core.load("simple_preprocessing")
        self._postprocess_model = self._triton_core.load("simple_postprocessing")
        self._logger = logger
        self._store_outputs_in_response = True
        print("KvAwareRoutingOperator initialized")

    async def execute(self, requests: list[RemoteInferenceRequest]):
        print("KvAwareRoutingOperator execute")
        if self._router is None:
            # inflight initialization because router setup requires event loop
            # which is only obtainable from async (coroutine).
            # This is okay because the router will be hit first and there
            # shouldn't be interested events before first call to this function. 
            loop = asyncio.get_running_loop()
            runtime = DistributedRuntime(loop)
            backend = runtime.namespace("router").component("generate")
            self._router = KvRouter(runtime, backend)
        print("Initialized router")
        self._logger.debug("Executing KvAwareRouting Request")
        background_tasks = []
        for request in requests:
            task = asyncio.create_task(self._execute_request(request))
            background_tasks.append(task)

        try:
            results = await asyncio.gather(*background_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    self._logger.exception(
                        f"Running request execution failed: {result}"
                    )
                else:
                    self._logger.debug(
                        f"Request execution completed with result: {result}"
                    )
        except Exception as e:
            self._logger.exception(f"Error during request execution: {e}")

    async def _execute_request(self, request: RemoteInferenceRequest):
        background_tasks = []
        sampling_params = {}

        response_sender = request.response_sender()

        """Preprocessing"""
        self._logger.debug(request)
        if "text_input" in request.inputs:
            query = request.inputs["text_input"].to_bytes_array()
        elif "prompt" in request.inputs:
            query = request.inputs["prompt"].to_bytes_array()
        elif "prompt" in request.parameters:
            query = request.parameters["prompt"]
        else:
            await response_sender.send(error=f"invalid request {request}", final=True)
            return

        if "sampling_params" in request.parameters:
            sampling_params = json.loads(
                request.parameters["sampling_params"].removeprefix("JSON:")
            )

        if "max_tokens" in request.inputs:
            request_output_len = request.inputs["max_tokens"]
        elif "max_tokens" in sampling_params:
            request_output_len = numpy.array(
                [[sampling_params["max_tokens"]]], dtype=numpy.int32
            )

        input_ids, input_lengths = await self._preprocess(query)
        self._logger.debug(input_ids, input_lengths)
        print(input_ids)

        # [FIXME] not rate limiting due to metric polling is not supported
        # KV aware routing
        lora_id = 0
        try:
            self._generate.component_id = await self._router.schedule(input_ids[0], lora_id)
        except Exception as e:
            print(str(e))
            if "No worker found" in str(e):
                self._generate.component_id = None
            else:
                self._logger.exception(f"Error during selecting worker: {e}")

        # [TODO] add disaggregated example
        """llm"""
        llm_inputs = {}
        llm_inputs["input_ids"] = input_ids
        llm_inputs["input_lengths"] = input_lengths
        llm_inputs["request_output_len"] = request_output_len

        async for llm_response in await self._generate.async_infer(
            inputs=llm_inputs,
        ):
            self._logger.debug(f"llm response completed: {llm_response}")
            background_tasks.append(
                asyncio.create_task(
                    self._send_llm_response(
                        llm_response,
                        response_sender,
                        final=llm_response.final,
                    )
                )
            )

        try:
            results = await asyncio.gather(*background_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    self._logger.exception(
                        f"Sending response failed with exception: {result}"
                    )
                else:
                    self._logger.debug(f"Response sent successfully: {result}")
        except Exception as e:
            self._logger.exception(f"Error during response sending: {e}")

        for output in llm_response.outputs:
            del output

    async def _preprocess(self, query):
        start_ids = None
        start_lengths = None
        if isinstance(query, str):
            query = [[query]]
        async for preprocess_response in self._preprocess_model.async_infer(
            inputs={"query": query}
        ):
            self._logger.debug(f"Preprocess response completed: {preprocess_response}")
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
            self._logger.debug(f"Received postprocess response: {postprocess_response}")
            output = postprocess_response.outputs["output"].to_string_array()
            outputs.append(output)

        return outputs

    async def _send_llm_response(self, llm_response, response_sender, final):
        tokens_batch = numpy.from_dlpack(llm_response.outputs["output_ids"])
        self._logger.debug(f"Output ids length: {tokens_batch}")
        sequence_length = numpy.from_dlpack(llm_response.outputs["sequence_length"])
        output = await self._postprocessing(tokens_batch, sequence_length)
        store_outputs_in_response = set()
        if self._store_outputs_in_response:
            store_outputs_in_response.add("text_output")
        await response_sender.send(
            outputs={"text_output": output[0]},
            final=final,
            store_outputs_in_response=store_outputs_in_response,
        )
