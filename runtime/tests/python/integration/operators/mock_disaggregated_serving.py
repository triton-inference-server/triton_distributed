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


from tritonserver import TritonError

from triton_distributed.runtime.operator import Operator
from triton_distributed.runtime.remote_operator import RemoteOperator
from triton_distributed.runtime.remote_request import RemoteInferenceRequest


class MockDisaggregatedServing(Operator):
    def __init__(
        self,
        name,
        version,
        request_plane,
        data_plane,
        params,
        repository,
        logger,
        triton_core,
    ):
        self._triton_core = triton_core
        self._request_plane = request_plane
        self._data_plane = data_plane
        self._params = params
        self._preprocessing_model = RemoteOperator(
            "preprocessing", self._request_plane, self._data_plane
        )
        self._context_model = RemoteOperator(
            "context", self._request_plane, self._data_plane
        )
        self._generate_model = RemoteOperator(
            "generation", self._request_plane, self._data_plane
        )
        self._postprocessing_model = RemoteOperator(
            "postprocessing", self._request_plane, self._data_plane
        )
        self._logger = logger

    async def _run_generate(self, context_response, response_sender):
        error = None
        generate_inputs = {}
        if not error:
            for output_name in ["KV_CACHE", "REQUEST_OUTPUT_LEN"]:
                if output_name not in context_response.outputs.keys():
                    error_msg = f"Expected '{output_name}' as output in llm model response, Got outputs {context_response.outputs.keys()}"
                    self._logger.error(error_msg)
                    self._logger.debug(f"context_response: {context_response}")
                    error = TritonError(error_msg)
                else:
                    generate_inputs[output_name] = context_response.outputs[output_name]

        postproc_result = []
        generate_responses = []
        if not error:
            try:
                # TODO: Run post-processing in parallel with generate
                async for response in await self._generate_model.async_infer(
                    inputs=generate_inputs
                ):
                    generate_responses.append(response)
                    self._logger.debug(f"Received response {response}")
                    if not generate_responses[-1].final:
                        postproc_result.append(
                            await self._run_postprocessing(
                                generate_responses[-1], response_sender, final=False
                            )
                        )
            except Exception as e:
                error = TritonError(repr(e))
                self._logger.exception("Failed to run post-processing")

        for generate_response in generate_responses:
            for tensor in generate_response.outputs.values():
                del tensor

        return postproc_result

    async def _run_postprocessing(self, llm_response, response_sender, final):
        self._logger.debug(f"going to run_post_processing final={final}")
        postproc_inputs = {}
        for output_name in ["OUTPUT_IDS", "SEQUENCE_LENGTH"]:
            if output_name not in llm_response.outputs.keys():
                error_msg = f"Expected '{output_name}' as output in llm model response, Got outputs {llm_response.outputs.items()}"
                self._logger.error(error_msg)
                self._logger.debug(f"llm_response: {llm_response}")
                raise Exception(error_msg)
            else:
                postproc_inputs[output_name] = llm_response.outputs[output_name]

        outputs = {}
        postproc_responses = []

        # TODO: Run post-processing in parallel with generate
        self._logger.debug(f"Sending request to post-process {postproc_inputs}")
        sending = []
        async for response in await self._postprocessing_model.async_infer(
            inputs=postproc_inputs
        ):
            self._logger.debug(f"Received response {response}")
            self._logger.debug(f"Got response from post-process {response}")
            postproc_responses.append(response)
            outputs["output"] = postproc_responses[-1].outputs["OUTPUT"]
            sending.append(await response_sender().send(outputs=outputs, final=False))
        return sending

    async def execute(self, requests: list[RemoteInferenceRequest]):
        self._logger.debug("in execute!")
        error = None
        for request in requests:
            """
            Pre-processing
            """
            preproc_responses = []
            async for response in await self._preprocessing_model.async_infer(
                inference_request=request
            ):
                preproc_responses.append(response)

            if not error and len(preproc_responses) != 1:
                error_msg = f"Expected exactly 1 response from preprocessing model, Got {len(preproc_responses)}"
                self._logger.error(error_msg)
                error = TritonError(error_msg)

            context_inputs = {}
            if not error:
                for output_name in ["INPUT_IDS", "INPUT_LENGTH", "REQUEST_OUTPUT_LEN"]:
                    if output_name not in preproc_responses[0].outputs.keys():
                        error_msg = f"Expected '{output_name}' as output in preprocessing model response, Got outputs {response.outputs.keys()}"
                        self._logger.error(error_msg)
                        error = TritonError(error_msg)
                    else:
                        context_inputs[output_name] = preproc_responses[0].outputs[
                            output_name
                        ]

            """
            Prefill
            """
            context_responses = []
            postproc_result = []
            if not error:
                async for response in await self._context_model.async_infer(
                    inputs=context_inputs
                ):
                    context_responses.append(response)

            if not error:
                if not error and len(context_responses) != 1:
                    error_msg = f"Expected exactly 1 response from context model, Got {len(context_responses)}"
                    self._logger.error(error_msg)
                    error = TritonError(error_msg)
                else:
                    postproc_result.append(
                        self._run_postprocessing(
                            context_responses[0], request.response_sender, final=False
                        )
                    )

            """
            Generate
            """
            if not error:
                postproc_result.append(
                    self._run_generate(context_responses[0], request.response_sender)
                )
                for result in postproc_result:
                    try:
                        await result
                    except Exception as e:
                        self._logger.exception(
                            f"Failed getting response from post-processing {result}: {e}"
                        )
                        error = TritonError(repr(e))

            for tensor in preproc_responses[0].outputs.values():
                del tensor
            for tensor in context_responses[0].outputs.values():
                del tensor

            if error:
                await request.response_sender().send(error=error, final=True)
            else:
                await request.response_sender().send(final=True)
