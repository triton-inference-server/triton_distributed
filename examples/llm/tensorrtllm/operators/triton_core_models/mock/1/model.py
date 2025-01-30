# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import threading
import time

import numpy as np
import triton_python_backend_utils as pb_utils

DEFAULT_OUTPUT_LEN = 1000


class TritonPythonModel:
    def initialize(self, args):
        self._logger = pb_utils.Logger

        model_config = json.loads(args["model_config"])

        self._generate_token_latency = (
            float(
                model_config["parameters"]["generate_token_latency_ms"]["string_value"]
            )
        ) / 1000

        self._context_token_latency = (
            float(
                model_config["parameters"]["context_token_latency_ms"]["string_value"]
            )
        ) / 1000

        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config
        )
        if not using_decoupled:
            raise pb_utils.TritonModelException(
                """the model `{}` can generate any number of responses per request,
                enable decoupled transaction policy in model configuration to
                serve this model""".format(
                    args["model_name"]
                )
            )

        for output_name in ["output_ids", "sequence_length", "context_phase_params"]:
            setattr(
                self,
                output_name.lower() + "_dtype",
                pb_utils.triton_string_to_numpy(
                    pb_utils.get_output_config_by_name(model_config, output_name)[
                        "data_type"
                    ]
                ),
            )

        # To keep track of response threads so that we can delay
        # the finalizing the model until all response threads
        # have completed.
        self.inflight_thread_count = 0
        self.inflight_thread_count_lck = threading.Lock()

    def response_thread(self, response_sender, inputs):
        streaming = inputs["streaming"][0]
        request_type = inputs["request_type"]
        output_ids = []
        output_sequence_length = inputs["request_output_len"][0]

        self._logger.log_verbose(
            f"Starting Response Thread: {threading.get_native_id()}"
        )
        self._logger.log_verbose(f"Inputs: {inputs}")
        self._logger.log_verbose(f"Streaming: {streaming}")
        self._logger.log_verbose(f"Request Type: {request_type}")

        input_sequence_length = inputs["input_lengths"][0]

        if inputs["request_type"] != "generate_only":
            for _ in inputs["input_ids"][0]:
                time.sleep(self._context_token_latency)
        if request_type != "context_only":
            for index in range(output_sequence_length):
                output_ids.append(inputs["input_ids"][0][index % input_sequence_length])
                if streaming:
                    output_ids_tensor = pb_utils.Tensor(
                        "output_ids",
                        np.array([[[output_ids[-1]]]]).astype(self.output_ids_dtype),
                    )
                    sequence_length_tensor = pb_utils.Tensor(
                        "sequence_length",
                        np.array([[1]]).astype(self.sequence_length_dtype),
                    )

                    response = pb_utils.InferenceResponse(
                        output_tensors=[output_ids_tensor, sequence_length_tensor]
                    )
                    flags = (
                        pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                        if index == output_sequence_length - 1
                        else 0
                    )
                    response_sender.send(response, flags=flags)

                time.sleep(self._generate_token_latency)
            if not streaming:
                output_ids_tensor = pb_utils.Tensor(
                    "output_ids", np.array([[output_ids]]).astype(self.output_ids_dtype)
                )
                sequence_length_tensor = pb_utils.Tensor(
                    "sequence_length",
                    np.array([[output_sequence_length]]).astype(
                        self.sequence_length_dtype
                    ),
                )

                response = pb_utils.InferenceResponse(
                    output_tensors=[output_ids_tensor, sequence_length_tensor]
                )
                response_sender.send(
                    response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
        if request_type == "context_only":
            output_ids.append(inputs["input_ids"][0][0])
            output_ids_tensor = pb_utils.Tensor(
                "output_ids",
                np.array([[output_ids]]).astype(self.output_ids_dtype),
            )
            sequence_length_tensor = pb_utils.Tensor("sequence_length", np.array([[1]]))

            context_phase_params = pb_utils.Tensor(
                "context_phase_params",
                np.array([[1, 2, 3, 4]]).astype(self.context_phase_params_dtype),
            )

            response = pb_utils.InferenceResponse(
                output_tensors=[
                    output_ids_tensor,
                    sequence_length_tensor,
                    context_phase_params,
                ]
            )
            response_sender.send(
                response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
            )

        # We must close the response sender to indicate to Triton that we are
        # done sending responses for the corresponding request. We can't use the
        # response sender after closing it. The response sender is closed by
        # setting the TRITONSERVER_RESPONSE_COMPLETE_FINAL.
        #           response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        with self.inflight_thread_count_lck:
            self.inflight_thread_count -= 1
            self._logger.log_verbose(
                f"Exiting Response Thread: {threading.get_native_id()}"
            )

    def _get_inputs(self, request):
        inputs = [
            "context_phase_params",
            "streaming",
            "min_length",
            "request_output_len",
            "input_lengths",
            "input_ids",
        ]
        result = {}
        for input_ in inputs:
            value = pb_utils.get_input_tensor_by_name(request, input_)
            if value is not None:
                result[input_] = value.as_numpy()
        input_parameters = json.loads(request.parameters())
        if "request_type" in input_parameters:
            result["request_type"] = input_parameters["request_type"]
        else:
            result["request_type"] = "aggregate"
        if "request_output_len" not in inputs:
            result["request_output_len"] = DEFAULT_OUTPUT_LEN
        if "streaming" not in result:
            result["streaming"] = [False]
        return result

    def execute(self, requests):
        for idx, request in enumerate(requests):
            inputs = self._get_inputs(request)

            # Start a separate thread to send the responses for the request. The
            # sending back the responses is delegated to this thread.
            thread = threading.Thread(
                target=self.response_thread,
                args=(request.get_response_sender(), inputs),
            )

            # A model using decoupled transaction policy is not required to send all
            # responses for the current request before returning from the execute.
            # To demonstrate the flexibility of the decoupled API, we are running
            # response thread entirely independent of the execute thread.
            thread.daemon = True

            with self.inflight_thread_count_lck:
                self.inflight_thread_count += 1
            thread.start()

        return None
