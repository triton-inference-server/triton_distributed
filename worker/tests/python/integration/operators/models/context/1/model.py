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
import time

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        self._context_delay = (
            int(model_config["parameters"]["context_delay_ms"]["string_value"])
        ) / 1000

        for output_name in [
            "KV_CACHE",
            "OUTPUT_IDS",
            "SEQUENCE_LENGTH",
            "REQUEST_OUTPUT_LEN",
        ]:
            setattr(
                self,
                output_name.lower() + "_dtype",
                pb_utils.triton_string_to_numpy(
                    pb_utils.get_output_config_by_name(model_config, output_name)[
                        "data_type"
                    ]
                ),
            )

    def execute(self, requests):
        responses = []
        for idx, request in enumerate(requests):
            # Get input tensors
            input_ids = pb_utils.get_input_tensor_by_name(
                request, "INPUT_IDS"
            ).as_numpy()
            input_lengths = pb_utils.get_input_tensor_by_name(
                request, "INPUT_LENGTH"
            ).as_numpy()
            request_output_len = pb_utils.get_input_tensor_by_name(
                request, "REQUEST_OUTPUT_LEN"
            ).as_numpy()

            time.sleep(self._context_delay)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            kv_cache_tensor = pb_utils.Tensor(
                "KV_CACHE", input_ids.astype(self.kv_cache_dtype)
            )

            output_ids_tensor = pb_utils.Tensor(
                "OUTPUT_IDS", input_ids.astype(self.output_ids_dtype)
            )
            sequence_length_tensor = pb_utils.Tensor(
                "SEQUENCE_LENGTH", input_lengths.astype(self.sequence_length_dtype)
            )
            request_output_len_tensor = pb_utils.Tensor(
                "REQUEST_OUTPUT_LEN", request_output_len
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    kv_cache_tensor,
                    output_ids_tensor,
                    sequence_length_tensor,
                    request_output_len_tensor,
                ]
            )
            responses.append(inference_response)

        return responses
