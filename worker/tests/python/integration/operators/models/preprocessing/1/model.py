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

import numpy as np
import triton_python_backend_utils as pb_utils

# from transformers import LlamaTokenizer
# llama_tokenizer = LlamaTokenizer.from_pretrained("/path/to/hfmodel")
from transformers import XLNetTokenizer


class TritonPythonModel:
    """
    This is a mock disaggregated serving pre-processing model.
    """

    def initialize(self, args):
        model_config = json.loads(args["model_config"])

        for output_name in ["INPUT_IDS", "INPUT_LENGTH", "REQUEST_OUTPUT_LEN"]:
            setattr(
                self,
                output_name.lower() + "_dtype",
                pb_utils.triton_string_to_numpy(
                    pb_utils.get_output_config_by_name(model_config, output_name)[
                        "data_type"
                    ]
                ),
            )

        # Using a mock hard coded auto-tokenizer
        self.tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

    def execute(self, requests):
        print("In preprocessing execute!", flush=True)
        responses = []

        for idx, request in enumerate(requests):
            # Get input tensors
            query = pb_utils.get_input_tensor_by_name(request, "query").as_numpy()
            request_output_len = pb_utils.get_input_tensor_by_name(
                request, "request_output_len"
            ).as_numpy()

            print(f"query(pre-proc) {query}", flush=True)
            tokenize = np.array(self.tokenizer.encode(query[0].decode()))
            print(f"tokenize(pre-proc) {tokenize.size}", flush=True)
            input_length = np.array([tokenize.size])

            # Just forwarding query to the pre-processed input_ids
            input_id_tensor = pb_utils.Tensor(
                "INPUT_IDS", tokenize.astype(self.input_ids_dtype)
            )
            # Just forwarding query to the pre-processed input_ids
            input_length_tensor = pb_utils.Tensor(
                "INPUT_LENGTH", input_length.astype(self.input_length_dtype)
            )
            request_output_len_tensor = pb_utils.Tensor(
                "REQUEST_OUTPUT_LEN", request_output_len
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    input_id_tensor,
                    input_length_tensor,
                    request_output_len_tensor,
                ]
            )
            responses.append(inference_response)

        return responses
