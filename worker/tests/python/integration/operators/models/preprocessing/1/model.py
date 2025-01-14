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
