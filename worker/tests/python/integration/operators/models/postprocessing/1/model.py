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
    def initialize(self, args):
        model_config = json.loads(args["model_config"])

        for output_name in ["OUTPUT"]:
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
        responses = []
        for idx, request in enumerate(requests):
            # Get input tensors
            output_ids = pb_utils.get_input_tensor_by_name(
                request, "OUTPUT_IDS"
            ).as_numpy()

            output_result = np.array(
                self.tokenizer.convert_ids_to_tokens((output_ids.tolist()))
            )
            print(f"Output Result \n\n {output_result}", flush=True)

            output_tensor = pb_utils.Tensor(
                "OUTPUT", output_result.astype(self.output_dtype)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)

        return responses
