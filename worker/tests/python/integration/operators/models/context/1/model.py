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
