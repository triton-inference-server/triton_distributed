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

import numpy
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, T5Tokenizer


class TritonPythonModel:
    """
    This model allows Triton to act like a api server for T3 ICP
    """

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        inputs = [
            {"name": "query", "data_type": "TYPE_STRING", "dims": [1]},
        ]
        outputs = [
            {"name": "start_ids", "data_type": "TYPE_INT32", "dims": [-1]},
            {"name": "start_lengths", "data_type": "TYPE_INT32", "dims": [-1]},
        ]

        # Store the model configuration as a dictionary.
        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config["input"]:
            input_names.append(input["name"])
        for output in config["output"]:
            output_names.append(output["name"])

        # Add only missing inputs and output to the model configuration.
        for input in inputs:
            if input["name"] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            if output["name"] not in output_names:
                auto_complete_model_config.add_output(output)

        return auto_complete_model_config

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        self.logger = pb_utils.Logger

        tokenizer_dir = model_config["parameters"]["tokenizer_dir"]["string_value"]

        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, legacy=False, padding_side="left", trust_remote_code=True
        )

        if isinstance(self._tokenizer, T5Tokenizer):
            self._tokenizer_bos_id = self._tokenizer.sp_model.bos_id()

        if not self._tokenizer.pad_token:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._tokenizer_end_id = self._tokenizer.encode(
            self._tokenizer.eos_token, add_special_tokens=False
        )[0]
        self._tokenizer_pad_id = self._tokenizer.encode(
            self._tokenizer.pad_token, add_special_tokens=False
        )[0]
        self._vocab_size = self._tokenizer.vocab_size

        for output_name in ["start_ids", "start_lengths"]:
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

        for request in requests:
            query = pb_utils.get_input_tensor_by_name(request, "query").as_numpy()

            # Preprocessing input data.
            if isinstance(self._tokenizer, T5Tokenizer):
                start_ids = [
                    numpy.array(
                        [self._tokenizer_bos_id]
                        + self._tokenizer.encode(
                            s[0].decode(), add_special_tokens=False
                        )
                    ).astype(numpy.int32)
                    for s in query
                ]
            else:
                start_ids = [
                    numpy.array(
                        self._tokenizer.encode(s[0].decode(), add_special_tokens=False)
                    ).astype(numpy.int32)
                    for s in query
                ]

            start_lengths = numpy.array([[len(ids)] for ids in start_ids]).astype(
                numpy.int32
            )

            max_len = 0
            for seq in start_ids:
                max_len = max(max_len, seq.shape[0])
            start_ids = numpy.stack(
                [
                    numpy.pad(
                        seq,
                        (0, max_len - seq.shape[0]),
                        "constant",
                        constant_values=(0, self._tokenizer_pad_id),
                    )
                    for seq in start_ids
                ]
            )

            start_ids_tensor = pb_utils.Tensor(
                "start_ids", numpy.array(start_ids).astype(self.start_ids_dtype)
            )
            start_lengths_tensor = pb_utils.Tensor(
                "start_lengths",
                numpy.array(start_lengths).astype(self.start_lengths_dtype),
            )

            outputs = [start_ids_tensor, start_lengths_tensor]

            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses
