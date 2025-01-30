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
from transformers import AutoTokenizer


class TritonPythonModel:
    """
    This model allows Triton to act like a api server for T3 ICP
    """

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        inputs = [
            {"name": "tokens_batch", "data_type": "TYPE_INT32", "dims": [-1, -1]},
            {"name": "sequence_lengths", "data_type": "TYPE_INT32", "dims": [-1]},
        ]
        outputs = [
            {"name": "output", "data_type": "TYPE_STRING", "dims": [-1]},
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

        # Parse model configs
        model_config = json.loads(args["model_config"])
        tokenizer_dir = model_config["parameters"]["tokenizer_dir"]["string_value"]

        skip_special_tokens = model_config["parameters"].get("skip_special_tokens")
        if skip_special_tokens is not None:
            skip_special_tokens_str = skip_special_tokens["string_value"].lower()
            if skip_special_tokens_str in [
                "true",
                "false",
                "1",
                "0",
                "t",
                "f",
                "y",
                "n",
                "yes",
                "no",
            ]:
                self.skip_special_tokens = skip_special_tokens_str in [
                    "true",
                    "1",
                    "t",
                    "y",
                    "yes",
                ]
            else:
                self.logger.log_warn(
                    f"[TensorRT-LLM][WARNING] Don't setup 'skip_special_tokens' correctly (set value is {skip_special_tokens['string_value']}). Set it as True by default."
                )
                self.skip_special_tokens = True
        else:
            self.logger.log_warn(
                "[TensorRT-LLM][WARNING] Don't setup 'skip_special_tokens'. Set it as True by default."
            )
            self.skip_special_tokens = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, legacy=False, padding_side="left", trust_remote_code=True
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for output_name in ["output"]:
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
        tokens_batch = []
        sequence_lengths = []
        for idx, request in enumerate(requests):
            for input_tensor in request.inputs():
                if input_tensor.name() == "tokens_batch":
                    tokens_batch.append(input_tensor.as_numpy())
                elif input_tensor.name() == "sequence_lengths":
                    sequence_lengths.append(input_tensor.as_numpy())
                else:
                    raise ValueError(f"unknown input {input_tensor.name}")

        # batch decode
        list_of_tokens = []
        req_idx_offset = 0
        req_idx_offsets = [req_idx_offset]
        for idx, token_batch in enumerate(tokens_batch):
            for batch_idx, beam_tokens in enumerate(token_batch):
                for beam_idx, tokens in enumerate(beam_tokens):
                    seq_len = sequence_lengths[idx][batch_idx][beam_idx]
                    list_of_tokens.append(tokens[:seq_len])
                    req_idx_offset += 1

            req_idx_offsets.append(req_idx_offset)

        all_outputs = self.tokenizer.batch_decode(
            list_of_tokens, skip_special_tokens=self.skip_special_tokens
        )

        # construct responses
        responses = []
        for idx, request in enumerate(requests):
            req_outputs = [
                x.encode("utf8")
                for x in all_outputs[req_idx_offsets[idx] : req_idx_offsets[idx + 1]]
            ]

            output_tensor = pb_utils.Tensor(
                "output", np.array(req_outputs).astype(self.output_dtype)
            )

            outputs = [output_tensor]

            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
