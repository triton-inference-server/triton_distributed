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

import numpy
import triton_python_backend_utils as pb_utils

try:
    import cupy
except Exception:
    cupy = None


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        """Auto Complets Model Config

        Model has one input and one output
        both of type int64

        Parameters
        ----------
        auto_complete_model_config : config
            Enables reading and updating config.pbtxt


        """

        input_config = {
            "name": "input",
            "data_type": "TYPE_INT64",
            "dims": [-1],
            "optional": False,
        }

        output_config = {
            "name": "output",
            "data_type": "TYPE_INT64",
            "dims": [-1],
        }

        copies_config = {
            "name": "input_copies",
            "data_type": "TYPE_INT64",
            "dims": [1],
        }

        auto_complete_model_config.add_input(input_config)
        auto_complete_model_config.add_output(output_config)
        auto_complete_model_config.add_output(copies_config)
        auto_complete_model_config.set_max_batch_size(0)
        auto_complete_model_config.set_model_transaction_policy({"decoupled": False})

        return auto_complete_model_config

    def initialize(self, args):
        self._model_config = json.loads(args["model_config"])
        self._model_instance_kind = args["model_instance_kind"]
        self._model_instance_device_id = int(args["model_instance_device_id"])
        self._config_parameters = self._model_config.get("parameters", {})
        self._input_copies = int(
            self._config_parameters.get("input_copies", {"string_value": "5"})[
                "string_value"
            ]
        )
        self._delay = float(
            self._config_parameters.get("delay", {"string_value": "0"})["string_value"]
        )

    def execute(self, requests):
        responses = []
        input_copies = self._input_copies
        delay = self._delay
        for request in requests:
            output_tensors = []
            parameters = json.loads(request.parameters())
            if parameters:
                input_copies = int(parameters.get("input_copies", self._input_copies))
                delay = float(parameters.get("delay", self._delay))
            for input_tensor in request.inputs():
                input_value = input_tensor.as_numpy()
                output_value = []
                if self._model_instance_kind == "GPU":
                    with cupy.cuda.Device(self._model_instance_device_id):
                        input_value = cupy.array(input_value)
                        output_value = cupy.tile(input_value, input_copies)
                        output_value = cupy.invert(output_value)
                        output_tensor = pb_utils.Tensor.from_dlpack(
                            "output", output_value
                        )
                else:
                    output_value = numpy.tile(input_value, input_copies)
                    output_value = numpy.invert(output_value)
                    output_tensor = pb_utils.Tensor("output", output_value)
                output_tensors.append(output_tensor)
                output_tensors.append(
                    pb_utils.Tensor(
                        "input_copies", numpy.array(input_copies).astype("int64")
                    )
                )
                time.sleep(len(output_value) * delay)

            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))
        return responses
