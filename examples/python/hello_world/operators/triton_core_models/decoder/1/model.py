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

        auto_complete_model_config.add_input(input_config)
        auto_complete_model_config.add_output(output_config)
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
        if self._model_instance_kind == "GPU" and cupy is None:
            raise RuntimeError("GPU Device set but cupy not installed")

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
                if self._model_instance_kind == "GPU" and cupy is not None:
                    with cupy.cuda.Device(self._model_instance_device_id):
                        input_value = cupy.array(input_value)
                        output_value = cupy.invert(input_value)
                        output_value = output_value[::input_copies]
                        output_tensor = pb_utils.Tensor.from_dlpack(
                            "output", output_value
                        )
                else:
                    output_value = numpy.invert(input_value)
                    output_value = output_value[::input_copies]
                    output_tensor = pb_utils.Tensor("output", output_value)
                output_tensors.append(output_tensor)
                time.sleep(len(output_value) * delay)
            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))
        return responses
