import json

import numpy as np
import triton_python_backend_utils as pb_utils

try:
    import cupy
except Exception:
    cupy = None


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        inputs = []
        outputs = []
        dims = [-1, -1]
        optional = True

        for data_type in ["type_int64"]:
            type_name = data_type.split("_")[1].lower()
            input_name = f"{type_name}_input"

            output_name = "fp64_output_partial"
            inputs.append(
                {
                    "name": input_name,
                    "data_type": data_type,
                    "dims": dims,
                    "optional": optional,
                }
            )
            outputs.append({"name": output_name, "data_type": data_type, "dims": dims})

            input_name = f"{type_name}_input_divisor"

            inputs.append(
                {
                    "name": input_name,
                    "data_type": data_type,
                    "dims": dims,
                    "optional": optional,
                }
            )

        outputs.append(
            {"name": "output_parameters", "data_type": "TYPE_STRING", "dims": [1]}
        )
        for input_ in inputs:
            auto_complete_model_config.add_input(input_)
        for output in outputs:
            auto_complete_model_config.add_output(output)

        auto_complete_model_config.set_max_batch_size(0)

        return auto_complete_model_config

    def initialize(self, args):
        self._model_config = json.loads(args["model_config"])
        self._request_gpu_memory = False
        if "parameters" in self._model_config:
            parameters = self._model_config["parameters"]
            if (
                "request_gpu_memory" in parameters
                and parameters["request_gpu_memory"]["string_value"] == "True"
            ):
                self._request_gpu_memory = True

    def execute(self, requests):
        responses = []
        for request in requests:
            output_tensors = []

            divisor = pb_utils.get_input_tensor_by_name(request, "int64_input_divisor")
            divisor = divisor.as_numpy()[0][0]

            dividends = pb_utils.get_input_tensor_by_name(request, "int64_input")

            dividends = dividends.as_numpy()

            output_value = np.array([np.divide(dividends, divisor)])

            if self._request_gpu_memory:
                output_value = cupy.array(output_value)

                output_tensor = pb_utils.Tensor.from_dlpack(
                    "fp64_output_partial", output_value
                )
            else:
                output_tensor = pb_utils.Tensor("fp64_output_partial", output_value)

            output_tensors.append(output_tensor)

            output_parameters = np.array([request.parameters()]).astype(np.object_)
            output_tensors.append(
                pb_utils.Tensor("output_parameters", output_parameters)
            )

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=output_tensors,
                )
            )
        return responses
