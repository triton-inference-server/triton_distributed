import json
import time

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.decoupled = self.model_config.get("model_transaction_policy", {}).get(
            "decoupled"
        )

    def execute(self, requests):
        if self.decoupled:
            return self.exec_decoupled(requests)
        else:
            return self.exec(requests)

    def exec(self, requests):
        responses = []
        for request in requests:
            params = json.loads(request.parameters())
            rep_count = params["REPETITION"] if "REPETITION" in params else 1

            input_np = pb_utils.get_input_tensor_by_name(
                request, "text_input"
            ).as_numpy()
            stream_np = pb_utils.get_input_tensor_by_name(request, "stream")
            stream = False
            if stream_np:
                stream = stream_np.as_numpy().flatten()[0]
            if stream:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            "STREAM only supported in decoupled mode"
                        )
                    )
                )
            else:
                out_tensor = pb_utils.Tensor(
                    "text_output", np.repeat(input_np, rep_count, axis=1)
                )
                responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses

    def exec_decoupled(self, requests):
        for request in requests:
            params = json.loads(request.parameters())
            rep_count = params["REPETITION"] if "REPETITION" in params else 1
            fail_last = params["FAIL_LAST"] if "FAIL_LAST" in params else False
            delay = params["DELAY"] if "DELAY" in params else None

            sender = request.get_response_sender()
            input_np = pb_utils.get_input_tensor_by_name(
                request, "text_input"
            ).as_numpy()
            stream_np = pb_utils.get_input_tensor_by_name(request, "stream")
            stream = False
            if stream_np:
                stream = stream_np.as_numpy().flatten()[0]
            out_tensor = pb_utils.Tensor("text_output", input_np)
            response = pb_utils.InferenceResponse([out_tensor])
            # If stream enabled, just send multiple copies of response
            # FIXME: Could split up response string into tokens, but this is simpler for now.
            if stream:
                for _ in range(rep_count):
                    if delay is not None:
                        time.sleep(delay)
                    sender.send(response)
                sender.send(
                    None
                    if not fail_last
                    else pb_utils.InferenceResponse(
                        error=pb_utils.TritonError("An Error Occurred")
                    ),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                )
            # If stream disabled, just send one response
            else:
                sender.send(
                    response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
        return None
