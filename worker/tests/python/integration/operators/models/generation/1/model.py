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
import threading
import time

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])

        self._output_token_latency = (
            int(model_config["parameters"]["inter_token_latency_ms"]["string_value"])
        ) / 1000

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config
        )
        if not using_decoupled:
            raise pb_utils.TritonModelException(
                """the model `{}` can generate any number of responses per request,
                enable decoupled transaction policy in model configuration to
                serve this model""".format(
                    args["model_name"]
                )
            )

        for output_name in ["OUTPUT_IDS", "SEQUENCE_LENGTH"]:
            setattr(
                self,
                output_name.lower() + "_dtype",
                pb_utils.triton_string_to_numpy(
                    pb_utils.get_output_config_by_name(model_config, output_name)[
                        "data_type"
                    ]
                ),
            )

        # To keep track of response threads so that we can delay
        # the finalizing the model until all response threads
        # have completed.
        self.inflight_thread_count = 0
        self.inflight_thread_count_lck = threading.Lock()

    def response_thread(self, response_sender, kv_cache, request_output_len):
        for idx in range(request_output_len):
            time.sleep(self._output_token_latency)
            output_ids_tensor = pb_utils.Tensor(
                "OUTPUT_IDS", kv_cache.astype(self.output_ids_dtype)
            )

            sequence_length = np.array([kv_cache.size])
            sequence_length_tensor = pb_utils.Tensor(
                "SEQUENCE_LENGTH", sequence_length.astype(self.sequence_length_dtype)
            )
            response = pb_utils.InferenceResponse(
                output_tensors=[output_ids_tensor, sequence_length_tensor]
            )
            response_sender.send(response)

        # We must close the response sender to indicate to Triton that we are
        # done sending responses for the corresponding request. We can't use the
        # response sender after closing it. The response sender is closed by
        # setting the TRITONSERVER_RESPONSE_COMPLETE_FINAL.
        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        with self.inflight_thread_count_lck:
            self.inflight_thread_count -= 1

    def execute(self, requests):
        for idx, request in enumerate(requests):
            # Get input tensors
            kv_cache = pb_utils.get_input_tensor_by_name(request, "KV_CACHE").as_numpy()
            request_output_len = pb_utils.get_input_tensor_by_name(
                request, "REQUEST_OUTPUT_LEN"
            ).as_numpy()

            # Start a separate thread to send the responses for the request. The
            # sending back the responses is delegated to this thread.
            thread = threading.Thread(
                target=self.response_thread,
                args=(
                    requests[0].get_response_sender(),
                    kv_cache,
                    request_output_len[0],
                ),
            )

            # A model using decoupled transaction policy is not required to send all
            # responses for the current request before returning from the execute.
            # To demonstrate the flexibility of the decoupled API, we are running
            # response thread entirely independent of the execute thread.
            thread.daemon = True

            with self.inflight_thread_count_lck:
                self.inflight_thread_count += 1

            thread.start()

        return None
