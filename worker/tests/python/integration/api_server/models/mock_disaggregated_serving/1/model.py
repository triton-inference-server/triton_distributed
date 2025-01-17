# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import gc
import json
import queue
import threading
import traceback
import uuid

import triton_python_backend_utils as pb_utils
import ucp
from triton_distributed.icp.nats_request_plane import NatsRequestPlane
from triton_distributed.icp.ucp_data_plane import UcpDataPlane
from triton_distributed.worker.remote_operator import RemoteOperator


class TritonPythonModel:
    """
    This model allows Triton to act like a api server for T3 ICP
    """

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        inputs = [
            {"name": "query", "data_type": "TYPE_STRING", "dims": [1]},
            {
                "name": "request_output_len",
                "data_type": "TYPE_INT32",
                "dims": [1],
            },
        ]
        outputs = [{"name": "output", "data_type": "TYPE_STRING", "dims": [-1]}]

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

        # We need to use decoupled transaction policy for saturating T3
        auto_complete_model_config.set_model_transaction_policy(dict(decoupled=True))

        # Disabling batching in Triton,
        auto_complete_model_config.set_max_batch_size(0)

        return auto_complete_model_config

    async def _connect(self):
        ucp.reset()
        self._request_plane = NatsRequestPlane(self._request_plane_uri)
        self._data_plane = UcpDataPlane()
        self._data_plane.connect()
        await self._request_plane.connect()

    async def _disconnect(self, timeout):
        self._data_plane.close(wait_for_release=timeout)
        await self._request_plane.close()

    async def _await_shutdown(self):
        """
        Primary coroutine running on the engine event loop. This coroutine is responsible for
        keeping the engine alive until a shutdown is requested.
        """
        # first await the shutdown signal
        while self._shutdown_event.is_set() is False:
            await asyncio.sleep(5)
        # Wait for the ongoing_requests
        while self._ongoing_request_count > 0:
            self.logger.log_info(
                "[API Server] Awaiting remaining {} requests".format(
                    self._ongoing_request_count
                )
            )
            await asyncio.sleep(5)
        for task in asyncio.all_tasks(loop=self._loop):
            if task is not asyncio.current_task():
                task.cancel()
        self.logger.log_info("[API Server] Shutdown complete")

    def _create_task(self, coro):
        """
        Creates a task on the event loop which is running on a separate thread.
        """
        assert (
            self._shutdown_event.is_set() is False
        ), "Cannot create tasks after shutdown has been requested"
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _event_loop(self, loop):
        """
        Runs the engine's event loop on a separate thread.
        """
        asyncio.set_event_loop(loop)
        self._loop.run_until_complete(self._await_shutdown())

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        self.logger = pb_utils.Logger

        # Starting asyncio event loop to process the received requests asynchronously.
        self._loop = asyncio.get_event_loop()
        self._event_thread = threading.Thread(
            target=self._event_loop, args=(self._loop,)
        )
        self._shutdown_event = asyncio.Event()
        self._event_thread.start()

        self._request_plane_uri = model_config["parameters"]["request_plane_uri"][
            "string_value"
        ]

        future = self._create_task(self._connect())
        try:
            _ = future.result(timeout=5)
        except TimeoutError:
            self.logger.log_error(
                "The connection to T3 ICP took too long, cancelling the task..."
            )
            future.cancel()
        except Exception as exc:
            self.logger.log_error(
                f"The connection to T3 ICP raised an exception: {exc!r}"
            )

        self._remote_worker_name = model_config["parameters"]["remote_worker_name"][
            "string_value"
        ]
        self._remote_operator = RemoteOperator(
            self._remote_worker_name, self._request_plane, self._data_plane
        )

        # Starting the response thread. It allows API Server to keep making progress while
        # response sender(s) are sending responses to server frontend.
        self._response_queue = queue.Queue()
        self._response_thread = threading.Thread(target=self.response_loop)
        self._response_thread.start()

        # Counter to keep track of ongoing request counts
        self._ongoing_request_count = 0

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

    def response_loop(self):
        while True:
            item = self._response_queue.get()
            # To signal shutdown a None item will be added to the queue.
            if item is None:
                break
            response_sender, response, response_flag = item
            del item
            try:
                response_sender.send(response, response_flag)
            except Exception as e:
                self.logger.log_error(
                    f"An error occurred while sending a response: {e}"
                )
            finally:
                if response_flag == pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL:
                    self._ongoing_request_count -= 1
                    del response_sender
                    if self._ongoing_request_count == 0:
                        gc.collect()

    def execute(self, requests):
        for request in requests:
            if request is not None:
                self._create_task(self.remote_execute(request))
        return None

    async def remote_execute(self, request):
        response_sender = request.get_response_sender()
        self._ongoing_request_count += 1

        query = pb_utils.get_input_tensor_by_name(request, "query").as_numpy()
        request_output_len = pb_utils.get_input_tensor_by_name(
            request, "request_output_len"
        ).as_numpy()

        request_id = str(uuid.uuid4())
        infer_request = self._remote_operator.create_request(
            inputs={"query": query, "request_output_len": request_output_len},
            request_id=request_id,
        )

        try:
            async for response in await self._remote_operator.async_infer(
                inference_request=infer_request
            ):
                if response.error:
                    raise pb_utils.TritonModelException(response.error.message())
                if not response.final:
                    output = response.outputs["output"]
                    output_value = output.to_bytes_array()
                    # Just forwarding query to the pre-processed input_ids
                    output_tensor = pb_utils.Tensor(
                        "output", output_value.astype(self.output_dtype)
                    )

                    inference_response = pb_utils.InferenceResponse(
                        output_tensors=[output_tensor]
                    )
                    self._response_queue.put_nowait(
                        (response_sender, inference_response, 0)
                    )
        except Exception as e:
            self.logger.log_error(
                f"Failed running remote inference {traceback.print_exc()}"
            )
            raise pb_utils.TritonModelException(repr(e))

        self._response_queue.put_nowait(
            (response_sender, None, pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        )

        return None

    def finalize(self):
        self.logger.log_info("[API Server] Issuing finalize to API Server")
        future = self._create_task(self._disconnect(timeout=5))
        try:
            _ = future.result(timeout=7)
        except TimeoutError:
            self.logger.log_error(
                "The connection to T3 ICP took too long, cancelling the task..."
            )
            future.cancel()
        except Exception as exc:
            self.logger.log_error(
                f"The connection to T3 ICP raised an exception: {exc!r}"
            )

        self._shutdown_event.set()
        # Shutdown the event thread.
        if self._event_thread is not None:
            self._event_thread.join()
            self._event_thread = None

        # Shutdown the response thread.
        self._response_queue.put(None)
        if self._response_thread is not None:
            self._response_thread.join()
            self._response_thread = None
