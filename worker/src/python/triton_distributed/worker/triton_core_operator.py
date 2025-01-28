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

import asyncio
import json
import logging
import os
import uuid
from typing import Optional

from google.protobuf import json_format, text_format
from tritonclient.grpc import model_config_pb2
from tritonserver import InvalidArgumentError, Server

from triton_distributed.icp.data_plane import DataPlane
from triton_distributed.icp.request_plane import RequestPlane
from triton_distributed.worker.logger import get_logger
from triton_distributed.worker.operator import Operator
from triton_distributed.worker.remote_request import RemoteInferenceRequest
from triton_distributed.worker.remote_response import RemoteInferenceResponse


class TritonCoreOperator(Operator):
    def __init__(
        self,
        name: str,
        version: int,
        triton_core: Server,
        request_plane: RequestPlane,
        data_plane: DataPlane,
        parameters: dict,
        repository: Optional[str] = None,
        logger: logging.Logger = get_logger(__name__),
    ):
        self._repository = repository
        self._name = name
        self._parameters = parameters
        self._triton_core = triton_core
        self._version = version
        self._logger = logger
        self._request_plane = request_plane
        self._data_plane = data_plane
        self._store_outputs_in_response = self._parameters.get(
            "store_outputs_in_response", False
        )

        if not self._repository:
            self._repository = "."

        if repository:
            triton_core.register_model_repository(repository)

        parameter_config = self._parameters.get("config", None)

        model_config = None

        try:
            model_config_path = os.path.join(
                self._repository, self._name, "config.pbtxt"
            )
            with open(model_config_path, "r") as config_file:
                model_config = text_format.Parse(
                    config_file.read(), model_config_pb2.ModelConfig()
                )
        except Exception:
            pass

        if parameter_config and model_config:
            model_config.MergeFrom(
                json_format.Parse(
                    json.dumps(parameter_config), model_config_pb2.ModelConfig()
                )
            )
            model_config = {"config": json_format.MessageToJson(model_config)}
        elif parameter_config:
            model_config = {"config": parameter_config}
        else:
            model_config = None
        self._local_model = self._triton_core.load(self._name, model_config)

    async def execute(self, requests: list[RemoteInferenceRequest]) -> None:
        request_id_map = {}
        response_queue: asyncio.Queue = asyncio.Queue()
        for request in requests:
            self._logger.debug("\n\nReceived request: \n\n%s\n\n", request)
            try:
                local_request = request.to_local_request(self._local_model)
            except Exception as e:
                message = f"Can't resolve tensors for request, ignoring request,{e}"
                self._logger.error(message)
                await request.response_sender().send(
                    error=InvalidArgumentError(message), final=True
                )
                continue

            request_id = str(uuid.uuid1())
            original_id = None
            if local_request.request_id is not None:
                original_id = local_request.request_id
            local_request.request_id = request_id
            request_id_map[request_id] = (request.response_sender(), original_id)

            local_request.response_queue = response_queue
            self._local_model.async_infer(local_request)

        while request_id_map:
            local_response = await response_queue.get()

            remote_response = RemoteInferenceResponse.from_local_response(
                local_response, self._store_outputs_in_response
            )

            response_sender, original_id = request_id_map[local_response.request_id]
            remote_response.request_id = original_id

            if local_response.final:
                del request_id_map[local_response.request_id]
            self._logger.debug("\n\nSending response\n\n%s\n\n", remote_response)
            await response_sender.send(remote_response)
