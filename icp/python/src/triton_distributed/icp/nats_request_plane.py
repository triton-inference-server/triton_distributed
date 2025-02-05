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

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Dict, Optional
from urllib.parse import urlsplit, urlunsplit

import nats

from triton_distributed.icp.protos.icp_pb2 import ModelInferRequest, ModelInferResponse
from triton_distributed.icp.request_plane import (
    RequestPlane,
    get_icp_final_response,
    get_icp_request_id,
    get_icp_response_error,
    get_icp_response_to_uri,
    set_icp_component_id,
    set_icp_request_id,
    set_icp_request_to_uri,
    set_icp_response_to_uri,
)

DEFAULT_REQUESTS_PORT = int(os.getenv("DEFAULT_REQUESTS_PORT", 4222))
DEFAULT_REQUESTS_HOST = os.getenv("DEFAULT_REQUESTS_HOST", "localhost")
DEFAULT_REQUESTS_URI = os.getenv(
    "DEFAULT_REQUESTS_URI", f"nats://{DEFAULT_REQUESTS_HOST}:{DEFAULT_REQUESTS_PORT}"
)


class AsyncModelInferRequestIterator:
    def __init__(self, requests: list[ModelInferRequest]) -> None:
        self._requests = requests

    def __aiter__(self) -> AsyncModelInferRequestIterator:
        return self

    async def __anext__(self):
        if not self._requests:
            raise StopAsyncIteration
        return self._requests.pop(0)


class AsyncModelInferResponseIterator:
    def __init__(
        self,
        queue: Optional[asyncio.Queue],
        raise_on_error=False,
    ) -> None:
        self._queue = queue
        self._complete = False
        self._raise_on_error = raise_on_error
        if not self._queue:
            self._complete = True

    def __aiter__(self) -> AsyncModelInferResponseIterator:
        return self

    async def __anext__(self):
        if self._complete or self._queue is None:
            raise StopAsyncIteration
        response = await self._queue.get()
        self._complete = get_icp_final_response(response)
        error = get_icp_response_error(response)
        if error is not None and self._raise_on_error:
            raise error
        return response

    def cancel(self) -> None:
        raise NotImplementedError()


def is_port_in_use(port: int, host: str = DEFAULT_REQUESTS_HOST) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


class NatsServer:
    def __init__(
        self,
        port: int = DEFAULT_REQUESTS_PORT,
        host: str = DEFAULT_REQUESTS_HOST,
        store_dir: str = "/tmp/nats_store",
        log_dir: str = "logs",
        debug: bool = False,
        clear_store: bool = True,
        dry_run: bool = False,
    ) -> None:
        self._process = None
        self.port = port
        self.url = f"{host}:{port}"
        command = [
            "/usr/local/bin/nats-server",
            "--jetstream",
            "--addr",
            str(host),
            "--port",
            str(port),
            "--store_dir",
            store_dir,
        ]

        if debug:
            command.extend(["--debug", "--trace"])

        if dry_run:
            print(command)
            return

        # Raise more intuitive error to developer if port is already in-use.
        if is_port_in_use(port, host):
            raise RuntimeError(
                f"ERROR: NATS Port {port} for device {host} already in use. Is a nats-server already running?"
            )

        if clear_store:
            shutil.rmtree(store_dir, ignore_errors=True)

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

            with open(f"{log_dir}/nats_server.stdout.log", "wt") as output_:
                with open(f"{log_dir}/nats_server.stderr.log", "wt") as output_err:
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=output_,
                        stderr=output_err,
                    )
                    self._process = process
        else:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
            )
            self._process = process

    def __del__(self):
        if self._process:
            self._process.terminate()
            self._process.kill()
            self._process.wait()


class NatsRequestPlane(RequestPlane):
    @property
    def component_id(self):
        return self._component_id

    @property
    def response_uri(self):
        return self._response_uri

    async def close(self):
        if self._nats_client:
            await self._nats_client.close()

    def __del__(self):
        if self._event_loop and not self._event_loop.is_closed():
            self._event_loop.run_until_complete(self.close())

    def __init__(
        self,
        request_plane_uri: str = DEFAULT_REQUESTS_URI,
        component_id: Optional[uuid.UUID] = None,
    ) -> None:
        self._request_plane_uri = request_plane_uri
        self._component_id = component_id if component_id else uuid.uuid1()

        self._response_stream_name = f"component-{self._component_id}-response"

        split_uri = urlsplit(self._request_plane_uri)._asdict()
        split_uri["path"] = self._response_stream_name
        self._response_uri = str(urlunsplit(split_uri.values()))

        self._model_streams: Dict[
            tuple[str, str],  # model_name, model_version
            tuple[
                str,  # stream_name
                Optional[nats.js.JetStreamContext.PullSubscription],  # general requests
                Optional[nats.js.JetStreamContext.PullSubscription],  # direct requests
            ],
        ] = {}

        self._posted_requests: Dict[
            uuid.UUID,  # request id
            tuple[
                Optional[asyncio.Queue],  # response queue
                Optional[Callable[[ModelInferResponse], None | Awaitable[None]]],
                Optional[Callable[[ModelInferResponse], Awaitable[None]]],
            ],
        ] = {}
        self._jet_stream: Optional[nats.js.JetStreamContext] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    def _replace_special_chars(self, stream_name):
        return stream_name.replace(".", "-")

    async def _get_model_stream(
        self, model_name: str, model_version: str, subscribe: bool
    ) -> tuple[
        str,
        Optional[nats.js.JetStreamContext.PullSubscription],
        Optional[nats.js.JetStreamContext.PullSubscription],
    ]:
        if self._jet_stream is None:
            raise ValueError(
                "Failed to get model stream: NATS Jetstream not connected!"
            )

        if (model_name, model_version) in self._model_streams:
            return self._model_streams[(model_name, model_version)]

        model_stream_name = self._replace_special_chars(
            f"model-{model_name}-{model_version}"
        )
        await self._jet_stream.add_stream(
            name=model_stream_name,
            subjects=[model_stream_name, model_stream_name + ".*"],
            retention=nats.js.api.RetentionPolicy.WORK_QUEUE,
        )

        general_requests = None
        directed_requests = None

        if subscribe:
            general_requests = await self._jet_stream.pull_subscribe(
                subject=model_stream_name,
                stream=model_stream_name,
                durable=model_stream_name,
            )
            directed_subject = f"{model_stream_name}.{self._component_id}"
            directed_durable = f"{model_stream_name}-{self._component_id}"
            directed_requests = await self._jet_stream.pull_subscribe(
                subject=directed_subject,
                stream=model_stream_name,
                durable=directed_durable,
            )

        return self._model_streams.setdefault(
            (model_name, model_version),
            (model_stream_name, general_requests, directed_requests),
        )

    async def _response_callback(self, message):
        await message.ack()
        response = ModelInferResponse()
        response.ParseFromString(message.data)
        request_id = get_icp_request_id(response)
        if request_id in self._posted_requests:
            response_queue, handler, async_handler = self._posted_requests[request_id]
            if get_icp_final_response(response):
                del self._posted_requests[request_id]
            if response_queue:
                return await response_queue.put(response)
            if async_handler is not None:
                return await async_handler(response)
            if handler is not None:
                return handler(response)

    async def connect(self):
        self._nats_client = await nats.connect(self._request_plane_uri)
        self._jet_stream = self._nats_client.jetstream()
        self._event_loop = asyncio.get_event_loop()

        await self._jet_stream.add_stream(
            name=self._response_stream_name,
            subjects=[self._response_stream_name],
            retention=nats.js.api.RetentionPolicy.WORK_QUEUE,
        )

        await self._jet_stream.subscribe(
            self._response_stream_name,
            cb=self._response_callback,
            durable=self._response_stream_name,
            stream=self._response_stream_name,
        )

    async def pull_requests(
        self,
        model_name: str,
        model_version: str,
        number_requests: int = 1,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[ModelInferRequest]:
        # Note directed requests and general requests are
        # pulled in parallel. Directed requests are consumed
        # first. If there are more requests than the batch size
        # then extra requests are scheduled for redlivery via nak

        requests: list[ModelInferRequest] = []
        acks = []
        _, general, directed = await self._get_model_stream(
            model_name, model_version, subscribe=True
        )

        tasks = [
            asyncio.create_task(
                subscription.fetch(batch=number_requests, timeout=timeout)
            )
            for subscription in [directed, general]
            if subscription
        ]

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        for task in pending:
            task.cancel()

        for task in tasks:
            if task not in done:
                continue
            try:
                for message in task.result():
                    if len(requests) < number_requests:
                        request = ModelInferRequest()
                        request.ParseFromString(message.data)
                        requests.append(request)
                        acks.append(message.ack())
                    else:
                        acks.append(message.nak())
            except nats.errors.TimeoutError:
                continue

        asyncio.gather(*acks)

        return AsyncModelInferRequestIterator(requests)

    @staticmethod
    async def _single_response(response: ModelInferResponse):
        yield response

    async def post_response(
        self,
        request: ModelInferRequest,
        responses: AsyncIterator[ModelInferResponse] | ModelInferResponse,
    ):
        if self._jet_stream is None:
            raise ValueError("Failed to post response: NATS Jetstream not connected!")

        request_id = get_icp_request_id(request)
        if request_id is None:
            raise ValueError("ICP request must have request id")

        response_to_uri = get_icp_response_to_uri(request)
        if not response_to_uri:
            raise ValueError("Attempting to send a response when non requested")

        parsed = urlsplit(response_to_uri)
        response_stream = parsed.path.replace("/", "")

        if isinstance(responses, ModelInferResponse):
            responses = NatsRequestPlane._single_response(responses)

        async for response in responses:
            set_icp_request_id(response, request_id)
            response.model_name = request.model_name
            response.model_version = request.model_version
            response.id = request.id
            set_icp_component_id(response, self._component_id)
            await self._jet_stream.publish(
                response_stream,
                response.SerializeToString(),
                stream=response_stream,
            )

    async def post_request(
        self,
        request: ModelInferRequest,
        *,
        component_id: Optional[uuid.UUID] = None,
        response_iterator: bool = False,
        response_handler: Optional[
            Callable[[ModelInferResponse], None | Awaitable[None]]
        ] = None,
    ) -> AsyncIterator[ModelInferResponse]:
        if self._jet_stream is None:
            raise ValueError("Failed to post request: NATS Jetstream not connected!")

        if response_iterator and response_handler:
            raise ValueError(
                "Can only specify either response handler or response iterator"
            )

        async_response_handler = None

        response_queue = None
        if response_handler or response_iterator:
            request_id = get_icp_request_id(request)
            if request_id is None:
                request_id = uuid.uuid1()
                set_icp_request_id(request, request_id)

            set_icp_response_to_uri(request, self._response_uri)
            set_icp_component_id(request, self._component_id)

            async_response_handler = (
                response_handler
                if asyncio.iscoroutinefunction(response_handler)
                else None
            )

            response_queue = None

            if response_iterator:
                response_queue = asyncio.Queue()
            self._posted_requests[request_id] = (
                response_queue,
                response_handler,
                async_response_handler,
            )

        stream_name, _, _ = await self._get_model_stream(
            request.model_name, request.model_version, subscribe=False
        )
        subject = stream_name

        if component_id:
            subject += f".{component_id}"

        split_uri = urlsplit(self._request_plane_uri)._asdict()
        split_uri["path"] = subject
        set_icp_request_to_uri(request, str(urlunsplit(split_uri.values())))
        await self._jet_stream.publish(
            subject,
            request.SerializeToString(),
            stream=stream_name,
        )

        return AsyncModelInferResponseIterator(response_queue)
