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
import datetime
import logging
import os
import uuid
from typing import Any, Awaitable, Callable, List, Optional, Union

import msgspec
import nats

from triton_distributed.icp import EventTopic
from triton_distributed.icp.event_plane import Event, EventSubscription
from triton_distributed.icp.on_demand_event import (
    EventMetadata,
    OnDemandEvent,
    _serialize_metadata,
)

logger = logging.getLogger(__name__)

DEFAULT_EVENTS_PORT = int(os.getenv("DEFAULT_EVENTS_PORT", 4222))
DEFAULT_EVENTS_HOST = os.getenv("DEFAULT_EVENTS_HOST", "localhost")
DEFAULT_EVENTS_PROTOCOL = os.getenv("DEFAULT_EVENTS_PROTOCOL", "nats")
DEFAULT_CONNECTION_TIMEOUT = int(os.getenv("DEFAULT_CONNECTION_TIMEOUT", 30))

EVENT_PLANE_NATS_PREFIX = "event_plane_nats_v1"


def compose_nats_uri(
    protocol: str = DEFAULT_EVENTS_PROTOCOL,
    host: str = DEFAULT_EVENTS_HOST,
    port: int = DEFAULT_EVENTS_PORT,
) -> str:
    """Compose a NATS URL from components.

    Args:
        protocol: The protocol to use (tls or nats). Defaults to DEFAULT_EVENTS_PROTOCOL.
        host: The host to connect to. Defaults to DEFAULT_EVENTS_HOST.
        port: The port to connect to. Defaults to DEFAULT_EVENTS_PORT.

    Returns:
        str: The composed NATS URL
    """
    return f"{protocol}://{host}:{port}"


class NatsEventSubscription(EventSubscription):
    def __init__(
        self,
        nc_sub: nats.aio.subscription.Subscription,
        nats_connection: Any,
        subject: str,
        topic: EventTopic,
    ):
        self._nc_sub: Optional[nats.aio.subscription.Subscription] = nc_sub
        self._nats = nats_connection
        self._subject = subject
        self._topic = topic
        self._unsubscribe_event: asyncio.Event = asyncio.Event()

    async def __anext__(self):
        if self._nc_sub is None:
            raise StopAsyncIteration
        if not self._nats.is_connected:
            if self._error is not None:
                raise RuntimeError(
                    f"NATS connection error: {self._error}"
                ) from self._error
            else:
                raise RuntimeError("NATS connection failure.")
        else:
            failure_task = asyncio.create_task(self._nats.wait_for_failure())
        next_task = asyncio.create_task(self._nc_sub.next_msg())
        _ = await asyncio.wait(
            [next_task, failure_task], return_when=asyncio.FIRST_COMPLETED
        )

        if failure_task.done():
            logger.warning("NATS connection failure.")
            try:
                next_task.cancel()
                await next_task
            except asyncio.CancelledError:
                pass
            raise RuntimeError("NATS connection failure.") from failure_task.exception()
        else:
            try:
                failure_task.cancel()
                await failure_task
            except asyncio.CancelledError:
                pass
            msg = next_task.result()
            metadata, event_payload = NatsEventPlane._extract_metadata_and_payload(
                msg.data
            )
            event = OnDemandEvent(event_payload, metadata)
            return event

    def __aiter__(self):
        return self

    async def unsubscribe(self):
        if self._nc_sub is None:
            return

        if self._nats.is_connected():
            await self._nc_sub.unsubscribe()
            self._nc_sub = None
        else:
            logger.warning("NATS not connected. Cannot unsubscribe.")

    @property
    def subject(self):
        return self._subject

    @property
    def topic(self):
        return self._topic

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.unsubscribe()
        return False  # Don't suppress exceptions


class NatsEventPlane:
    """EventPlane implementation using NATS."""

    def __init__(
        self,
        server_uri: str = compose_nats_uri(),
        component_id: Optional[uuid.UUID] = uuid.uuid4(),
        run_callback_in_parallel: bool = False,
    ):
        """Initialize the NATS event plane.

        Args:
            server_uri: URI of the NATS server. If None, will be composed using environment variables.
            component_id: Component ID.
        """
        self._run_callback_in_parallel = run_callback_in_parallel
        if server_uri is None:
            server_uri = compose_nats_uri()
        self._server_uri = server_uri
        if component_id is None:
            component_id = uuid.uuid4()
        self._component_id = component_id
        self._nc = nats.NATS()
        self._error: Optional[Exception] = None
        self._connected = False
        self._failure_event: Optional[asyncio.Event] = None

    async def wait_for_failure(self):
        """Wait for a failure event."""
        if self._failure_event is not None:
            await self._failure_event.wait()
            raise RuntimeError("NATS connection failure.") from self._error
        else:
            raise RuntimeError("NATS connection failure event is None")

    def is_connected(self):
        return self._connected

    async def connect(self):
        """Connect to the NATS server."""
        if self._connected:
            return

        async def error_cb(e):
            logger.warning("NATS error: %s", e)
            self._error = e
            if self._failure_event is not None:
                self._failure_event.set()
                self._failure_event = asyncio.Event()
            else:
                logger.error(f"NATS connection failure event is None for error {e}")

        async def reconnected_cb():
            logger.debug("NATS reconnected")
            self._connected = True

        async def disconnected_cb():
            logger.debug("NATS disconnected")
            self._connected = False

        async def closed_cb():
            logger.debug("NATS closed")
            self._connected = False

        self._failure_event = asyncio.Event()
        try:
            async with asyncio.timeout(DEFAULT_CONNECTION_TIMEOUT):
                logger.debug(f"Connecting to NATS server: {self._server_uri}")
                connect_task = asyncio.create_task(
                    self._nc.connect(
                        self._server_uri,
                        error_cb=error_cb,
                        reconnected_cb=reconnected_cb,
                        disconnected_cb=disconnected_cb,
                        closed_cb=closed_cb,
                    )
                )
                failed_task = asyncio.create_task(self.wait_for_failure())
                await asyncio.wait(
                    [connect_task, failed_task], return_when=asyncio.FIRST_COMPLETED
                )
                if failed_task.done():
                    try:
                        connect_task.cancel()
                        await connect_task
                    except asyncio.CancelledError:
                        pass
                    raise RuntimeError(
                        "NATS connection failure."
                    ) from failed_task.exception()
                else:
                    try:
                        failed_task.cancel()
                        await failed_task
                    except asyncio.CancelledError:
                        pass
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"NATS connection timeout {DEFAULT_CONNECTION_TIMEOUT} reached."
            )
        logger.debug(f"Connected to NATS server: {self._server_uri}")
        self._connected = True

    async def publish(
        self,
        payload: bytes | Any,
        event_type: Optional[str] = None,
        event_topic: Optional[EventTopic] = None,
        timestamp: Optional[datetime.datetime] = datetime.datetime.now(datetime.UTC),
        event_id: Optional[uuid.UUID] = uuid.uuid4(),
    ) -> Event:
        """Publish an event to the NATS server.

        Args:
            payload: Event payload.
            event_type: Type of the event.
            event_topic: EventTopic of the event.
        """
        if not self._connected:
            if self._error:
                raise RuntimeError(
                    f"NATS connection error: {self._error}"
                ) from self._error
            else:
                raise RuntimeError("NATS not connected.")

        if timestamp is None:
            timestamp = datetime.datetime.now(datetime.UTC)

        if event_id is None:
            event_id = uuid.uuid4()

        event_metadata = EventMetadata(
            event_id=event_id,
            event_topic=event_topic,
            event_type=event_type if event_type else str(type(payload).__name__),
            timestamp=timestamp,
            component_id=self._component_id,
        )

        metadata_serialized = _serialize_metadata(event_metadata)
        metadata_size = len(metadata_serialized).to_bytes(4, byteorder="big")

        # Concatenate metadata size, metadata, and event payload
        if isinstance(payload, bytes):
            message = metadata_size + metadata_serialized + payload
        else:
            message = metadata_size + metadata_serialized + msgspec.json.encode(payload)

        subject = self._compose_publish_subject(event_metadata)
        await self._nc.publish(subject, message)

        event_with_metadata = OnDemandEvent(
            payload, metadata_serialized, event_metadata
        )
        return event_with_metadata

    async def subscribe(
        self,
        callback: Optional[Callable[[Event], Awaitable[None]]] = None,
        event_topic: Optional[EventTopic | str | List[str]] = None,
        event_type: Optional[str] = "*",
        component_id: Optional[uuid.UUID] = None,
    ) -> EventSubscription:
        """Subscribe to events on the NATS server.

        Args:
            callback: Callback function to be called when an event is received.
            event_topic: Event event_topic.
            event_type: Event type.
            component_id: Component ID.
        """
        if not self._connected:
            if self._error:
                raise RuntimeError(
                    f"NATS connection error: {self._error}"
                ) from self._error
            else:
                raise RuntimeError("NATS not connected.")

        async def _message_handler(msg):
            metadata, event_payload = NatsEventPlane._extract_metadata_and_payload(
                msg.data
            )
            event = OnDemandEvent(event_payload, metadata)

            async def wrapper():
                if callback is not None:
                    await callback(event)  # Ensure it's a proper coroutine

            if self._run_callback_in_parallel:
                if callback is not None:
                    asyncio.create_task(wrapper())  # Run in parallel
            else:
                if callback is not None:
                    await callback(event)  # Await normally

        subject_str, topic = self._compose_subscribe_subject(
            event_topic, event_type, component_id
        )
        _cb = _message_handler if callback is not None else None
        sub = await self._nc.subscribe(subject_str, cb=_cb)
        event_sub = NatsEventSubscription(sub, self, subject_str, topic)

        return event_sub

    async def disconnect(self):
        """Disconnect from the NATS server."""
        if not self._connected:
            return
        await self._nc.close()
        self._error = RuntimeError("NATS connection closed by disconnect.")
        if self._failure_event is not None:
            self._failure_event.set()
        self._connected = False

    def _compose_publish_subject(self, event_metadata: EventMetadata):
        return f"{EVENT_PLANE_NATS_PREFIX}.{event_metadata.event_type}.{event_metadata.component_id}.{str(event_metadata.event_topic) + '.' if event_metadata.event_topic else ''}trunk"

    def _compose_subscribe_subject(
        self,
        event_topic: Optional[Union[EventTopic, str, List[str]]] = None,
        event_type: Optional[str] = None,
        component_id: Optional[uuid.UUID] = None,
    ):
        if isinstance(event_topic, str) or isinstance(event_topic, list):
            event_topic_obj = EventTopic(event_topic)
        else:
            event_topic_obj = event_topic
        return (
            f"{EVENT_PLANE_NATS_PREFIX}.{event_type or '*'}.{component_id or '*'}.{str(event_topic_obj) + '.' if event_topic else ''}>",
            event_topic_obj,
        )

    @staticmethod
    def _extract_metadata_and_payload(message: bytes):
        # Extract metadata size
        message_view = memoryview(message)

        metadata_size = int.from_bytes(message_view[:4], byteorder="big")

        # Extract metadata and event
        metadata_serialized = message_view[4 : 4 + metadata_size]
        event = message_view[4 + metadata_size :]

        return metadata_serialized, event

    @property
    def component_id(self) -> uuid.UUID:
        return self._component_id

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
        return False  # Don't suppress exceptions
