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
import os
import uuid
from datetime import datetime
from typing import Awaitable, Callable, Optional

import nats

from triton_distributed.icp import EventMetadata, EventTopic
from triton_distributed.icp.event_plane import (
    Event,
    EventSubscription,
    _serialize_metadata,
)

DEFAULT_EVENTS_PORT = int(os.getenv("DEFAULT_EVENTS_PORT", 4222))
DEFAULT_EVENTS_HOST = os.getenv("DEFAULT_EVENTS_HOST", "localhost")
DEFAULT_EVENTS_URI = os.getenv(
    "DEFAULT_EVENTS_URI", f"nats://{DEFAULT_EVENTS_HOST}:{DEFAULT_EVENTS_PORT}"
)


class NatsEventSubscription(EventSubscription):
    """
    Manages a subscription to NATS events. This class implements an asynchronous iterator
    interface to receive events one by one.

    Attributes:
        _nc_sub (Optional[nats.aio.subscription.Subscription]): The subscription object from NATS.
    """

    def __init__(self, nc_sub: nats.aio.subscription.Subscription):
        """
        Initializes the NatsEventSubscription with an existing NATS subscription.

        Args:
            nc_sub (nats.aio.subscription.Subscription):
                The subscription object provided by NATS.
        """
        self._nc_sub: Optional[nats.aio.subscription.Subscription] = nc_sub

    async def __anext__(self) -> Event:
        """
        Asynchronously retrieves the next message from the subscription. Converts it into
        an `Event` before returning.

        Returns:
            Event: The next event object including metadata and payload.

        Raises:
            StopAsyncIteration: If the subscription is already unsubscribed or closed.
        """
        if self._nc_sub is None:
            raise StopAsyncIteration
        msg = await self._nc_sub.next_msg()
        metadata, event_payload = self._extract_metadata_and_payload(msg.data)
        event = Event(event_payload, metadata)
        return event

    def __aiter__(self):
        """
        Returns the iterator object for asynchronous iteration. Required by the
        `AsyncIterator` interface.
        """
        return self

    async def unsubscribe(self):
        """
        Unsubscribes from the NATS subject. After calling this method, the subscription can
        no longer yield events.
        """
        if self._nc_sub is None:
            return
        await self._nc_sub.unsubscribe()
        self._nc_sub = None

    @staticmethod
    def _extract_metadata_and_payload(message: bytes):
        """
        Extracts serialized metadata and the payload from a raw message.

        The first 4 bytes indicate the length of the metadata payload. The method splits
        the message accordingly.

        Args:
            message (bytes): Raw message from NATS.

        Returns:
            tuple[bytes, bytes]: A tuple (metadata_serialized, payload).
        """
        message_view = memoryview(message)
        metadata_size = int.from_bytes(message_view[:4], byteorder="big")
        metadata_serialized = message_view[4 : 4 + metadata_size]
        event = message_view[4 + metadata_size :]
        return metadata_serialized, event


class NatsEventPlane:
    """
    EventPlane implementation using NATS.

    This class provides an interface to publish and subscribe to events
    via a NATS server. It encapsulates NATS connection details, subject composition,
    metadata handling, and payload publishing.

    Example:
        ```python
        import asyncio
        import uuid

        async def main():
            # Create a NatsEventPlane instance
            event_plane = NatsEventPlane(
                server_uri="nats://localhost:4222",
                component_id=uuid.uuid4(),
                run_callback_in_parallel=True
            )
            await event_plane.connect()

            # Publish an event
            payload = b"Hello, NATS!"
            event_topic = EventTopic("test_topic")
            published_event = await event_plane.publish(
                payload,
                event_type="test_event",
                event_topic=event_topic,
            )

            print("Published event ID:", published_event.event_id)

            # Disconnect when done
            await event_plane.disconnect()

        asyncio.run(main())
        ```
    """

    def __init__(
        self,
        server_uri: str,
        component_id: uuid.UUID,
        run_callback_in_parallel: bool = False,
    ):
        """
        Initializes the NATS event plane.

        Args:
            server_uri (str): URI of the NATS server (e.g. "nats://localhost:4222").
            component_id (uuid.UUID): Unique identifier for this component; used for
                scoping the published events.
            run_callback_in_parallel (bool, optional):
                If True, the event plane will create tasks for the subscription callbacks,
                allowing them to run in parallel. Defaults to False.
        """
        self._run_callback_in_parallel = run_callback_in_parallel
        self._server_uri = server_uri
        self._component_id = component_id
        self._nc = nats.NATS()

    async def connect(self):
        """
        Connects to the NATS server using the configured URI.

        This method should be called before publishing or subscribing to events.

        Raises:
            nats.aio.errors.ErrNoServers: If the client fails to connect to any specified server.
        """
        await self._nc.connect(self._server_uri)

    async def publish(
        self, payload: bytes, event_type: str, event_topic: Optional[EventTopic]
    ) -> Event:
        """
        Publishes an event to the NATS server.

        The method constructs a metadata object, serializes it, and prepends it
        to the payload. This message is then published to a subject derived from
        the event type, component ID, and optionally the event topic.

        Args:
            payload (bytes):
                Event payload as raw bytes. Example: `b'example payload'`.
            event_type (str):
                A short string classifying the type of event. Typically used as a filter
                when subscribing.
            event_topic (Optional[EventTopic]):
                The hierarchical topic (or topics) under which this event is published.
                Use None to publish without a topic suffix.

        Returns:
            Event: An `Event` instance containing both the payload and the metadata used
            during publication.

        Example:
            ```python
            event_plane = NatsEventPlane("nats://localhost:4222", component_id=uuid.uuid4())
            await event_plane.connect()
            published = await event_plane.publish(
                b"My Event Payload",
                event_type="example_event",
                event_topic=EventTopic("some_topic")
            )
            print("Published event ID:", published.event_id)
            ```
        """
        event_metadata = EventMetadata(
            event_id=uuid.uuid4(),
            event_topic=event_topic,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            component_id=self._component_id,
        )

        metadata_serialized = _serialize_metadata(event_metadata)
        metadata_size = len(metadata_serialized).to_bytes(4, byteorder="big")

        # Concatenate metadata size, metadata, and event payload
        message = metadata_size + metadata_serialized + payload

        subject = self._compose_publish_subject(event_metadata)
        await self._nc.publish(subject, message)

        event_with_metadata = Event(payload, metadata_serialized, event_metadata)
        return event_with_metadata

    async def subscribe(
        self,
        callback: Optional[Callable[[Event], Awaitable[None]]],
        event_topic: Optional[EventTopic] = None,
        event_type: Optional[str] = None,
        component_id: Optional[uuid.UUID] = None,
    ) -> EventSubscription:
        """
        Subscribes to events on the NATS server that match the specified filters.

        When an event is received, it is converted into an `Event` object and passed
        to the callback. Depending on `run_callback_in_parallel`, the callback may be run
        in separate tasks or sequentially.

        Args:
            callback (Optional[Callable[[Event], Awaitable[None]]]):
                A coroutine that processes incoming `Event` objects. If None, the subscription
                can still be used as an async iterator to read events.
            event_topic (Optional[EventTopic], optional):
                Event topic filter. Use None to allow all topics. Defaults to None.
            event_type (Optional[str], optional):
                Event type filter. Use None to allow all event types. Defaults to None.
            component_id (Optional[uuid.UUID], optional):
                Component ID filter. If provided, only events from this component are received.
                Defaults to None.

        Returns:
            EventSubscription:
                An object that can be used to retrieve events via iteration or
                manually unsubscribe from the subject.

        Example:
            ```python
            import asyncio
            import uuid

            async def handle_event(event: Event):
                print("Received event:", event.event_id, event.payload)

            async def main():
                event_plane = NatsEventPlane("nats://localhost:4222", uuid.uuid4())
                await event_plane.connect()

                subscription = await event_plane.subscribe(
                    callback=handle_event,
                    event_topic=EventTopic("some_topic"),
                    event_type="some_event_type",
                )

                # Alternatively, handle events via iteration:
                async for evt in subscription:
                    print("Iterated event ID:", evt.event_id)

            asyncio.run(main())
            ```
        """

        async def _message_handler(msg):
            metadata, event_payload = self._extract_metadata_and_payload(msg.data)
            event = Event(event_payload, metadata)

            async def wrapper():
                if callback is not None:
                    await callback(event)

            if self._run_callback_in_parallel:
                if callback is not None:
                    asyncio.create_task(wrapper())
            else:
                if callback is not None:
                    await callback(event)

        subject = self._compose_subscribe_subject(event_topic, event_type, component_id)

        _cb = _message_handler if callback is not None else None
        sub = await self._nc.subscribe(subject, cb=_cb)
        event_sub = NatsEventSubscription(sub)

        return event_sub

    async def disconnect(self):
        """
        Disconnects from the NATS server.

        This method should be called when the event plane is no longer needed, ensuring
        that all NATS connections are properly closed.
        """
        await self._nc.close()

    def _compose_publish_subject(self, event_metadata: EventMetadata) -> str:
        """
        Composes the subject string for publishing events based on event type, component ID,
        and (optionally) event topic.

        Args:
            event_metadata (EventMetadata): The metadata associated with the event.

        Returns:
            str: The NATS subject string for publishing (e.g. `"ep.my_event_type.<component_uuid>.<topic>.trunk"`).
        """
        return (
            f"ep.{event_metadata.event_type}.{event_metadata.component_id}."
            f"{str(event_metadata.event_topic) + '.' if event_metadata.event_topic else ''}trunk"
        )

    def _compose_subscribe_subject(
        self,
        event_topic: Optional[EventTopic],
        event_type: Optional[str],
        component_id: Optional[uuid.UUID],
    ) -> str:
        """
        Composes the wildcard subscription subject for NATS based on optional filters.

        Args:
            event_topic (Optional[EventTopic]): Filter by event topic, or None to allow all topics.
            event_type (Optional[str]): Filter by event type, or None to allow all types.
            component_id (Optional[uuid.UUID]): Filter by component ID, or None for all components.

        Returns:
            str: NATS wildcard subscription subject (e.g. `"ep.some_event_type.some_uuid.some_topic.>"`).
        """
        return (
            f"ep.{event_type or '*'}.{component_id or '*'}."
            f"{str(event_topic) + '.' if event_topic else ''}>"
        )

    def _extract_metadata_and_payload(self, message: bytes):
        """
        Extracts serialized metadata and payload from a published message. The first 4 bytes
        determine the size of the metadata block, which is used to separate the metadata
        from the actual event payload.

        Args:
            message (bytes): Raw message as received from NATS.

        Returns:
            tuple[bytes, bytes]: (metadata_serialized, event_payload).
        """
        message_view = memoryview(message)
        metadata_size = int.from_bytes(message_view[:4], byteorder="big")

        # Extract metadata and event
        metadata_serialized = message_view[4 : 4 + metadata_size]
        event = message_view[4 + metadata_size :]

        return metadata_serialized, event
