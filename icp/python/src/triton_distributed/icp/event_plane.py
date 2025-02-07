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

import dataclasses
import json
import re
import uuid
from abc import abstractmethod
from datetime import datetime
from typing import Any, AsyncIterator, Awaitable, Callable, List, Optional, Union

EVENT_TOPIC_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def _validate_subjects(subjects: List[str]) -> bool:
    """
    Checks if all strings in the list adhere to a pattern allowing alphanumeric,
    underscores (_), and hyphens (-).

    Args:
        subjects (List[str]): A list of strings to validate.

    Returns:
        bool: True if all strings match the pattern, otherwise False.

    Example:
        ```python
        valid = _validate_subjects(["my_topic", "another-topic", "123"])
        invalid = _validate_subjects(["invalid topic", "topic!"])
        print(valid)   # True
        print(invalid) # False
        ```
    """
    pattern = EVENT_TOPIC_PATTERN
    return all(pattern.match(subject) for subject in subjects)


@dataclasses.dataclass
class EventTopic:
    """
    Represents a hierarchical topic used to categorize and filter events.

    Internally, the topic is stored as a single string joined by periods (`"."`)
    if a list of strings was provided. For example: `["level1", "level2"]` -> `"level1.level2"`.
    """

    event_topic: str

    def __init__(self, event_topic: Union[List[str], str]):
        """
        Initializes the event topic.

        Args:
            event_topic (Union[List[str], str]):
                The event topic as a list of valid strings or a single string. Each string
                must match `[a-zA-Z0-9_-]+`. A list is treated as a hierarchical path.

        Raises:
            ValueError: If any part of the provided topic is invalid.
        """
        if isinstance(event_topic, str):
            event_topic = [event_topic]
        if not _validate_subjects(event_topic):
            raise ValueError(
                "Invalid event_topic string(s). Each topic part must only contain "
                "alphanumeric characters, underscores, or hyphens."
            )
        event_topic = ".".join(event_topic)
        self.event_topic = event_topic

    def __str__(self) -> str:
        """
        Returns the internal event topic as a string.

        Returns:
            str: The topic string joined by periods.
        """
        return self.event_topic


@dataclasses.dataclass
class EventMetadata:
    """
    Stores metadata describing an event, including identification, origin,
    categorization, and timestamps.

    Attributes:
        event_id (uuid.UUID): Unique ID for this event.
        event_type (str): The type or category of the event.
        timestamp (datetime): The creation or publication time of the event.
        component_id (uuid.UUID): Identifier of the component that published the event.
        event_topic (Optional[EventTopic]): Hierarchical topic categorizing the event.
    """

    event_id: uuid.UUID
    event_type: str
    timestamp: datetime
    component_id: uuid.UUID
    event_topic: Optional[EventTopic] = None


def _deserialize_metadata(event_metadata_serialized: bytes) -> EventMetadata:
    """
    Deserializes event metadata from a JSON-encoded byte string.

    The expected JSON structure matches the `EventMetadata` dataclass, with
    `event_topic` stored as a list and certain fields (UUIDs, datetime) needing
    reconstruction.

    Args:
        event_metadata_serialized (bytes): JSON-encoded metadata.

    Returns:
        EventMetadata: A reconstructed `EventMetadata` object.

    Example:
        ```python
        raw_metadata = b'{
            "event_id": "7dbacac6-8789-4736-bad3-73d4d584086a",
            "event_type": "example",
            "timestamp": "2025-01-01T00:00:00",
            "component_id": "2cc10541-c23f-4645-b617-bd5488bcc4fb",
            "event_topic": ["my_topic"]
        }'
        metadata = _deserialize_metadata(raw_metadata)
        print(metadata.event_id)  # 7dbacac6-...
        ```
    """
    event_metadata_dict = json.loads(event_metadata_serialized.decode("utf-8"))
    metadata = EventMetadata(
        **{
            **event_metadata_dict,
            "event_topic": EventTopic(**event_metadata_dict["event_topic"])
            if event_metadata_dict["event_topic"]
            else None,
            "event_id": uuid.UUID(event_metadata_dict["event_id"]),
            "component_id": uuid.UUID(event_metadata_dict["component_id"]),
            "timestamp": datetime.fromisoformat(event_metadata_dict["timestamp"]),
        }
    )
    return metadata


def _serialize_metadata(event_metadata: EventMetadata) -> bytes:
    """
    Serializes an `EventMetadata` instance to JSON, suitable for storage or
    transmission. Certain fields like UUID or datetime objects are converted to
    string representations.

    Args:
        event_metadata (EventMetadata): The metadata object to serialize.

    Returns:
        bytes: JSON-encoded representation of the metadata.

    Example:
        ```python
        metadata = EventMetadata(
            event_id=uuid.uuid4(),
            event_type="example",
            timestamp=datetime.utcnow(),
            component_id=uuid.uuid4(),
            event_topic=EventTopic("my_topic")
        )
        serialized = _serialize_metadata(metadata)
        print(serialized.decode("utf-8"))
        ```
    """
    serialized: dict[str, Any] = {}
    for key, value in event_metadata.__dict__.items():
        if isinstance(value, uuid.UUID):
            serialized[key] = str(value)
        elif isinstance(value, datetime):
            serialized[key] = value.isoformat()
        elif isinstance(value, EventTopic):
            serialized[key] = list(value.event_topic.split("."))
        else:
            serialized[key] = value
    json_string = json.dumps(serialized, indent=4)
    return json_string.encode("utf-8")


class Event:
    """
    Represents a single event with metadata and payload. The payload is stored
    in bytes form, while metadata may be lazily deserialized only when needed.

    Typical usage:
        - Access the payload via `event.payload`.
        - Access event_id, event_type, timestamp, etc., via their respective properties.
    """

    def __init__(
        self,
        payload: bytes,
        event_metadata_serialize: bytes,
        event_metadata: Optional[EventMetadata] = None,
        metadata_deserialize: Optional[
            Callable[[bytes], EventMetadata]
        ] = _deserialize_metadata,
    ):
        """
        Initializes an Event object.

        Args:
            payload (bytes): The raw payload of the event.
            event_metadata_serialize (bytes):
                The serialized metadata, stored for lazy deserialization if
                `event_metadata` is not provided.
            event_metadata (Optional[EventMetadata], optional):
                Deserialized event metadata object. If None, `metadata_deserialize` is used
                to deserialize `event_metadata_serialize` on demand.
            metadata_deserialize (Optional[Callable[[bytes], EventMetadata]], optional):
                A function that takes serialized bytes and returns an `EventMetadata`.
                Defaults to `_deserialize_metadata`.
        """
        self._payload = payload
        self._event_metadata_serialize = event_metadata_serialize
        self._event_metadata = event_metadata
        self._metadata_deserialize = metadata_deserialize

    @property
    def _metadata(self) -> EventMetadata:
        """
        Lazily fetches the event metadata, deserializing it if necessary.

        Returns:
            EventMetadata: The event's metadata.

        Raises:
            ValueError: If no deserialization function is available.
        """
        if not self._event_metadata:
            if not self._metadata_deserialize:
                raise ValueError("No metadata deserialization function provided.")
            self._event_metadata = self._metadata_deserialize(
                self._event_metadata_serialize
            )
        return self._event_metadata

    @property
    def event_id(self) -> uuid.UUID:
        """Returns the unique UUID of this event."""
        return self._metadata.event_id

    @property
    def event_type(self) -> str:
        """Returns the type of this event (e.g. 'error', 'info', 'metrics')."""
        return self._metadata.event_type

    @property
    def timestamp(self) -> datetime:
        """Returns the timestamp at which this event was created."""
        return self._metadata.timestamp

    @property
    def component_id(self) -> uuid.UUID:
        """Returns the UUID of the component that generated this event."""
        return self._metadata.component_id

    @property
    def event_topic(self) -> Optional[EventTopic]:
        """Returns the topic (or topics) associated with this event."""
        return self._metadata.event_topic

    @property
    def payload(self) -> bytes:
        """Returns the raw payload of this event as bytes."""
        return self._payload


class EventSubscription(AsyncIterator[Event]):
    """
    Abstract base class representing a subscription to a continuous stream
    of events. Provides async iteration over `Event` objects.
    """

    @abstractmethod
    async def __anext__(self) -> Event:
        """
        Retrieves the next Event from the subscription.

        Raises:
            StopAsyncIteration: If no further events are available or the subscription
                                has been unsubscribed.

        Returns:
            Event: The next event in the subscription.
        """
        pass

    @abstractmethod
    def __aiter__(self):
        """
        Returns the async iterator for the subscription. Typically returns `self`.
        """
        return self

    @abstractmethod
    async def unsubscribe(self):
        """
        Unsubscribes from further event notifications. After calling this method,
        any attempt to receive further events should raise `StopAsyncIteration`.
        """
        pass


class EventPlane:
    """
    Abstract interface for an event publishing and subscribing system.

    Users of this interface can publish new events or subscribe to existing
    event streams. Concrete implementations might be based on different
    backends such as NATS, Kafka, or in-process pub/sub systems.
    """

    @abstractmethod
    async def connect(self):
        """
        Establishes any necessary connection to the underlying event system.

        For example, if using a message broker, this might open a socket,
        authenticate, etc.
        """
        pass

    @abstractmethod
    async def publish(
        self,
        event: Union[bytes, Any],
        event_type: str,
        event_topic: Optional[EventTopic],
    ) -> EventMetadata:
        """
        Publishes an event to the event plane.

        Args:
            event (Union[bytes, Any]):
                The event payload. Implementations may convert non-bytes objects
                to bytes as needed.
            event_type (str):
                Identifies the category or type of event for filtering or routing.
            event_topic (Optional[EventTopic]):
                An optional hierarchical topic or sub-topic for further classification.

        Returns:
            EventMetadata: The metadata for the published event.

        Example:
            ```python
            async def send_example_event(plane: EventPlane):
                metadata = await plane.publish(
                    event=b"some data",
                    event_type="example",
                    event_topic=EventTopic("my_topic")
                )
                print("Event published with ID:", metadata.event_id)
            ```
        """
        pass

    @abstractmethod
    async def subscribe(
        self,
        callback: Callable[[bytes, bytes], Awaitable[None]],
        event_topic: Optional[EventTopic] = None,
        event_type: Optional[str] = None,
        component_id: Optional[uuid.UUID] = None,
    ) -> EventSubscription:
        """
        Subscribes to events on the event plane that match the specified filters.
        When an event is received, the given callback is invoked with the raw
        metadata bytes and payload bytes.

        NOTE: This interface signature differs from the concrete implementation in
        `NatsEventPlane`, which passes an `Event` object to the callback. In this abstract
        interface, it is shown as two separate byte parameters. Implementations may decide
        to pass them combined or as an `Event` instance.

        Args:
            callback (Callable[[bytes, bytes], Awaitable[None]]):
                A coroutine that processes two byte sequences: one for metadata, one for payload.
            event_topic (Optional[EventTopic]):
                Filter to a specific topic or sub-topics, or None to receive all.
            event_type (Optional[str]):
                Filter to a specific event type, or None to receive all types.
            component_id (Optional[uuid.UUID]):
                Restricts events to those originating from a specific component.
                Use None for no restriction.

        Returns:
            EventSubscription:
                A handle that can be used to asynchronously iterate over events or unsubscribe.
        """
        pass

    @abstractmethod
    async def disconnect(self):
        """
        Closes the connection to the underlying event system, if applicable. After calling
        this method, no further events can be published or received.
        """
        pass
