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
import re
import uuid
from abc import abstractmethod
from datetime import datetime
from typing import Any, AsyncIterator, Awaitable, Callable, List, Optional, Union

import msgspec

EVENT_TOPIC_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def _validate_subjects(subjects: List[str]) -> bool:
    """
    Checks if all strings in the list are alphanumeric and can contain underscores (_) and hyphens (-).

    :param subjects: List of strings to validate
    :return: True if all strings are valid, False otherwise
    """
    pattern = EVENT_TOPIC_PATTERN

    return all(pattern.match(subject) for subject in subjects)


@dataclasses.dataclass
class EventTopic:
    """Event event_topic class for identifying event streams."""

    event_topic: str

    def __init__(self, event_topic: Union[List[str], str]):
        """Initialize the event_topic.

        Args:
            event_topic (Union[List[str], str]): The event_topic as a list of strings or a single string. Strings should be alphanumeric + underscore and '-' characters only. The list forms a hierarchy of topics.
        """

        if isinstance(event_topic, str):
            event_topic = [event_topic]
        if not _validate_subjects(event_topic):
            raise ValueError(
                "Invalid event_topic string. Only alphanumeric characters, underscores, and hyphens are allowed."
            )
        event_topic = ".".join(event_topic)
        self.event_topic = event_topic

    def __str__(self):
        return self.event_topic


@dataclasses.dataclass
class EventMetadata:
    """
    Class keeps metadata of an event.
    """

    event_id: uuid.UUID
    event_type: str
    timestamp: datetime
    component_id: uuid.UUID
    event_topic: Optional[EventTopic] = None


def _deserialize_metadata(event_metadata_serialized: bytes):
    event_metadata_dict = msgspec.json.decode(event_metadata_serialized)
    topic_meta = event_metadata_dict["event_topic"]
    topic_list = topic_meta["event_topic"].split(".")
    metadata = EventMetadata(
        **{
            **event_metadata_dict,
            "event_topic": EventTopic(topic_list)
            if event_metadata_dict["event_topic"]
            else None,
            "event_id": uuid.UUID(event_metadata_dict["event_id"]),
            "component_id": uuid.UUID(event_metadata_dict["component_id"]),
            "timestamp": datetime.fromisoformat(event_metadata_dict["timestamp"]),
        }
    )
    return metadata


def _serialize_metadata(event_metadata: EventMetadata) -> bytes:
    def hook(obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, EventTopic):
            return list(obj.event_topic.split("."))
        else:
            raise NotImplementedError(f"Type {type(obj)} is not serializable.")

    json_string = msgspec.json.encode(event_metadata, enc_hook=hook)
    return json_string


class Event:
    """Event class for representing events."""

    def __init__(
        self,
        payload: bytes,
        event_metadata_serialize: bytes,
        event_metadata: Optional[EventMetadata] = None,
        metadata_deserialize: Optional[
            Callable[[bytes], EventMetadata]
        ] = _deserialize_metadata,
    ):
        """Initialize the event.

        Args:
            event_metadata (EventMetadata): Event metadata
            event (bytes): Event payload
        """
        self._payload = payload
        self._event_metadata_serialize = event_metadata_serialize
        self._event_metadata = event_metadata
        self._metadata_deserialize = metadata_deserialize

    @property
    def _metadata(self):
        if not self._event_metadata:
            if not self._metadata_deserialize:
                raise ValueError("No metadata deserialization function provided.")
            self._event_metadata = self._metadata_deserialize(
                self._event_metadata_serialize
            )
        return self._event_metadata

    @property
    def event_id(self) -> uuid.UUID:
        return self._metadata.event_id

    @property
    def event_type(self) -> str:
        return self._metadata.event_type

    @property
    def timestamp(self) -> datetime:
        return self._metadata.timestamp

    @property
    def component_id(self) -> uuid.UUID:
        return self._metadata.component_id

    @property
    def event_topic(self) -> Optional[EventTopic]:
        return self._metadata.event_topic

    @property
    def payload(self) -> bytes:
        return self._payload


class EventSubscription(AsyncIterator[Event]):
    @abstractmethod
    async def __anext__(self) -> Event:
        pass

    @abstractmethod
    def __aiter__(self):
        return self

    @abstractmethod
    def unsubscribe(self):
        pass


class EventPlane:
    """EventPlane interface for publishing and subscribing to events."""

    @abstractmethod
    async def connect(self):
        """Connect to the event plane."""
        pass

    @abstractmethod
    async def publish(
        self,
        event: Union[bytes, Any],
        event_type: str,
        event_topic: Optional[EventTopic],
    ) -> Event:
        """Publish an event to the event plane.

        Args:
            event (Union[bytes, Any]): Event payload
            event_type (str): Event type
            event_topic (Optional[EventTopic]): Event event_topic
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
        """Subscribe to events on the event plane.

        Args:
            callback (Callable[[bytes, bytes], Awaitable[None]]): Callback function to be called when an event is received
            event_topic (Optional[EventTopic]): Event event_topic
            event_type (Optional[str]): Event type
            component_id (Optional[uuid.UUID]): Component ID
        """
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from the event plane."""
        pass
