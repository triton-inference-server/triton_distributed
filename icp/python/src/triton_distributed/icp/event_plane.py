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
from typing import Any, AsyncIterator, Awaitable, Callable, List, Optional, Tuple, Union

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

    @classmethod
    def from_raw(cls, event_metadata_serialized: bytes):
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

    def to_raw(self) -> bytes:
        serialized = {}
        for key, value in self.__dict__.items():
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
    ) -> EventMetadata:
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
    ):
        """Subscribe to events on the event plane.

        Args:
            callback (Callable[[bytes, bytes], Awaitable[None]]): Callback function to be called when an event is received
            event_topic (Optional[EventTopic]): Event event_topic
            event_type (Optional[str]): Event type
            component_id (Optional[uuid.UUID]): Component ID
        """
        pass

    @abstractmethod
    def subscribe_iter(
        self,
        event_topic: Optional[EventTopic] = None,
        event_type: Optional[str] = None,
        component_id: Optional[uuid.UUID] = None,
    ) -> AsyncIterator[Tuple[bytes, bytes]]:
        """
        Subscribe to events on the event plane and return an async iterator.

        Args:
            event_topic (Optional[EventTopic]): Event event_topic
            event_type (Optional[str]): Event type
            component_id (Optional[uuid.UUID]): Component ID
        """
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from the event plane."""
        pass
