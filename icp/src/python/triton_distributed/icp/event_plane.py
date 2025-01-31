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


import uuid
from abc import abstractmethod
from datetime import datetime
from typing import Any, AsyncIterator, Awaitable, Callable, List, Optional, Tuple, Union

from pydantic import BaseModel


class Topic(BaseModel):
    """Event topic class for identifying event streams."""

    topic: str

    def __init__(self, topic: Union[List[str], str]):
        """Initialize the topic.

        Args:
            topic (Union[List[str], str]): The topic as a list of strings or a single string. Strings should be alphanumeric + underscore and '-' characters only. The list forms a hierarchy of topics.
        """

        if isinstance(topic, str):
            _topic = topic
        else:
            _topic = ".".join(topic)
        super().__init__(topic=_topic)

    def __str__(self):
        return self.topic


class EventMetadata(BaseModel):
    """
    Class keeps metadata of an event.
    """

    event_id: uuid.UUID
    topic: Optional[Topic] = None
    event_type: str
    timestamp: datetime
    component_id: uuid.UUID

    @classmethod
    def from_raw(cls, event_metadata_serialized: bytes):
        return cls.model_validate_json(event_metadata_serialized)


class EventPlane:
    """EventPlane interface for publishing and subscribing to events."""

    @abstractmethod
    async def connect(self):
        pass

    @abstractmethod
    async def publish(
        self, event: Union[bytes, Any], event_type: str, topic: Optional[Topic]
    ) -> EventMetadata:
        """Publish an event to the event plane.

        Args:
        """
        pass

    @abstractmethod
    async def subscribe(
        self,
        callback: Callable[[bytes, bytes], Awaitable[None]],
        topic: Optional[Topic] = None,
        event_type: Optional[str] = None,
        component_id: Optional[uuid.UUID] = None,
    ):
        pass

    @abstractmethod
    def subscribe_iter(
        self,
        topic: Optional[Topic] = None,
        event_type: Optional[str] = None,
        component_id: Optional[uuid.UUID] = None,
    ) -> AsyncIterator[Tuple[bytes, bytes]]:
        pass

    @abstractmethod
    async def disconnect(self):
        pass
