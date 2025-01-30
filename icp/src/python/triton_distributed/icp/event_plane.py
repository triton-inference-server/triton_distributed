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
from typing import Any, List, Optional, Union

from pydantic import BaseModel


class Topic:
    """Event topic class for identifying event streams."""

    def __init__(self, topic: Union[List[str], str]):
        if isinstance(topic, str):
            self._topic = topic
        else:
            self._topic = ".".join(topic)

    def __str__(self):
        return self._topic

    def to_string(self):
        """Convert Topic to string."""
        return str(self)

    @staticmethod
    def from_string(topic_str: str):
        """Create Topic from string."""
        return Topic(topic_str)


class EventMetadata(BaseModel):
    event_id: uuid.UUID
    topic: Topic
    event_type: str
    timestamp: datetime
    component_id: uuid.UUID


class EventMetadataWrapped:
    def __init__(self, event_metadata_serialized: bytes):
        self._event_metadata_serialized = event_metadata_serialized

    def get_metadata(self) -> EventMetadata:
        return EventMetadata.parse_raw(self._event_metadata_serialized)


class EventPlane:
    """EventPlane interface for publishing and subscribing to events."""

    @abstractmethod
    async def connect(self):
        pass

    @abstractmethod
    async def publish(
        self, event_type: str, topic: Topic, payload: Union[bytes, Any]
    ) -> EventMetadata:
        pass

    @abstractmethod
    async def subscribe(
        self,
        callback,
        topic: Optional[Topic] = None,
        event_type: Optional[str] = None,
        component_id: Optional[uuid.UUID] = None,
    ):
        pass

    @abstractmethod
    async def disconnect(self):
        pass
