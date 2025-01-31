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
from typing import Any, AsyncIterator, Awaitable, Callable, List, Optional, Union

from pydantic import BaseModel


class Topic(BaseModel):
    """Event topic class for identifying event streams."""

    topic: str

    def __init__(self, topic: Union[List[str], str]):
        if isinstance(topic, str):
            _topic = topic
        else:
            _topic = ".".join(topic)
        super().__init__(topic=_topic)

    def __str__(self):
        return self.topic


class EventMetadata(BaseModel):
    event_id: uuid.UUID
    topic: Optional[Topic] = None
    event_type: str
    timestamp: datetime
    component_id: uuid.UUID


class EventMetadataWrapped:
    def __init__(self, event_metadata_serialized: bytes):
        self._event_metadata_serialized = event_metadata_serialized

    def get_metadata(self) -> EventMetadata:
        return EventMetadata.model_validate_json(self._event_metadata_serialized)


class EventPlane:
    """EventPlane interface for publishing and subscribing to events."""

    @abstractmethod
    async def connect(self):
        pass

    @abstractmethod
    async def publish(
        self, payload: Union[bytes, Any], event_type: str, topic: Optional[Topic]
    ) -> EventMetadata:
        pass

    @abstractmethod
    async def subscribe(
        self,
        callback: Callable[[bytes, EventMetadataWrapped], Awaitable[None]],
        topic: Optional[Topic] = None,
        event_type: Optional[str] = None,
        component_id: Optional[uuid.UUID] = None,
    ):
        pass

    @abstractmethod
    async def subscribe_iter(
        self,
        topic: Optional[Topic] = None,
        event_type: Optional[str] = None,
        component_id: Optional[uuid.UUID] = None,
    ) -> AsyncIterator[bytes, EventMetadataWrapped]:
        pass

    @abstractmethod
    async def disconnect(self):
        pass
