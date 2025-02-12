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
from datetime import datetime
from typing import Optional

import msgspec

from triton_distributed.icp.event_plane import Event, EventMetadata, EventTopic


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


class LazyEvent(Event):
    """LazyEvent class for representing events."""

    def __init__(
            self,
            payload: bytes,
            event_metadata_serialized: bytes,
            event_metadata: Optional[EventMetadata] = None,
    ):
        """Initialize the event.

        Args:
            event_metadata (EventMetadata): Event metadata
            event (bytes): Event payload
        """
        self._payload = payload
        self._event_metadata_serialized = event_metadata_serialized
        self._event_metadata = event_metadata

    @property
    def _metadata(self):
        if not self._event_metadata:
            self._event_metadata = _deserialize_metadata(
                self._event_metadata_serialized
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
