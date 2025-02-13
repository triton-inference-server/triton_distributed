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
import builtins
import dataclasses
import uuid
from datetime import datetime
from typing import Any, Optional, Type

import msgspec

from triton_distributed.icp.event_plane import Event, EventTopic


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
    topic_list = topic_meta["event_topic"].split(".") if topic_meta else []
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


def _get_type(type_name: str):
    # Check in builtins for the type
    builtin_type = getattr(builtins, type_name, None)
    if builtin_type and isinstance(builtin_type, type):
        return builtin_type

    # Check in globals for the type
    global_type = globals().get(type_name)
    if global_type and isinstance(global_type, type):
        return global_type

    return None


class OnDemandEvent(Event):
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

    def typed_payload(self, payload_type: Optional[Type | str] = None) -> Any:
        if payload_type is None:
            payload_type = self.event_type

        if isinstance(payload_type, str):
            payload_type = _get_type(payload_type)
        if payload_type is not None and payload_type is not bytes:
            try:
                return msgspec.json.decode(self._payload, type=payload_type)
            except Exception as e:
                raise ValueError(
                    f"Unable to convert payload {self._payload!r} to type {payload_type} from event type {self.event_type}"
                ) from e
        elif payload_type is bytes:
            return bytes(self._payload)
        else:
            raise ValueError(
                f"Unable to convert payload {self._payload!r} to type {payload_type} from event type {self.event_type}"
            )
