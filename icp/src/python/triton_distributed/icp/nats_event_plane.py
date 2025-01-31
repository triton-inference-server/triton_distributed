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
from typing import Callable, Optional

import nats

from triton_distributed.icp import EventMetadata, EventMetadataWrapped, Topic


class NatsEventPlane:
    """EventPlane implementation using NATS."""

    def __init__(self, server_uri: str, component_id: uuid.UUID):
        self._server_uri = server_uri
        self._component_id = component_id
        self._nc = nats.NATS()

    async def connect(self):
        await self._nc.connect(self._server_uri)

    async def publish(
        self, payload: bytes, event_type: str, topic: Optional[Topic]
    ) -> EventMetadata:
        event_metadata = EventMetadata(
            event_id=uuid.uuid4(),
            topic=topic,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            component_id=self._component_id,
        )

        metadata_serialized = event_metadata.model_dump_json().encode("utf-8")
        metadata_size = len(metadata_serialized).to_bytes(4, byteorder="big")

        # Concatenate metadata size, metadata, and payload
        message = metadata_size + metadata_serialized + payload

        subject = self._compose_publish_subject(event_metadata)
        await self._nc.publish(subject, message)
        return event_metadata

    async def subscribe(
        self,
        callback: Callable[[bytes, EventMetadataWrapped], None],
        topic: Optional[Topic] = None,
        event_type: Optional[str] = None,
        component_id: Optional[uuid.UUID] = None,
    ):
        async def _message_handler(msg):
            metadata, payload = self._extract_metadata_and_payload(msg.data)
            await callback(payload, EventMetadataWrapped(metadata))

        subject = self._comoase_subscribe_subject(topic, event_type, component_id)
        await self._nc.subscribe(subject, cb=_message_handler)

    async def disconnect(self):
        await self._nc.close()

    def _compose_publish_subject(self, event_metadata: EventMetadata):
        return f"ep.{event_metadata.event_type}.{event_metadata.component_id}.{str(event_metadata.topic) + '.' if event_metadata.topic else ''}trunk"

    def _comoase_subscribe_subject(
        self,
        topic: Optional[Topic],
        event_type: Optional[str],
        component_id: Optional[uuid.UUID],
    ):
        return f"ep.{event_type or '*'}.{component_id or '*'}.{str(topic) + '.' if topic else ''}>"

    def _extract_metadata_and_payload(self, message: bytes):
        # Extract metadata size
        metadata_size = int.from_bytes(message[:4], byteorder="big")

        # Extract metadata and payload
        metadata_serialized = message[4 : 4 + metadata_size]
        payload = message[4 + metadata_size :]

        return metadata_serialized, payload
