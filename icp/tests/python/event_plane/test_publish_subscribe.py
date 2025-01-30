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
import uuid
from typing import List

import pytest

from triton_distributed.icp import NatsEventPlane, Topic
from triton_distributed.icp.event_plane import EventMetadata, EventMetadataWrapped


@pytest.mark.asyncio
class TestEventPlaneFunctional:
    @pytest.mark.asyncio
    async def test_single_publisher_subscriber(self, nats_server, event_plane):
        print(f"Print loop test: {id(asyncio.get_running_loop())}")

        received_events: List[EventMetadata] = []

        async def callback(_payload, event_metadata: EventMetadataWrapped):
            received_events.append(event_metadata.get_metadata())

        topic = Topic("test.topic")
        event_type = "test_event"
        payload = b"test_payload"

        await event_plane.subscribe(callback, topic=topic, event_type=event_type)
        event_metadata = await event_plane.publish(event_type, topic, payload)

        # Allow time for message to propagate
        await asyncio.sleep(2)

        assert len(received_events) == 1
        assert received_events[0].event_id == event_metadata.event_id

    @pytest.mark.asyncio
    async def test_one_publisher_multiple_subscribers(self, nats_server):
        results_1: List[EventMetadata] = []
        results_2: List[EventMetadata] = []
        results_3: List[EventMetadata] = []

        async def callback_1(event, _metadata):
            results_1.append(event)

        async def callback_2(event, _metadata):
            results_2.append(event)

        async def callback_3(event, _metadata):
            results_3.append(event)

        topic = Topic(["test"])
        event_type = "multi_event"
        payload = b"multi_payload"

        # async with event_plane_context() as event_plane1:
        server_url = "nats://localhost:4222"

        component_id = uuid.uuid4()
        event_plane2 = NatsEventPlane(server_url, component_id)
        await event_plane2.connect()

        await event_plane2.subscribe(callback_1, topic=topic)
        await event_plane2.subscribe(callback_2, topic=topic)
        await event_plane2.subscribe(callback_3, event_type=event_type)

        component_id = uuid.uuid4()
        event_plane1 = NatsEventPlane(server_url, component_id)
        await event_plane1.connect()

        ch1 = Topic(["test", "1"])
        ch2 = Topic(["test", "2"])
        await event_plane1.publish(event_type, ch1, payload)
        await event_plane1.publish(event_type, ch2, payload)

        # Allow time for message propagation
        await asyncio.sleep(2)

        assert len(results_1) == 2
        assert len(results_2) == 2
        assert len(results_3) == 2

        await event_plane1.disconnect()
        await event_plane2.disconnect()
