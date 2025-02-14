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
import dataclasses
import uuid
from typing import List

import pytest

from triton_distributed.icp import Event, EventTopic, NatsEventPlane

from .utils import event_plane, nats_server

pytestmark = pytest.mark.pre_merge


@pytest.mark.asyncio
class TestEventPlaneFunctional:
    @pytest.mark.asyncio
    async def test_single_publisher_subscriber(self, nats_server, event_plane):
        print(f"Print loop test: {id(asyncio.get_running_loop())}")

        received_events: List[Event] = []

        async def callback(event):
            received_events.append(event)
            print(event)

        event_topic = EventTopic(["test", "event_topic"])
        event_type = "test_event"
        event = b"test_payload"

        await event_plane.subscribe(
            callback, event_topic=event_topic, event_type=event_type
        )
        event_metadata = await event_plane.publish(event, event_type, event_topic)

        # Allow time for message to propagate
        await asyncio.sleep(2)

        assert len(received_events) == 1
        assert received_events[0].event_id == event_metadata.event_id

    @pytest.mark.asyncio
    async def test_single_publisher_subscriber_iterator(self, nats_server, event_plane):
        print(f"Print loop test: {id(asyncio.get_running_loop())}")

        received_events: List[Event] = []

        event_topic = EventTopic(["test", "event_topic"])
        event_type = "test_event"
        event = b"test_payload"

        subscription = await event_plane.subscribe(
            event_topic=event_topic, event_type=event_type
        )
        event_metadata = await event_plane.publish(
            event, event_topic=event_topic, event_type=event_type
        )

        # Allow time for message to propagate
        await asyncio.sleep(2)

        async for x in subscription:
            print(x.timestamp)
            print(x.event_id)
            print(x.event_type)
            print(x.event_topic)
            print(x.payload)
            received_events.append(x)
            break

        assert len(received_events) == 1
        assert received_events[0].event_id == event_metadata.event_id

    @pytest.mark.asyncio
    async def test_default_subscription(self, nats_server, event_plane):
        print(f"Print loop test: {id(asyncio.get_running_loop())}")

        received_events: List[Event] = []

        event = b"test_payload"

        subscription = await event_plane.subscribe()
        event_metadata = await event_plane.publish(
            event,
        )

        # Allow time for message to propagate
        await asyncio.sleep(2)

        async for x in subscription:
            print(x.timestamp)
            print(x.event_id)
            print(x.event_type)
            print(x.event_topic)
            print(x.payload)
            received_events.append(x)
            break

        assert len(received_events) == 1
        assert received_events[0].event_id == event_metadata.event_id

    @pytest.mark.asyncio
    async def test_custom_type(self, nats_server, event_plane):
        print(f"Print loop test: {id(asyncio.get_running_loop())}")

        received_events: List[Event] = []

        @dataclasses.dataclass
        class MyEvent:
            test: str
            index: int

        event = MyEvent("hello", 0)

        subscription = await event_plane.subscribe()
        event_metadata = await event_plane.publish(
            event,
        )

        # Allow time for message to propagate
        await asyncio.sleep(2)

        async for x in subscription:
            print(x.timestamp)
            print(x.event_id)
            print(x.event_type)
            print(x.event_topic)
            print(x.payload)
            print(x.typed_payload(MyEvent))
            received_events.append(x)
            break

        assert len(received_events) == 1
        assert received_events[0].event_id == event_metadata.event_id
        assert isinstance(received_events[0].typed_payload(MyEvent), type(event))
        assert isinstance(received_events[0].typed_payload(dict), dict)

    @pytest.mark.asyncio
    async def test_one_publisher_multiple_subscribers(self, nats_server):
        results_1: List[Event] = []
        results_2: List[Event] = []
        results_3: List[Event] = []

        async def callback_1(event):
            results_1.append(event)

        async def callback_2(event):
            results_2.append(event)

        async def callback_3(event):
            results_3.append(event)

        event_topic = EventTopic(["test"])
        event_type = "multi_event"
        event = b"multi_payload"

        # async with event_plane_context() as event_plane1:
        server_url = "tls://localhost:4222"

        component_id = uuid.uuid4()
        event_plane2 = NatsEventPlane(server_url, component_id)
        try:
            await event_plane2.connect()

            try:
                subscription1 = await event_plane2.subscribe(
                    callback_1, event_topic=event_topic
                )
                try:
                    subscription2 = await event_plane2.subscribe(
                        callback_2, event_topic=event_topic
                    )
                    try:
                        subscription3 = await event_plane2.subscribe(
                            callback_3, event_type=event_type
                        )

                        component_id = uuid.uuid4()
                        event_plane1 = NatsEventPlane(server_url, component_id)
                        try:
                            await event_plane1.connect()

                            ch1 = EventTopic(["test", "1"])
                            ch2 = EventTopic(["test", "2"])
                            await event_plane1.publish(event, event_type, ch1)
                            await event_plane1.publish(event, event_type, ch2)

                            # Allow time for message propagation
                            await asyncio.sleep(2)

                            assert len(results_1) == 2
                            assert len(results_2) == 2
                            assert len(results_3) == 2
                        finally:
                            await event_plane1.disconnect()
                    finally:
                        await subscription3.unsubscribe()
                finally:
                    await subscription2.unsubscribe()
            finally:
                await subscription1.unsubscribe()

        finally:
            await event_plane2.disconnect()

    @pytest.mark.asyncio
    async def test_context_manager(self, nats_server):
        """Test that context managers properly handle connection/disconnection and subscription/unsubscription."""
        received_events: List[Event] = []
        event_topic = EventTopic(["test", "event_topic"])
        event_type = "test_event"
        event = b"test_payload"

        # Test successful operation with context managers
        async with NatsEventPlane() as plane:
            assert plane.is_connected()

            async def callback(event):
                received_events.append(event)

            async with await plane.subscribe(
                callback, event_topic=event_topic, event_type=event_type
            ) as subscription:
                assert subscription._nc_sub is not None
                event_metadata = await plane.publish(event, event_type, event_topic)
                await asyncio.sleep(2)  # Allow time for message to propagate

            # After subscription context, should be unsubscribed
            assert subscription._nc_sub is None

        # After plane context, should be disconnected
        assert not plane.is_connected()
        assert len(received_events) == 1
        assert received_events[0].event_id == event_metadata.event_id

        # Test error handling in context managers
        with pytest.raises(RuntimeError):
            async with NatsEventPlane() as plane:
                async with await plane.subscribe(
                    callback, event_topic=event_topic, event_type=event_type
                ):
                    raise RuntimeError("Test error")
                # Should not reach here
                pytest.fail("Should have raised exception")
            # Should not reach here
            pytest.fail("Should have raised exception")

        # Even after error, resources should be cleaned up
        assert not plane.is_connected()
