import asyncio
import uuid
from typing import List

import pytest

from triton_distributed.icp.eventplane import Event, EventTopic
from triton_distributed.icp.eventplane_nats import EventPlaneNats


@pytest.mark.asyncio
class TestEventPlaneFunctional:
    @pytest.mark.asyncio
    async def test_single_publisher_subscriber(self, nats_server, event_plane):
        print(f"Print loop test: {id(asyncio.get_running_loop())}")

        received_events: List[Event] = []

        async def callback(event: Event):
            received_events.append(event)

        event_topic = EventTopic("test.event_topic")
        event_type = "test_event"
        payload = b"test_payload"

        await event_plane.subscribe(
            callback, event_topic=event_topic, event_type=event_type
        )

        event = await event_plane.create_event(event_type, event_topic, payload)
        await event_plane.publish(event)

        # Allow time for message to propagate
        await asyncio.sleep(2)

        assert len(received_events) == 1
        assert received_events[0].event_id == event.event_id

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
        payload = b"multi_payload"

        # async with event_plane_context() as event_plane1:
        server_url = "nats://localhost:4222"

        component_id = uuid.uuid4()
        event_plane2 = EventPlaneNats(server_url, component_id)
        await event_plane2.connect()

        await event_plane2.subscribe(callback_1, event_topic=event_topic)
        await event_plane2.subscribe(callback_2, event_topic=event_topic)
        await event_plane2.subscribe(callback_3, event_type=event_type)

        component_id = uuid.uuid4()
        event_plane1 = EventPlaneNats(server_url, component_id)
        await event_plane1.connect()

        ch1 = EventTopic(["test", "1"])
        ch2 = EventTopic(["test", "2"])
        event1 = await event_plane1.create_event(event_type, ch1, payload)
        await event_plane1.publish(event1)
        event2 = await event_plane1.create_event(event_type, ch2, payload)
        await event_plane1.publish(event2)

        # Allow time for message propagation
        await asyncio.sleep(2)

        assert len(results_1) == 2
        assert len(results_2) == 2
        assert len(results_3) == 2

        await event_plane1.disconnect()
        await event_plane2.disconnect()
