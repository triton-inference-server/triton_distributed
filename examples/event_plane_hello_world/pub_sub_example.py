import asyncio
import uuid

from triton_distributed.icp.eventplane import EventTopic
from triton_distributed.icp.eventplane_nats import EventPlaneNats


async def test_single_publisher_subscriber():
    # async with aclosing(event_plane()) as event_plane_instance:
    # event_plane_instance = await anext(event_plane)

    server_url = "nats://localhost:4223"
    component_id = uuid.uuid4()
    plane = EventPlaneNats(server_url, component_id)

    await plane.connect()
    received_events = []

    async def callback(event):
        print(event)
        received_events.append(event)

    channel = EventTopic("test.channel")
    event_type = "test_event"
    payload = b"test_payload"

    await plane.subscribe(callback, channel=channel, event_type=event_type)

    event = await plane.create_event(event_type, channel, payload)
    await plane.publish(event)

    # Allow time for message to propagate
    await asyncio.sleep(3)

    print(f"received_events: {received_events}")
    # assert received_events[0][0].event_id == event.event_id

    await plane.disconnect()


if __name__ == "__main__":
    asyncio.run(test_single_publisher_subscriber())
