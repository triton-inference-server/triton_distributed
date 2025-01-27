import argparse
import asyncio
import uuid

from triton_distributed.eventplane.eventplane import Channel
from triton_distributed.eventplane.eventplane_nats import EventPlaneNats


async def main(component_id, event_type, publisher_id):
    server_url = "nats://localhost:4222"
    event_plane = EventPlaneNats(server_url, component_id)

    await event_plane.connect()

    try:
        channel = Channel(["publisher", str(publisher_id)])

        for i in range(10):
            payload = f"Payload from publisher {publisher_id}".encode()
            event = await event_plane.create_event(event_type, channel, payload)
            await event_plane.publish(event)
            print(f"Published event from publisher {publisher_id}: {event.event_id}")
            await asyncio.sleep(0.01)
    finally:
        await event_plane.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event publisher script")
    parser.add_argument("--component_id", type=uuid.UUID, default=uuid.uuid4(), help="Component ID (UUID)")
    parser.add_argument("--event_type", type=str, default="test_event", help="Event type")
    parser.add_argument("--publisher_id", type=int, required=True, help="Publisher ID")

    args = parser.parse_args()
    asyncio.run(main(args.component_id, args.event_type, args.publisher_id))
