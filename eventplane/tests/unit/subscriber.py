import argparse
import asyncio
import uuid

from triton_distributed.eventplane.eventplane import Channel
from triton_distributed.eventplane.eventplane_nats import EventPlaneNats


async def main(subscriber_id, channel, event_type, component_id):
    server_url = "nats://localhost:4222"
    event_plane = EventPlaneNats(server_url, uuid.uuid4())

    async def callback(event):
        print(f"Subscriber {subscriber_id} received event: {event.event_id} payload: {event.payload.decode()}")

    await event_plane.connect()

    try:
        channel = Channel(channel.split(".")) if channel else None
        print(f"Subscribing to channel: {channel}")
        await event_plane.subscribe(callback, channel=channel, event_type=event_type, component_id=component_id)
        print(f"Subscriber {subscriber_id} is listening on channel {channel} with event type '{event_type or 'all'}' " +
              f"component ID '{component_id}'")

        while True:
            await asyncio.sleep(5)  # Keep the subscriber running
            print(f"Subscriber {subscriber_id} is still running")
    finally:
        await event_plane.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event subscriber script")
    parser.add_argument("--subscriber_id", type=int, required=True, help="Subscriber ID")
    parser.add_argument("--channel", type=str, default=None,
                        help="Channel to subscribe to (comma-separated for multiple levels)")
    parser.add_argument("--event_type", type=str, default=None,
                        help="Event type to filter (default: None for all types)")
    parser.add_argument("--component_id", type=None, default=None, help="Component ID (UUID) for the subscriber")

    args = parser.parse_args()

    asyncio.run(main(args.subscriber_id, args.channel, args.event_type, args.component_id))
