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


import argparse
import asyncio
import uuid

from triton_distributed.icp.nats_event_plane import (
    DEFAULT_EVENTS_URI,
    EventTopic,
    NatsEventPlane,
)


async def main(args):
    server_url = DEFAULT_EVENTS_URI
    event_plane = NatsEventPlane(server_url, uuid.uuid4())

    async def callback(received_event):
        print(
            f"""
            Subscriber {args.subscriber_id}
            received event: {received_event.event_id}
            event payload: {received_event.payload.tobytes().decode("utf-8")}
            event.topic: {received_event.event_topic}
            event.type: {received_event.event_type}
            event.component_id: {received_event.component_id}
            event.timestamp: {received_event.timestamp}
            """
        )

    await event_plane.connect()

    try:
        event_topic = (
            EventTopic(args.event_topic.split(".")) if args.event_topic else None
        )
        print(f"Subscribing to event_topic: {args.event_topic}")
        await event_plane.subscribe(
            callback,
            event_topic=event_topic,
            event_type=args.event_type,
            component_id=args.component_id,
        )
        print(
            f"Subscriber {args.subscriber_id} is listening on event_topic {event_topic} with event type '{args.event_type or 'all'}' "
            + f"component ID '{args.component_id}'"
        )

        while True:
            await asyncio.sleep(5)  # Keep the subscriber running
            print(f"Subscriber {args.subscriber_id} is still running")
    finally:
        await event_plane.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event subscriber script")
    parser.add_argument(
        "--subscriber-id", type=int, required=True, help="Subscriber ID"
    )
    parser.add_argument(
        "--event-topic",
        type=str,
        default=None,
        help="Event EventTopic to subscribe to (comma-separated for multiple levels)",
    )
    parser.add_argument(
        "--event-type",
        type=str,
        default=None,
        help="Event type to filter (default: None for all types)",
    )
    parser.add_argument(
        "--component-id",
        type=uuid.UUID,
        default=None,
        help="Component ID (UUID) for the subscriber",
    )

    args = parser.parse_args()

    asyncio.run(main(args))
