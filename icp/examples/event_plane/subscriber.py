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

from triton_distributed.icp import NatsEventPlane, Topic


async def main(subscriber_id, topic, event_type, component_id):
    server_url = "nats://localhost:4222"
    event_plane = NatsEventPlane(server_url, uuid.uuid4())

    async def callback(event):
        print(
            f"Subscriber {subscriber_id} received event: {event.event_id} payload: {event.payload.decode()}"
        )

    await event_plane.connect()

    try:
        topic = Topic(topic.split(".")) if topic else None
        print(f"Subscribing to topic: {topic}")
        await event_plane.subscribe(
            callback,
            topic=topic,
            event_type=event_type,
            component_id=component_id,
        )
        print(
            f"Subscriber {subscriber_id} is listening on topic {topic} with event type '{event_type or 'all'}' "
            + f"component ID '{component_id}'"
        )

        while True:
            await asyncio.sleep(5)  # Keep the subscriber running
            print(f"Subscriber {subscriber_id} is still running")
    finally:
        await event_plane.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event subscriber script")
    parser.add_argument(
        "--subscriber_id", type=int, required=True, help="Subscriber ID"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Event Topic to subscribe to (comma-separated for multiple levels)",
    )
    parser.add_argument(
        "--event_type",
        type=str,
        default=None,
        help="Event type to filter (default: None for all types)",
    )
    parser.add_argument(
        "--component_id",
        type=uuid.UUID,
        default=None,
        help="Component ID (UUID) for the subscriber",
    )

    args = parser.parse_args()

    asyncio.run(
        main(args.subscriber_id, args.topic, args.event_type, args.component_id)
    )
