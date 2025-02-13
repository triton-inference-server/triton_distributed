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
    event_plane = NatsEventPlane(server_url, args.component_id)

    await event_plane.connect()

    try:
        event_topic = (
            EventTopic(args.event_topic.split(".")) if args.event_topic else None
        )

        event = args.payload.encode()
        await event_plane.publish(event, args.event_type, event_topic)
        print(f"Published event from publisher {args.event_topic}")
    finally:
        await event_plane.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event publisher script")
    parser.add_argument(
        "--component-id",
        type=uuid.UUID,
        default=uuid.uuid4(),
        help="Component ID (UUID)",
    )
    parser.add_argument(
        "--event-topic",
        type=str,
        default=None,
        help="Event EventTopic to subscribe to (comma-separated for multiple levels)",
    )
    parser.add_argument(
        "--event-type", type=str, default="test_event", help="Event type"
    )
    parser.add_argument(
        "--payload",
        type=str,
        default="test_payload",
        help="Payload to be published with event.",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
