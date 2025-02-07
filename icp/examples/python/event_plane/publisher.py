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


async def main(component_id, event_type, publisher_id, event_count):
    server_url = DEFAULT_EVENTS_URI
    event_plane = NatsEventPlane(server_url, component_id)

    await event_plane.connect()

    try:
        event_topic = EventTopic(["publisher", str(publisher_id)])

        for i in range(event_count):
            event = f"Payload from publisher {publisher_id}".encode()
            await event_plane.publish(event, event_type, event_topic)
            print(f"Published event from publisher {publisher_id}")
            await asyncio.sleep(0.01)
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
        "--event-type", type=str, default="test_event", help="Event type"
    )
    parser.add_argument("--publisher-id", type=int, required=True, help="Publisher ID")
    parser.add_argument(
        "--event-count", type=int, default=10, help="Event count to be published."
    )

    args = parser.parse_args()
    asyncio.run(
        main(args.component_id, args.event_type, args.publisher_id, args.event_count)
    )
