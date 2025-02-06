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

from triton_distributed.icp.nats_event_plane import (
    DEFAULT_EVENTS_URI,
    EventTopic,
    NatsEventPlane,
)


async def single_publisher_subscriber_example():
    # async with aclosing(event_plane()) as event_plane_instance:
    # event_plane_instance = await anext(event_plane)

    server_url = DEFAULT_EVENTS_URI
    component_id = str(uuid.uuid4())
    plane = NatsEventPlane(server_url, component_id)

    await plane.connect()
    received_events = []

    async def callback(event):
        print(event)
        received_events.append(event)

    event_topic = EventTopic(["test", "event_topic"])
    event_type = "test_event"
    event = b"my_payload"

    await plane.subscribe(callback, event_topic=event_topic, event_type=event_type)

    await plane.publish(event, event_type, event_topic)

    # Allow time for message to propagate
    await asyncio.sleep(3)

    print(f"received_events: {received_events}")
    # assert received_events[0][0].event_id == event.event_id

    await plane.disconnect()


if __name__ == "__main__":
    asyncio.run(single_publisher_subscriber_example())
