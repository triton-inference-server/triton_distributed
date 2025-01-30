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

from triton_distributed.icp import NatsEventPlane, Topic


async def test_single_publisher_subscriber():
    # async with aclosing(event_plane()) as event_plane_instance:
    # event_plane_instance = await anext(event_plane)

    server_url = "nats://localhost:4223"
    component_id = uuid.uuid4()
    plane = NatsEventPlane(server_url, component_id)

    await plane.connect()
    received_events = []

    async def callback(event):
        print(event)
        received_events.append(event)

    topic = Topic("test.topic")
    event_type = "test_event"
    payload = b"my_payload"

    await plane.subscribe(callback, topic=topic, event_type=event_type)

    event = plane.create_event(event_type, topic, payload)
    await plane.publish(event)

    # Allow time for message to propagate
    await asyncio.sleep(3)

    print(f"received_events: {received_events}")
    # assert received_events[0][0].event_id == event.event_id

    await plane.disconnect()


if __name__ == "__main__":
    asyncio.run(test_single_publisher_subscriber())
