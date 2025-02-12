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
import logging
import json
import uuid

from triton_distributed.icp.nats_event_plane import (
    DEFAULT_EVENTS_URI,
    EventTopic,
    NatsEventPlane,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(filename)s: %(levelname)s: %(funcName)s(): %(lineno)d:\t%(message)s",
)

logger = logging.getLogger(__name__)

async def main(args):
    server_url = DEFAULT_EVENTS_URI
    event_plane = NatsEventPlane(server_url, args.component_id)

    await event_plane.connect()

    try:
        event_topic = EventTopic(["publisher", str(args.publisher_id)])

        for i in range(args.event_count):
            event = f"Payload from publisher {args.publisher_id} idx {i}".encode()
            await event_plane.publish(event, 
                                      args.event_type, 
                                      event_topic)
            logger.info(f"Published event from publisher {args.publisher_id}")
            # Serialize sent events to JSON
            if args.save_events_path:
                with open(args.save_events_path, "a+") as json_file:
                    event_obj = {"event_payload": str(event.decode("utf-8")),
                            "event_id": str(args.publisher_id), 
                            "event_topic": str(event_topic),
                            "event_type": args.event_type,
                            "component_id": str(args.component_id)}
                    json_file.write(json.dumps(event_obj))
                    json_file.write("\n")
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
    parser.add_argument(
        "--save-events-path",
        type=str,
        default=None,
        help="Path to save received events as JSON",
    )

    args = parser.parse_args()
    asyncio.run(
        main(args)
    )
