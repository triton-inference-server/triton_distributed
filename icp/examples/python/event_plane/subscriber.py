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
import uuid
import json
import sys
import signal

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
    event_plane = NatsEventPlane(server_url, uuid.uuid4())

    async def callback(event):
        logger.info(
            f"Subscriber {args.subscriber_id} received event: {event.event_id} event payload: {event.payload}"
        )
        # Serialize received events to JSON
        if args.save_events_path:
            with open(args.save_events_path, "a+") as json_file:
                event_obj = {
                        "event_payload": event.payload.tobytes().decode("utf-8"),
                        "event_id": str(event.event_id), 
                        "event_topic": str(event_topic),
                        "event_type": f"{args.event_type or 'all'}",
                        "component_id": str(args.component_id)}
                json_file.write(json.dumps(event_obj))
                json_file.write("\n")

    await event_plane.connect()

    try:
        event_topic = EventTopic(args.event_topic.split(".")) if args.event_topic else None
        logger.info(f"Subscribing to event_topic: {args.event_topic}")
        await event_plane.subscribe(
            callback,
            event_topic=event_topic,
            event_type=args.event_type,
            component_id=args.component_id,
        )
        logger.info(
            f"Subscriber {args.subscriber_id} is listening on event_topic {event_topic} with event type '{args.event_type or 'all'}' "
            + f"component ID '{args.component_id}'"
        )

        # Handle signals to stop the subscriber

        def handler(signum, frame):
            logger.info(f"Signal detected: {signum} stopping subscriber")
            sys.exit(0)


        signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
        for sig in signals:
            try:
                signal.signal(sig, handler)
            except Exception:
                pass



        while True:
            await asyncio.sleep(5)  # Keep the subscriber running
            logger.info(f"Subscriber {args.subscriber_id} is still running")
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
    parser.add_argument(
        "--save-events-path",
        type=str,
        default=None,
        help="Path to save received events as JSON",
    )

    args = parser.parse_args()

    asyncio.run(main(args))
