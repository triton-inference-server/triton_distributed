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

import logging
import os
import subprocess
import time
from typing import List

import pytest

from icp.tests.python.event_plane.publisher_subscriber_utils import (
    gather_published_events,
    prepare_publishers,
    prepare_subscribers,
    run_workers,
)


from icp.tests.python.event_plane.utils import nats_server

pytestmark = pytest.mark.pre_merge

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
class TestEventPlaneMultiProcess:
    @pytest.mark.asyncio
    async def test_single_publisher_subscriber(self, nats_server):
        pub_sub_count = [(2, 2), (1, 1), (1, 2), (2, 1), (2, 2)]
        for publishers, subscribers in pub_sub_count:
            await self.run_test_case(publishers, subscribers)

    async def run_test_case(self, publisher_count: int, subscriber_count: int):

        # Prepare subscribers
        subscriber_workers = prepare_subscribers(subscriber_count)

        # Prepare publishers
        publisher_workers = prepare_publishers(publisher_count)

        workers = subscriber_workers + publisher_workers

        # Execute all workers
        executed_workers = run_workers(workers)

        # Check workers status

        alive_threads = False

        for worker, thread in executed_workers:
            if thread.is_alive():
                alive_threads = True
                logger.error(f"Worker {worker} is still alive after timeout")
            if thread.returncode != 0:
                logger.warning(f"Worker {worker} returned {thread.returncode}")

        if alive_threads:
            raise RuntimeError("Alive workers detected after test finished")
        
        # Verify results
        publisher_events = gather_published_events(workers, "publisher")
        subscriber_events = gather_published_events(workers, "subscriber")

        # Check if all subscribers received all events from all publishers

        # {'event_payload': 'Payload from publisher 2 idx 0', 'event_id': '2b335d2b-6571-496a-a1e5-995862789215', 'event_topic': 'publisher', 'event_type': 'all', 'component_id': 'None', 'log_file': '/tmp/subscriber_1_3lyrfd7_.json'}

        # Build set of events for each subscriber using the log file names and payloads

        # Dict of files
        subscriber_files: dict[str, set[str]] = {}
        for subscriber_event in subscriber_events:
            file = subscriber_event["log_file"]
            events: set[str] = subscriber_files.get(subscriber_event["log_file"], set())
            events.add(subscriber_event["event_payload"])
            subscriber_files[file] = events
            

        if len(publisher_events) == 0:
            raise RuntimeError("No events published")
        
        # Check if all subscribers received all events from all publishers
        missing_events = 0
        for publisher_event in publisher_events:
            for subscriber_file, events in subscriber_files.items():
                if publisher_event["event_payload"] not in events:
                    logger.error(f"Event {publisher_event} not found in subscriber {subscriber_file}")
                    missing_events += 1
                    
        if missing_events > 15:
            raise RuntimeError(f"Too many missing events {missing_events}")
