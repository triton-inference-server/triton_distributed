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


import os
import subprocess
import time
from typing import List

import pytest

from icp.tests.python.event_plane.publisher_subscriber_utils import (
    check_recieved_events,
    gather_published_events,
    run_publishers,
    run_subscribers,
)

pytestmark = pytest.mark.pre_merge


@pytest.mark.asyncio
class TestEventPlaneSpecificSubscriptions:
    @pytest.mark.asyncio
    async def test_subscription_on_event_type(self, nats_server):
        processes: List[subprocess.Popen] = []
        subscriber_count = 2
        publisher_count = 2

        try:
            # Start subscribers
            subscriber_logs = run_subscribers(processes, subscriber_count)
            event_type2_subscriber_logs = run_subscribers(
                processes, 1, event_type="test_event2"
            )
            time.sleep(0.5)

            # Start publishers
            event_type2_publisher_logs = run_publishers(
                processes, publisher_count, event_type="test_event2"
            )
            event_type_publisher_logs = run_publishers(
                processes, publisher_count, event_type="test_event"
            )

            # Let the processes run for 10 seconds
            print(
                f"Running test case with {publisher_count} publisher(s) and {subscriber_count} subscriber(s)."
            )

            time.sleep(0.5)

            # Verify logs
            event_type2_events = gather_published_events(event_type2_publisher_logs)
            event_type_events = gather_published_events(event_type_publisher_logs)
            check_recieved_events(event_type2_events, event_type2_subscriber_logs)
            check_recieved_events(
                event_type_events.union(event_type2_events), subscriber_logs
            )
            print("Test case passed!")
        finally:
            # Terminate all processes
            for proc in processes:
                proc.terminate()
            for proc in processes:
                proc.wait()

            # Clean up log files
            for log_file in (
                event_type_publisher_logs
                + event_type2_publisher_logs
                + subscriber_logs
                + event_type2_subscriber_logs
            ):
                os.remove(log_file)
