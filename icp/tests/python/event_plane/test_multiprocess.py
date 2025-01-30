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
from python.event_plane.publisher_subscriber_utils import (
    check_recieved_events,
    gather_published_events,
    run_publishers,
    run_subscribers,
)


@pytest.mark.asyncio
class TestEventPlaneMultiProcess:
    @pytest.mark.asyncio
    async def test_single_publisher_subscriber(self, nats_server):
        pub_sub_count = [(2, 2), (1, 1), (1, 2), (2, 1), (2, 2)]
        for publishers, subscribers in pub_sub_count:
            await self.run_test_case(publishers, subscribers)

    async def run_test_case(self, publisher_count: int, subscriber_count: int):
        processes: List[subprocess.Popen] = []

        try:
            # Start subscribers
            subscriber_logs = run_subscribers(processes, subscriber_count)
            time.sleep(0.5)

            # Start publishers
            publisher_logs = run_publishers(processes, publisher_count)

            print(
                f"Running test case with {publisher_count} publisher(s) and {subscriber_count} subscriber(s)."
            )

            time.sleep(0.5)

            # Verify logs
            all_events = gather_published_events(publisher_logs)
            check_recieved_events(all_events, subscriber_logs)
            print("Test case passed!")
        finally:
            # Terminate all processes
            for proc in processes:
                proc.terminate()
            for proc in processes:
                proc.wait()

            # Clean up log files
            for log_file in publisher_logs + subscriber_logs:
                os.remove(log_file)
