import os
import time

import pytest

from .publisher_subscriber_utils import *
from .utils import nats_server


@pytest.mark.asyncio
class TestEventPlaneMultiProcess:
    @pytest.mark.asyncio
    async def test_single_publisher_subscriber(self, nats_server):
        pub_sub_count = [(2, 2), (1, 1), (1, 2), (2, 1), (2, 2)]
        for publishers, subscribers in pub_sub_count:
            await self.run_test_case(publishers, subscribers)

    async def run_test_case(self, publisher_count, subscriber_count):
        processes = []

        try:

            # Start subscribers
            subscriber_logs = run_subscribers(processes, subscriber_count)
            time.sleep(0.5)

            # Start publishers
            publisher_logs = run_publishers(processes, publisher_count)

            print(f"Running test case with {publisher_count} publisher(s) and {subscriber_count} subscriber(s).")

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
