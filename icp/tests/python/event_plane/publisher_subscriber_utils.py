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


import subprocess
import sys
import tempfile
from typing import List, Optional, Set


def check_recieved_events(all_events: Set[str], subscriber_logs: List[str]):
    for log_file in subscriber_logs:
        with open(log_file, "r") as f:
            subscriber_events: Set[str] = set()
            for line in f:
                if "received event" in line:
                    parts = line.split()
                    event_id = parts[4]  # Extract event_id from log
                    subscriber_events.add(event_id)
            assert all_events == subscriber_events, (
                f"Subscriber did not receive all events. "
                f"Missing: {all_events - subscriber_events}"
            )


def gather_published_events(log_files: List[str]) -> Set[str]:
    all_events: Set[str] = set()
    for log_file in log_files:
        publisher_events: Set[str] = set()
        with open(log_file, "r") as f:
            for line in f:
                if "Published event" in line:
                    parts = line.split()
                    event_id = parts[5]  # Extract event_id from log
                    publisher_events.add(event_id)
        assert publisher_events, "No events were published"
        all_events = all_events.union(publisher_events)
    return all_events


def run_publishers(
    processes: List[subprocess.Popen],
    publisher_count: int,
    event_type: Optional[str] = None,
) -> List[str]:
    publisher_logs: List[str] = []
    for i in range(publisher_count):
        log_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w", prefix=f"publisher_{i + 1}_", suffix=".log"
        )
        publisher_args = [sys.executable, "publisher.py", "--publisher_id", str(i + 1)]
        if event_type:
            publisher_args.extend(["--event_type", event_type])
        proc = subprocess.Popen(publisher_args, stdout=log_file, stderr=log_file)
        processes.append(proc)
        publisher_logs.append(log_file.name)
    return publisher_logs


def run_subscribers(
    processes: List[subprocess.Popen],
    subscriber_count: int,
    event_type: Optional[str] = None,
) -> List[str]:
    subscriber_logs: List[str] = []
    for i in range(subscriber_count):
        log_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w", prefix=f"subscriber_{i + 1}_", suffix=".log"
        )
        subscriber_args = [
            sys.executable,
            "subscriber.py",
            "--subscriber_id",
            str(i + 1),
            "--topic",
            "publisher",
        ]
        if event_type:
            subscriber_args.extend(["--event_type", event_type])
        proc = subprocess.Popen(subscriber_args, stdout=log_file, stderr=log_file)
        processes.append(proc)
        subscriber_logs.append(log_file.name)
    return subscriber_logs
