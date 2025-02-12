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


import signal
import contextlib
import fcntl
import logging
import os
import pathlib
from dataclasses import dataclass
import re
import select
import socket
import threading
import typing
import logging
import json
import sys
import tempfile
from typing import List, Optional, Set, Tuple, Any
import subprocess


# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(filename)s: %(levelname)s: %(funcName)s(): %(lineno)d:\t%(message)s",
# )

logger = logging.getLogger(__name__)



def _stream_reader(pipe, prefix=""):
    """Read lines from a pipe and print them immediately."""
    for line in iter(pipe.readline, ''):  # iterates until EOF
        logger.error("%s %s", prefix ,line)
    pipe.close()


def _read_outputs(_process, _logger, _outputs):
    # Set stdout and stderr file descriptors to non-blocking mode
    try:
        fcntl.fcntl(_process.stdout, fcntl.F_SETFL, os.O_NONBLOCK)
        fcntl.fcntl(_process.stderr, fcntl.F_SETFL, os.O_NONBLOCK)
    except ValueError:  # when selecting on closed files
        return

    buffers = {_process.stdout: "", _process.stderr: ""}
    rds = [_process.stdout, _process.stderr]
    while rds:
        try:
            readable, _, _ = select.select(rds, [], [], 1)
        except ValueError:  # when selecting on closed files
            break

        for rd in readable:
            try:
                data = os.read(rd.fileno(), 4096)
                if not data:
                    rds.remove(rd)
                    continue

                decoded_data = data.decode("utf-8")
                buffers[rd] += decoded_data
                lines = buffers[rd].splitlines(keepends=True)

                if buffers[rd].endswith("\n"):
                    complete_lines = lines
                    buffers[rd] = ""
                else:
                    complete_lines = lines[:-1]
                    buffers[rd] = lines[-1]

                for line in complete_lines:
                    line = line.rstrip()
                    _logger.info(line)
                    _outputs.append(line)
            except OSError:  # Reading from an empty non-blocking file
                pass


class ScriptThread(threading.Thread):
    """A class that runs external script in a separate thread."""

    def __init__(self, cmd, workdir=None, group=None, target=None, name=None, args=(), kwargs=None) -> None:
        """Initializes the ScriptThread object."""
        super().__init__(group, target, name, args, kwargs, daemon=True)
        self.cmd = cmd
        self.workdir = workdir
        self._process_spawned_or_spawn_error_flag: Optional[threading.Event] = None
        self.active = False
        self._process = None
        self.returncode = None
        self._output: List[str] = []
        self._logger = logging.getLogger(self.name)

    def __enter__(self):
        """Starts the script thread."""
        self.start(threading.Event())
        if self._process_spawned_or_spawn_error_flag is not None:
            self._process_spawned_or_spawn_error_flag.wait()
        else:
            raise RuntimeError("Missing wait flag")
        return self

    def __exit__(self, *args):
        """Stops the script thread and waits for it to join."""
        self.stop()
        self.join()
        self._process_spawned_or_spawn_error_flag = None

    def start(self, flag: typing.Optional[threading.Event] = None) -> None:
        """Starts the script thread."""
        if flag is None:
            flag = threading.Event()
        self._logger.info(f"Starting {self.name} script with \"{' '.join(self.cmd)}\" cmd")
        self._process_spawned_or_spawn_error_flag = flag
        super().start()

    def stop(self):
        """Sets the active flag to False to stop the script thread."""
        self._logger.info(f"Stopping {self.name} script")
        self.active = False

    def run(self):
        """Runs the script in a separate process."""
        import psutil

        self.returncode = None
        self._output = []
        self._process = None

        os.environ.setdefault("PYTHONUNBUFFERED", "1")  # to not buffer logs
        try:
            with psutil.Popen(
                self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, cwd=self.workdir
            ) as process:
                self._process = process
                self.active = True
                if self._process_spawned_or_spawn_error_flag:
                    self._process_spawned_or_spawn_error_flag.set()
                while self.active and process.poll() is None and process.returncode is None:
                    try:
                        _read_outputs(process, self._logger, self._output)
                    except KeyboardInterrupt:
                        self.stop()

        finally:
            if self._process_spawned_or_spawn_error_flag:
                self._process_spawned_or_spawn_error_flag.set()
            if self.process:
                while self.process.poll() is None:
                    _read_outputs(self.process, self._logger, self._output)
                _read_outputs(self.process, self._logger, self._output)
                self.returncode = process.wait()  # pytype: disable=name-error
                self._logger.info(f"{self.name} process finished with {self.returncode}")

            self.active = False
            self._process = None

    @property
    def output(self):
        """Return process stream output."""
        return "\n".join(self._output)

    @property
    def process(self):
        """Return process object."""
        return self._process

@dataclass
class WorkerRecord():
    name: str
    command: List[str]
    needs_kill_to_stop: bool
    json_file: str
    tag: str


def gather_published_events(workers: List[WorkerRecord], tag: Optional[str] = None) -> List[dict[str, Any]]:
    all_events_list = []
    for worker in workers:
        if tag is None or worker.tag == tag:
            log_file = worker.json_file
            logger.info(f"Reading events from: {log_file}")
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        json_line = json.loads(line)
                        json_line["log_file"] = log_file
                        all_events_list.append(json_line)
            except FileNotFoundError:
                logger.info(f"File {log_file} not found")
                continue
    return all_events_list


def made_teporary_file(file_type, id):
    json_file = tempfile.NamedTemporaryFile(
        delete=False, mode="w", prefix=f"{file_type}_{id + 1}_", suffix=".json"
    )
    return json_file.name


def prepare_subscribers(
    subscriber_count: int,
    event_type: Optional[str] = None,
) -> List[WorkerRecord]:
    subscriber_workers: List[WorkerRecord] = []
    for i in range(subscriber_count):
        json_file = made_teporary_file("subscriber", i)
        subscriber_args = [
            sys.executable,
            "-u",
            "./icp/examples/python/event_plane/subscriber.py",
            "--subscriber-id",
            str(i + 1),
            "--event-topic",
            "publisher",
            "--save-events-path",
            json_file,
        ]
        if event_type:
            subscriber_args.extend(["--event-type", event_type])
        subscriber_workers.append(WorkerRecord(name=f"subscriber_{i}", command=subscriber_args, needs_kill_to_stop=True, json_file=json_file, tag="subscriber"))
    return subscriber_workers
SCRIPT_WAIT_TIME = 6

def prepare_publishers(
    publisher_count: int,
    event_type: Optional[str] = None,
) -> List[WorkerRecord]:
    publisher_workers: List[WorkerRecord] = []
    for i in range(publisher_count):
        json_file = made_teporary_file("publisher", i)
        publisher_args = [
            sys.executable,
            "-u",
            "./icp/examples/python/event_plane/publisher.py",
            "--publisher-id",
            str(i + 1),
            "--save-events-path",
            json_file,
        ]
        if event_type:
            publisher_args.extend(["--event-type", event_type])
        publisher_workers.append(WorkerRecord(name=f"publisher_{i}", command=publisher_args, needs_kill_to_stop=False, json_file=json_file, tag="publisher"))
    return publisher_workers

def run_workers_recursively(workers: List[WorkerRecord], all_workers, threads):
    worker = workers[0]
    logger.info(f"Starting {worker.name} worker")
    with ScriptThread(worker.command, name=worker.name) as script_thread:
        threads.append(script_thread)
        if len(workers) > 1:
            return run_workers_recursively(workers[1:], all_workers, threads)
        else:
            logger.info(f"Workers length: {len(all_workers)} Threads length: {len(threads)}")
            assert len(all_workers) == len(threads)
            # Wait for workers, which don't need kill to stop work
            for worker, thread in zip(all_workers, threads):
                logger.info(f"Checking if worker needs kill for {worker} worker to finish")
                if not worker.needs_kill_to_stop:
                    logger.info(f"Waiting for {worker} worker to finish")
                    thread.join(timeout=SCRIPT_WAIT_TIME)
            # Send SIGTERM to workers, which needs kill to stop
            logger.info("Sending SIGINT to workers")
            for worker, thread in zip(all_workers, threads):
                if worker.needs_kill_to_stop:
                    logger.info(f"Sending SIGINT to {worker.name} worker process {thread.process.pid}")
                    thread.process.send_signal(signal.SIGTERM)
                    logger.info(f"Waiting for {worker.name} worker to finish")
                    thread.join(timeout=SCRIPT_WAIT_TIME)
            logger.info("All workers finished")
            return zip(all_workers, threads)

def run_workers(workers: List[WorkerRecord]):
    return run_workers_recursively(workers, workers, [])
    
