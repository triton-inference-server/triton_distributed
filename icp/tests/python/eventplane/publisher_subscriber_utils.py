import subprocess
import sys
import tempfile


def check_recieved_events(all_events, subscriber_logs):
    for log_file in subscriber_logs:
        with open(log_file, "r") as f:
            subscriber_events = set()
            for line in f:
                if "received event" in line:
                    parts = line.split()
                    event_id = parts[4]  # Extract event_id from log
                    subscriber_events.add(event_id)
            assert all_events == subscriber_events, (
                f"Subscriber did not receive all events. "
                f"Missing: {all_events - subscriber_events}"
            )


def gather_published_events(log_files):
    all_events = set()
    for log_file in log_files:
        publisher_events = set()
        with open(log_file, "r") as f:
            for line in f:
                if "Published event" in line:
                    parts = line.split()
                    event_id = parts[5]  # Extract event_id from log
                    publisher_events.add(event_id)
        assert publisher_events, "No events were published"
        all_events = all_events.union(publisher_events)
    return all_events


def run_publishers(processes, publisher_count, event_type=None):
    publisher_logs = []
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


def run_subscribers(processes, subscriber_count, event_type=None):
    subscriber_logs = []
    for i in range(subscriber_count):
        log_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w", prefix=f"subscriber_{i + 1}_", suffix=".log"
        )
        subscriber_args = [
            sys.executable,
            "subscriber.py",
            "--subscriber_id",
            str(i + 1),
            "--channel",
            "publisher",
        ]
        if event_type:
            subscriber_args.extend(["--event_type", event_type])
        proc = subprocess.Popen(subscriber_args, stdout=log_file, stderr=log_file)
        processes.append(proc)
        subscriber_logs.append(log_file.name)
    return subscriber_logs
