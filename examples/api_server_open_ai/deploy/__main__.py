import argparse
import asyncio
import shutil
import signal
import sys
from pathlib import Path

from triton_distributed.worker import Deployment, OperatorConfig, WorkerConfig


def parse_args(args=None):
    example_dir = Path(__file__).parent.absolute().parent.absolute()

    default_log_dir = example_dir.joinpath("logs")

    parser = argparse.ArgumentParser(description="Hello World Deployment")

    parser.add_argument(
        "--initialize-request-plane",
        default=False,
        action="store_true",
        help="Initialize the request plane, should only be done once per deployment",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(default_log_dir),
        help="log dir folder",
    )

    parser.add_argument(
        "--clear-logs", default=False, action="store_true", help="clear log dir"
    )

    parser.add_argument("--log-level", type=int, default=1)

    parser.add_argument(
        "--request-plane-uri", type=str, default="nats://localhost:4223"
    )

    args = parser.parse_args(args)

    return args


deployment = None


def handler(signum, frame):
    exit_code = 0
    if deployment:
        print("Stopping Workers")
        exit_code = deployment.stop()
    print(f"Workers Stopped Exit Code {exit_code}")
    sys.exit(exit_code)


signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
for sig in signals:
    try:
        signal.signal(sig, handler)
    except Exception:
        pass


async def main(args):
    global deployment
    log_dir = Path(args.log_dir)

    if args.clear_logs:
        shutil.rmtree(log_dir)

    log_dir.mkdir(exist_ok=True)

    # define all your worker configs as before: encoder, decoder, etc.
    api_server_op = OperatorConfig(
        name="api_server",
        implementation="ApiServerOperator",  # matches the .py file's operator class
        max_inflight_requests=1,
    )

    api_server = WorkerConfig(operators=[api_server_op], name="api_server")

    deployment = Deployment(
        [
            (api_server, 1),
        ],
        initialize_request_plane=True,
        log_dir=args.log_dir,
        log_level=args.log_level,
    )
    deployment.start()
    while True:
        await asyncio.sleep(10)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
