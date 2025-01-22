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

    # API Server
    parser.add_argument(
        "--api-server-host",
        type=str,
        required=False,
        default="127.0.0.1",
        help="API Server host",
    )

    parser.add_argument(
        "--api-server-port",
        type=int,
        required=False,
        default=8000,
        help="API Server port",
    )

    # Misc
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Tokenizer to use for chat template in chat completions API",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=False,
        default="prefill",
        help="Model name",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        required=False,
        default="info",
        help="Logging level (e.g., debug, info, warning, error, critical)",
    )

    parser.add_argument(
        "--log-format",
        type=str,
        required=False,
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        help="Logging format",
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
        implementation="api_server_open_ai.operators.api_server_operator:ApiServerOperator",
        parameters={
            "api_server_host": args.api_server_host,
            "api_server_port": args.api_server_port,
            "tokenizer": args.tokenizer,
            "model_name": args.model_name,
        },
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
