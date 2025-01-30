import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an example of the TensorRT-LLM pipeline."
    )

    example_dir = Path(__file__).parent.absolute().parent.absolute()

    default_operator_repository = example_dir.joinpath("operators")

    default_log_dir = ""

    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(default_log_dir),
        help="log dir folder",
    )

    parser.add_argument(
        "--initialize-request-plane",
        default=False,
        action="store_true",
        help="Initialize the request plane, should only be done once per deployment",
    )

    parser.add_argument(
        "--log-level", type=int, default=1, help="log level applied to all workers"
    )

    parser.add_argument(
        "--request-plane-uri",
        type=str,
        default="nats://localhost:4223",
        help="URI of request plane",
    )

    parser.add_argument(
        "--starting-metrics-port",
        type=int,
        default=50000,
        help="Metrics port for first worker. Each worker will expose metrics on subsequent ports, ex. worker 1: 50000, worker 2: 50001, worker 3: 50002",
    )

    parser.add_argument(
        "--context-worker-count", type=int, default=0, help="Number of context workers"
    )

    parser.add_argument(
        "--generate-worker-count",
        type=int,
        default=0,
        help="Number of generate workers",
    )

    parser.add_argument(
        "--aggregate-worker-count",
        type=int,
        required=False,
        default=0,
        help="Number of baseline workers",
    )

    parser.add_argument(
        "--operator-repository",
        type=str,
        default=str(default_operator_repository),
        help="Operator repository",
    )

    parser.add_argument(
        "--worker-name",
        type=str,
        required=False,
        default="llama",
        help="Name of the worker",
    )

    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
        help="Ignore EOS token when generating",
    )

    return parser.parse_args()
