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
        "--log-level", type=int, default=3, help="log level applied to all workers"
    )

    parser.add_argument(
        "--request-plane-uri",
        type=str,
        default="nats://localhost:4222",
        help="URI of request plane",
    )

    parser.add_argument(
        "--nats-port",
        type=int,
        default=4222,
        help="Port for NATS server",
    )

    parser.add_argument(
        "--starting-metrics-port",
        type=int,
        default=50000,
        help="Metrics port",
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
        "--model",
        type=str,
        required=False,
        default="llama-3.1-8b-instruct",
        choices=[
            "mock",
            "llama-3.1-70b-instruct",
            "llama-3.1-8b-instruct",
            "llama-3-8b-instruct-generate",
            "llama-3-8b-instruct-context",
            "llama-3-8b-instruct",
            "llama-3-8b-instruct-default",
            "llama-3-70b-instruct-context",
            "llama-3-70b-instruct-generate",
            "llama-3-70b-instruct",
        ],
        help="model to serve",
    )

    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
        help="Ignore EOS token when generating",
    )

    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
        help="Dry run the command",
    )

    parser.add_argument(
        "--disaggregated-serving",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
        help="Enable disaggregated serving",
    )

    return parser.parse_args()
