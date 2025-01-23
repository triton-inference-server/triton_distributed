import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an example of the VLLM pipeline."
    )

    example_dir = Path(__file__).parent.absolute().parent.absolute()
    default_log_dir = example_dir.joinpath("logs")
    default_operator_repository = example_dir.joinpath("operators")

    parser = argparse.ArgumentParser(description="Hello World Deployment")

    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(default_log_dir),
        help="log dir folder",
    )

    parser.add_argument(
        "--context-worker-count",
        type=int,
        required=False,
        default=1,
        help="Number of context workers",
    )

    parser.add_argument(
        "--generate-worker-count",
        type=int,
        required=False,
        default=1,
        help="Number of generate workers",
    )

    parser.add_argument(
        "--nats-url",
        type=str,
        required=False,
        default="nats://localhost:4222",
        help="URL of NATS server",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=False,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model name",
    )

    parser.add_argument(
        "--worker-name",
        type=str,
        required=False,
        default="llama",
        help="Worker name",
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        required=False,
        default=None,
        help="Maximum input/output latency length.",
    )

    parser.add_argument(
        "--max-batch-size",
        type=int,
        required=False,
        default=10000,
        help="Max batch size",
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        required=False,
        default=0.45,
        help="GPU memory utilization (fraction of memory from 0.0 to 1.0)",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        required=False,
        default="float16",
        help="Attention data type (float16, TODO: fp8)",
    )

    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        required=False,
        default="auto",
        help="Key-value cache data type",
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

    ## Logical arguments for vLLM engine

    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
        help="Enable prefix caching",
    )

    parser.add_argument(
        "--enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
        help="Enable chunked prefill",
    )

    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
        help="Enforce eager execution",
    )

    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
        help="Ignore EOS token when generating",
    )

    parser.add_argument(
        "--baseline-tp-size",
        type=int,
        default=1,
        help="Tensor parallel siz of a baseline worker.",
    )

    parser.add_argument(
        "--context-tp-size",
        type=int,
        default=1,
        help="Tensor parallel size of a context worker.",
    )

    parser.add_argument(
        "--generate-tp-size",
        type=int,
        default=1,
        help="Tensor parallel size of a generate worker.",
    )

    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="maximum number of sequences per iteration",
    )

    parser.add_argument(
        "--disable-async-output-proc",
        action="store_true",
        help="Disable async output processing",
    )

    parser.add_argument(
        "--disable-log-stats",
        action="store_true",
        help="Disable logging statistics",
    )

    return parser.parse_args()
