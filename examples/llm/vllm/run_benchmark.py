#!/usr/bin/env python3

import argparse
import logging
import os
import subprocess
from datetime import datetime

LOGGER = logging.getLogger(__name__)


# Function to run the benchmark
def run_benchmark(
    input_tokens_cached,
    input_tokens_uncached,
    output_tokens,
    model,
    tokenizer,
    url,
    artifact_dir,
    request_count,
    load_type,
    load_value,
):
    # Create a directory for the test
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(
        artifact_dir,
        f"{load_type}_{load_value}_{timestamp}",
    )
    os.makedirs(run_folder, exist_ok=True)

    # Construct the benchmark command
    command = (
        f"genai-perf profile -m {model} --endpoint-type chat --streaming --num-dataset-entries 1000 "
        f"--service-kind openai --endpoint v1/chat/completions --request-count {request_count} --warmup-request-count 10 "
        f"--random-seed 123 --synthetic-input-tokens-stddev 0 --output-tokens-stddev 0 "
        f"--tokenizer {tokenizer} --synthetic-input-tokens-mean {input_tokens_uncached} --output-tokens-mean {output_tokens} "
        f"--extra-inputs seed:100 --extra-inputs min_tokens:{output_tokens} --extra-inputs max_tokens:{output_tokens} "
        f"--profile-export-file my_profile_export.json --url {url} --artifact-dir {run_folder} "
        f"--num-prefix-prompts 1 --prefix-prompt-length {input_tokens_cached} "
    )
    if load_type == "rps":
        command += f"--request-rate {load_value} "
    elif load_type == "concurrency":
        command += f"--concurrency {int(load_value)} "
    else:
        raise ValueError(f"Invalid load type: {load_type}")

    command += "-- -v --async "

    print(command)

    # Print information about the run
    LOGGER.info(
        f"ISL cached: {input_tokens_cached}, ISL uncached: {input_tokens_uncached}, OSL: {output_tokens}, {load_type}: {load_value}"
    )
    LOGGER.info(f"Saving artifacts in: {run_folder}")
    LOGGER.info(f"Command: {command}")
    # Save the command to a file in artifacts folder with execution permissions and shebang
    with open(os.path.join(run_folder, "run.sh"), "w") as f:
        f.write("#!/bin/bash\n")
        f.write(command)
        f.close()
        subprocess.run(["chmod", "+x", os.path.join(run_folder, "run.sh")])

    # Prepare output file
    output_file = os.path.join(run_folder, "output.txt")

    # Run the command and capture both stdout and stderr
    with open(output_file, "w") as f:
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        for line in process.stdout:
            print(line.decode(), end="")
            f.write(line.decode())
        for line in process.stderr:
            print(line.decode(), end="")
            f.write(line.decode())

    process.wait()


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark for GenAI")
    parser.add_argument(
        "--isl-cached",
        type=int,
        default=0,
        help="Input sequence length (cached)",
    )
    parser.add_argument(
        "--isl-uncached",
        type=int,
        required=True,
        help="Input sequence length (uncached)",
    )
    parser.add_argument(
        "--osl",
        type=int,
        required=True,
        help="Output sequence length",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
        help="Tokenizer name",
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="URL of the API server",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="artifacts/",
        help="Directory to save artifacts",
    )
    parser.add_argument(
        "--request-count",
        type=int,
        required=True,
        help="Number of requests to send",
    )
    parser.add_argument(
        "--load-type",
        type=str,
        required=True,
        help="Type of load: rps or concurrency",
    )
    parser.add_argument(
        "--load-value",
        type=float,
        required=True,
        help="Value of load",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_benchmark(
        args.isl_cached,
        args.isl_uncached,
        args.osl,
        args.model,
        args.tokenizer,
        args.url,
        args.artifact_dir,
        args.request_count,
        args.load_type,
        args.load_value,
    )


if __name__ == "__main__":
    main()
