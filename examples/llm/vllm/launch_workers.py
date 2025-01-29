#!/usr/bin/env python3

import argparse
import datetime
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time

import requests

LOGGER = logging.getLogger(__name__)

# Change these paths to match your new codebase layout
EXAMPLE_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
LOG_DIR = os.path.join(EXAMPLE_ROOT, "logs")
WORKER_LOG_DIR = os.path.join(LOG_DIR, "workers")

processes = []
nats_store = ""


def update_env():
    # Example environment variables you might need:
    # os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
    # os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # ...
    pass


def handler(signum, frame):
    for process in processes:
        if process:
            process.terminate()
            process.kill()
    if processes:
        LOGGER.info("exiting")
        LOGGER.info(processes)
        shutil.rmtree(nats_store, ignore_errors=True)
        sys.exit(0)


# Set up signal handling
signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
for sig in signals:
    try:
        signal.signal(sig, handler)
    except Exception:
        pass


def get_visible_devices() -> list[int]:
    """Returns a list of GPU IDs that are visible (per CUDA_VISIBLE_DEVICES or nvidia-smi)."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_devices:
        return [int(s) for s in visible_devices.split(",")]
    else:
        output = os.popen("nvidia-smi --list-gpus").read()
        count = len(output.strip().split("\n"))
        return list(range(count))


def _get_artifacts_dir(args):
    """Same artifact-dir logic as in the old script."""
    if args.baseline_workers is not None and args.baseline_workers > 0:
        artifact_dir = (
            f"{args.artifact_dir}/{datetime.datetime.today().strftime('%Y-%m-%d')}/"
            f"isl_cached_{args.isl_cached}_isl_uncached_{args.isl_uncached}_osl_{args.osl}/"
            f"{'chunked_' if args.enable_chunked_prefill else ''}"
            f"baseline_tp{args.baseline_tp_size}dp{args.baseline_workers}"
        )
    else:
        artifact_dir = (
            f"{args.artifact_dir}/{datetime.datetime.today().strftime('%Y-%m-%d')}/"
            f"isl_cached_{args.isl_cached}_isl_uncached_{args.isl_uncached}_osl_{args.osl}/"
            f"context_tp{args.context_tp_size}dp{args.context_workers}_"
            f"generate_tp{args.generate_tp_size}dp{args.generate_workers}"
        )
    return artifact_dir


def _kill_processes():
    """Kills leftover processes. Adjust the grep patterns as needed for your new code."""
    cmd = (
        "ps aux | grep -e 'multi' -e '_worker' -e 'nats' -e '_api' | grep -v grep | "
        "awk '{print $2}' | xargs kill -9"
    )
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        LOGGER.info("Kill leftover processes command executed successfully")
        LOGGER.info("Output: %s", result.stdout)
    except subprocess.CalledProcessError as e:
        LOGGER.error("Command failed with error: %s", e)
        LOGGER.error("Error output: %s", e.stderr)


def _launch_api_server(args, index):
    """
    Launch the new API server using the new command:
      python -m llm.api_server \
         --tokenizer <args.model_ckpt>
         --request-plane-uri <args.nats_url>
         --api-server-host <address>
         --api-server-port <port>
         --model-name <args.model_name>
    """
    address = str(args.api_server_url).split(":")[1][2:]  # e.g. "hostname"
    port = str(args.api_server_url).split(":")[-1]  # e.g. "8005"

    command = [
        "python3",
        "-m",
        "llm.api_server",
        "--tokenizer",
        args.model_ckpt,
        "--request-plane-uri",
        args.nats_url,
        "--api-server-host",
        address,
        "--api-server-port",
        port,
        "--model-name",
        args.model_name,
    ]

    LOGGER.info("Launching API server: %s", " ".join(command))

    if args.dry_run:
        return None

    process = subprocess.Popen(command)
    time.sleep(1)
    return process


def _generate_gpu_assigment(ngpus: int | list[int], task_types, world_size):
    """
    The old assignment logic: we chunk GPUs among tasks.

    task_types is a list of:
      (str: 'context'|'generate'|'baseline', workers: int, tp_size: int)

    This returns a dictionary:
      task_id_to_runs_dict = {
         <task_id> : [ ( [gpu_ids...], <task_type> ), ... ],
         ...
      }
    With length = world_size, so each rank can get tasks_asigment[rank].
    """
    gpu_ids = ngpus if isinstance(ngpus, list) else list(range(ngpus))
    gpuid_index = 0
    gpuid = gpu_ids[gpuid_index]
    task_id = 0
    workers_division = []
    worker_id = 0
    for task_type, num_workers, tp_size in task_types:
        for _ in range(num_workers):
            tasks = []
            for _ in range(tp_size):
                tasks.append((worker_id, task_id, gpuid, task_type))
                gpuid_index += 1
                if gpuid_index >= len(gpu_ids):
                    gpuid_index = 0
                    task_id += 1
                gpuid = gpu_ids[gpuid_index]
            worker_id += 1
            workers_division.append(tasks)

    # Ensure consistency in tasks
    for task in workers_division:
        _, task_id_first, _, _ = task[0]
        for t in task[1:]:
            if t[1] != task_id_first:
                raise ValueError(f"Task ID mismatch in {task}")

    # Convert the list-of-lists into a dict keyed by task_id
    task_id_to_runs_dict = {}
    for task in workers_division:
        # all sub-items share the same task_id
        first_item = task[0]
        w_id, t_id, _, t_type = first_item
        gpus_in_tp = [gpuid for (_, _, gpuid, _) in task]
        if t_id not in task_id_to_runs_dict:
            task_id_to_runs_dict[t_id] = []
        task_id_to_runs_dict[t_id].append((gpus_in_tp, t_type))

    if len(task_id_to_runs_dict) != world_size:
        raise ValueError(
            f"World size mismatch: {len(task_id_to_runs_dict)} != {world_size}. "
            f"Maybe you requested more tasks than #SLURM_NTASKS?"
        )
    return task_id_to_runs_dict


def parse_slurm_nodelist(env_var="SLURM_STEP_NODELIST"):
    """
    Parse the SLURM node list. Return a list of hostnames.
    If not in SLURM, returns ["localhost"].
    """
    nodelist_str = os.getenv(env_var)
    if not nodelist_str:
        return ["localhost"]

    def expand_range(range_str, padding):
        start, end = range_str.split("-")
        return [str(i).zfill(padding) for i in range(int(start), int(end) + 1)]

    prefix_pattern = re.compile(r"([a-zA-Z\-]+)\[?(\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)\]?")
    match = prefix_pattern.match(nodelist_str)
    if not match:
        raise ValueError(f"Invalid nodelist format: {nodelist_str}")

    prefix, node_ranges = match.groups()
    nodes = []
    for part in node_ranges.split(","):
        if "-" in part:
            padding = len(part.split("-")[0])
            nodes.extend([f"{prefix}{node}" for node in expand_range(part, padding)])
        else:
            padding = len(part)
            nodes.append(f"{prefix}{part.zfill(padding)}")

    return nodes


def generate_common_args(args):
    """
    Returns the list of common arguments used for each worker in the new 'llm.vllm.deploy' command.
    """
    common = [
        "python3",
        "-m",
        "llm.vllm.deploy",
        "--model-name",
        args.model_ckpt,
        "--kv-cache-dtype",
        "fp8",
        "--dtype",
        "auto",
        "--worker-name",
        args.model_name,
        "--disable-async-output-proc",
        "--disable-log-stats",
        "--context-tp-size",
        args.context_tp_size,
        "--generate-tp-size",
        args.generate_tp_size,
        "--baseline-tp-size",
        args.baseline_tp_size,
        # The new script might require or accept a single max-model-len, etc.
    ]
    if args.max_model_len is not None:
        common += ["--max-model-len", str(args.max_model_len)]

    # We point to the external NATS URL as the "request-plane-uri".
    # If you no longer need an external NATS server, adapt accordingly.
    common += [
        "--request-plane-uri",
        args.nats_url,
    ]

    return common


def generate_profiling_args(args, task_type: str, worker_id: int):
    """
    Builds NSight Systems command prefix for optional profiling.
    """
    timestamp = datetime.datetime.today().strftime("%Y%m%d%H%M%S")
    job_id = os.environ.get("SLURM_JOBID", "0")
    artifact_dir = _get_artifacts_dir(args)

    # Example nsys profile command
    return [
        "nsys",
        "profile",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop-shutdown",
        "--force-overwrite=true",
        "--sample=none",
        "--cpuctxsw=none",
        "--trace=cuda,nvtx,osrt,ucx,python-gil",
        "--python-sampling=true",
        "--python-sampling-frequency=100",
        "--trace-fork-before-exec=true",
        "--gpu-metrics-device=all",
        "--cuda-graph-trace=node",
        f"--output={artifact_dir}/prof_job-{job_id}_task-{task_type}_worker-{worker_id}_{timestamp}.nsys-rep",
    ]


def _launch_workers(args):
    """
    Launch context/generate/baseline workers using the new
    `python -m llm.vllm.deploy` command.
    """
    env = os.environ.copy()

    context_workers = args.context_workers or 0
    generate_workers = args.generate_workers or 0
    baseline_workers = args.baseline_workers or 0

    # For each type, we pass (type_name, number_of_workers, tensor_parallel_size).
    task_types = [
        ("context", context_workers, args.context_tp_size),
        ("generate", generate_workers, args.generate_tp_size),
        ("baseline", baseline_workers, args.baseline_tp_size),
    ]

    # Standard SLURM environment variables
    rank = int(os.environ.get("SLURM_PROCID", 0))
    ntasks = int(os.environ.get("SLURM_NTASKS", 1))
    LOGGER.info(f"Ntasks configuration {ntasks}")

    # Build a dictionary assigning tasks to each rank
    tasks_assignment = _generate_gpu_assigment(args.visible_devices, task_types, ntasks)
    # tasks_assignment[rank] => list of ( [gpus_ids], <task_type> )
    tasks_for_this_rank = tasks_assignment[rank]

    # Count how many tasks appear before me, so we can label them properly in profiling logs
    previous_tasks = sum(len(tasks_assignment[r]) for r in range(rank))

    commands = []
    logpaths = []
    cuda_devices = []

    # We'll keep a directory for logs
    if not args.dry_run:
        shutil.rmtree(WORKER_LOG_DIR, ignore_errors=True)
        os.makedirs(WORKER_LOG_DIR, exist_ok=True)

    # We'll add some environment variables that might be read inside the new code:
    env["VLLM_TORCH_HOST"] = args.host
    env["VLLM_TORCH_PORT"] = "36183"  # example port; adapt as needed
    env["VLLM_BASELINE_WORKERS"] = str(baseline_workers)
    env["VLLM_CONTEXT_WORKERS"] = str(context_workers)
    env["VLLM_GENERATE_WORKERS"] = str(generate_workers)
    env["VLLM_BASELINE_TP_SIZE"] = str(args.baseline_tp_size)
    env["VLLM_CONTEXT_TP_SIZE"] = str(args.context_tp_size)
    env["VLLM_GENERATE_TP_SIZE"] = str(args.generate_tp_size)
    env["VLLM_LOGGING_LEVEL"] = args.log_level
    env["VLLM_DATA_PLANE_BACKEND"] = args.data_plane_backend
    env["PYTHONUNBUFFERED"] = "1"

    for worker_id_in_rank, (gpu_ids, task_type) in enumerate(tasks_for_this_rank):
        # We'll build the command line by starting with the common arguments
        cmd = generate_common_args(args)

        # We need to specify either "context-worker-count", "generate-worker-count",
        # or "baseline-worker-count" for each process. Adjust to your new code's flags.
        if task_type == "context":
            # Example: one context worker in this process
            cmd += [
                "--context-worker-count",
                "1",
                "--generate-worker-count",
                "0",
            ]
            # Possibly override max-batch-size
            cmd += ["--max-batch-size", str(args.context_max_batch_size)]
            cmd += [
                "--gpu-memory-utilization",
                str(args.context_gpu_memory_utilization),
            ]
            # If you have a separate max-num-seqs for context:
            max_num_seqs = -1
            if args.max_num_seqs != -1:
                max_num_seqs = args.max_num_seqs
            if args.context_max_num_seqs != -1:
                max_num_seqs = args.context_max_num_seqs
            if max_num_seqs != -1:
                cmd += ["--max-num-seqs", str(max_num_seqs)]

        elif task_type == "generate":
            cmd += [
                "--context-worker-count",
                "0",
                "--generate-worker-count",
                "1",
            ]
            cmd += ["--max-batch-size", str(args.generate_max_batch_size)]
            cmd += [
                "--gpu-memory-utilization",
                str(args.generate_gpu_memory_utilization),
            ]
            # max-num-seqs logic
            max_num_seqs = -1
            if args.generate_max_num_seqs != -1:
                max_num_seqs = args.generate_max_num_seqs
            if args.max_num_seqs != -1:
                max_num_seqs = args.max_num_seqs
            if max_num_seqs != -1:
                cmd += ["--max-num-seqs", str(max_num_seqs)]

        elif task_type == "baseline":
            # If your new code still uses a "baseline" concept:
            cmd += [
                "--baseline-worker-count",
                "1",
            ]
            cmd += ["--max-batch-size", str(args.baseline_max_batch_size)]
            cmd += [
                "--gpu-memory-utilization",
                str(args.baseline_gpu_memory_utilization),
            ]
            if args.enable_chunked_prefill:
                cmd += ["--enable-chunked-prefill"]
            if args.enable_prefix_caching:
                cmd += ["--enable-prefix-caching"]
            if args.max_num_seqs != -1:
                cmd += ["--max-num-seqs", str(args.max_num_seqs)]
        else:
            raise ValueError(f"Unknown task_type={task_type}")

        # If you only want the first worker on rank 0 to initialize the request-plane (like nats):
        # or if each worker should do it, adapt as needed:
        if rank == 0 and worker_id_in_rank == 0:
            cmd += ["--initialize-request-plane"]

        # Possibly add profiling
        if args.profile_workers:
            global_worker_index = previous_tasks + worker_id_in_rank
            cmd = generate_profiling_args(args, task_type, global_worker_index) + cmd
            env["RUN_PROFILING"] = "1"

        # We'll keep track for logging
        worker_name = f"{task_type}_{rank}_{worker_id_in_rank}"
        logpaths.append(os.path.join(WORKER_LOG_DIR, worker_name))
        cuda_devices.append(gpu_ids)
        commands.append(cmd)

    # Actually launch each worker
    spawned = []
    for i, (cmd, logpath, devices) in enumerate(zip(commands, logpaths, cuda_devices)):
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
        # ID for debugging
        env["VLLM_WORKER_ID"] = str(i + previous_tasks)

        LOGGER.info(
            f"Launching worker with CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
            f"VLLM_WORKER_ID={env['VLLM_WORKER_ID']} :\n {' '.join(cmd)}"
        )

        if args.dry_run:
            spawned.append(None)
            continue

        process = subprocess.Popen(cmd, env=env)
        time.sleep(1)
        spawned.append(process)

    return spawned


def _launch_nats_server(args):
    """
    Launch an external NATS server on the given address:port from --nats-url.
    If your new code has replaced this with `--initialize-request-plane`,
    you can remove or comment out this function.
    """
    address = str(args.nats_url).split(":")[1][2:]
    port = str(args.nats_url).split(":")[-1]
    LOGGER.info(f"Launch NATS.io {address}:{port}")
    command = [
        "/usr/local/bin/nats-server",
        "--jetstream",
        "-a",
        address,
        "--port",
        port,
        "--store_dir",
        args.nats_store,
    ]
    if args.nats_debug:
        command.extend(["--debug", "--trace"])

    LOGGER.info(" ".join(command))
    if args.dry_run:
        return None

    shutil.rmtree(args.nats_store, ignore_errors=True)
    process = subprocess.Popen(command)
    time.sleep(1)
    return process


def run_benchmark(args):
    """
    Same logic from old script for running the benchmark with 'run_benchmark.py'.
    If you now use a different tool (e.g., genai-perf), adapt accordingly.
    """
    LOGGER.info(f"RUNNING BENCHMARKS for load values: {args.load_value}")
    artifact_dir = _get_artifacts_dir(args)
    for load_value in args.load_value:
        request_count = args.request_count_per_load_value * load_value
        if args.min_request_count is not None:
            request_count = max(request_count, args.min_request_count)

        benchmark_command = [
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "run_benchmark.py"
            ),
            "--model",
            args.model_name,
            "--url",
            args.api_server_url,
            "--isl-cached",
            str(args.isl_cached),
            "--isl-uncached",
            str(args.isl_uncached),
            "--osl",
            str(args.osl),
            "--load-type",
            args.load_type,
            "--load-value",
            str(load_value),
            "--request-count",
            str(request_count),
            "--artifact-dir",
            artifact_dir,
        ]

        if args.dry_run:
            LOGGER.info("Benchmark command (dry run): %s", " ".join(benchmark_command))
        else:
            LOGGER.info(f"RUNNING BENCHMARK FOR load_value={load_value}")
            LOGGER.info(" ".join(benchmark_command))
            subprocess.run(benchmark_command, check=True)


def wait_for_server(url, timeout=300):
    """
    Wait for the server to become ready by sending a sample request
    to /v1/chat/completions. If it responds with status=200, we proceed.
    """
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "llama",
        "messages": [{"role": "system", "content": "What is the capital of France?"}],
        "temperature": 0,
        "top_p": 0.95,
        "max_tokens": 25,
        "min_tokens": 25,
        "stream": True,
        "n": 1,
        "frequency_penalty": 0.0,
        "stop": [],
    }
    start = time.time()
    while True:
        try:
            response = requests.post(url, data=json.dumps(data), headers=headers)
            if response.status_code == 200:
                LOGGER.info(f"Health check success: {response.content}")
                return True
            else:
                if time.time() - start > timeout:
                    raise RuntimeError(f"Server not responding: {response.status_code}")
                LOGGER.warning(f"Server not ready yet: {response.status_code}")
                time.sleep(5)
        except requests.exceptions.RequestException as e:
            LOGGER.warning(f"Server not responding: {e}")
            time.sleep(5)
            if time.time() - start > timeout:
                raise RuntimeError(f"Server not responding within {timeout}s: {e}")


def _parse_args():
    global nats_store
    parser = argparse.ArgumentParser(
        description="Launch multi-node vLLM with new commands"
    )

    parser.add_argument(
        "--model-ckpt",
        type=str,
        default="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
        help="Model checkpoint name/path",
    )
    parser.add_argument(
        "--model-name", type=str, default="llama", help="Friendly model name"
    )

    parser.add_argument("--log-dir", default=LOG_DIR)
    parser.add_argument("--dry-run", action="store_true", default=False)

    parser.add_argument("--context-tp-size", type=int, default=1)
    parser.add_argument("--generate-tp-size", type=int, default=1)
    parser.add_argument("--baseline-tp-size", type=int, default=1)

    parser.add_argument("--context-max-batch-size", type=int, default=10000)
    parser.add_argument("--generate-max-batch-size", type=int, default=10000)
    parser.add_argument("--baseline-max-batch-size", type=int, default=10000)

    parser.add_argument("--log-level", type=str, default="INFO")

    # We'll parse the first node from SLURM for our nats-url
    hosts = parse_slurm_nodelist()
    host = hosts[0]

    parser.add_argument("--nats-url", type=str, default=f"nats://{host}:4223")
    parser.add_argument("--nats-store", type=str, default="/tmp/nats/triton-3-demo")
    parser.add_argument("--nats-debug", action="store_true", default=False)

    parser.add_argument(
        "--context-workers", type=int, help="DP count of context workers"
    )
    parser.add_argument(
        "--generate-workers", type=int, help="DP count of generate workers"
    )
    parser.add_argument(
        "--baseline-workers", type=int, help="DP count of baseline workers"
    )

    parser.add_argument("--api-server-url", type=str, default=f"http://{host}:8005")
    parser.add_argument("--workers-only", action="store_true", default=False)

    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=-1)
    parser.add_argument("--context-max-num-seqs", type=int, default=-1)
    parser.add_argument("--generate-max-num-seqs", type=int, default=-1)

    # Benchmark related
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-timeout", type=int, default=300)
    parser.add_argument("--isl-cached", type=int, default=0)
    parser.add_argument("--isl-uncached", type=int, default=2048)
    parser.add_argument("--osl", type=int, default=128)
    parser.add_argument(
        "--load-type", type=str, default="concurrency", choices=["concurrency", "rps"]
    )
    parser.add_argument("--load-value", type=int, nargs="+", default=[32])
    parser.add_argument("--request-count-per-load-value", type=int, default=100)
    parser.add_argument("--min-request-count", type=int, default=None)

    parser.add_argument(
        "--data-plane-backend",
        type=str,
        default="nccl",
        choices=["nccl", "ucx"],
        help="Data plane backend for vLLM kv cache transfer.",
    )
    parser.add_argument("--enable-chunked-prefill", action="store_true", default=False)
    parser.add_argument("--enable-prefix-caching", action="store_true", default=False)
    parser.add_argument("--artifact-dir", type=str, default="artifacts")

    parser.add_argument("--baseline-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--context-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--generate-gpu-memory-utilization", type=float, default=0.5)

    parser.add_argument("--profile-workers", action="store_true", default=False)

    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(filename)s: %(levelname)s: %(funcName)s(): %(lineno)d:\t%(message)s",
    )

    # Capture the visible GPUs
    visible_devices = get_visible_devices()
    args.visible_devices = [int(g) for g in visible_devices]
    args.host = host
    nats_store = args.nats_store

    # Validate logic for baseline vs. context/generate
    using_context = args.context_workers is not None and args.context_workers > 0
    using_generate = args.generate_workers is not None and args.generate_workers > 0
    using_baseline = args.baseline_workers is not None and args.baseline_workers > 0

    if using_context != using_generate and not using_baseline:
        parser.error(
            "--context-workers and --generate-workers must be used together, unless you use baseline."
        )

    if not using_context and not using_baseline:
        parser.error(
            "You must specify either --context-workers and --generate-workers OR --baseline-workers"
        )
    if using_context and using_baseline:
        parser.error(
            "--context-workers/--generate-workers are mutually exclusive with --baseline-workers"
        )

    # If you want to enforce only 1 generate worker for now:
    # if using_generate and args.generate_workers > 1:
    #     parser.error("Only one generate worker is currently supported")

    return args


def main():
    try:
        args = _parse_args()
        LOGGER.info(args)
        LOGGER.info(f"Example root: {EXAMPLE_ROOT}")

        # Provide default environment variables or custom logic
        update_env()

        # Make sure logs exist
        os.makedirs(LOG_DIR, exist_ok=True)

        rank = int(os.environ.get("SLURM_PROCID", 0))

        # Then launch local workers (for each rank)
        worker_procs = _launch_workers(args)
        processes.extend(worker_procs)

        # If not just launching workers, rank0 starts the NATS server + API
        if not args.workers_only:
            if rank == 0:
                # Internal requeest plane handling is used in the new code
                # processes.append(_launch_nats_server(args))
                processes.append(_launch_api_server(args, 0))

        # If rank == 0 and we want to run a benchmark, wait for the server
        if rank == 0 and args.benchmark:
            # Wait for the new API endpoint: /v1/chat/completions
            if wait_for_server(
                args.api_server_url + "/v1/chat/completions", args.benchmark_timeout
            ):
                LOGGER.info("Server is ready; starting benchmark.")
                run_benchmark(args)
                LOGGER.info("Benchmark finished.")
                # Optionally exit after benchmark if desired:
                # return

        # Wait for all processes
        for p in processes:
            if p:
                LOGGER.info(f"waiting for {p}")
                p.wait()

    except Exception as e:
        LOGGER.error(e, exc_info=True)
    finally:
        _kill_processes()


if __name__ == "__main__":
    main()
