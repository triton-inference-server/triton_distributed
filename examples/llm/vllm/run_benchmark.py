#!/usr/bin/env python3

import argparse
import datetime
import json
import logging
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time

import requests

LOGGER = logging.getLogger(__name__)

EXAMPLE_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

LOG_DIR = EXAMPLE_ROOT + "/logs"

WORKER_LOG_DIR = LOG_DIR + "/workers"

processes = []

nats_store = ""


def update_env():
    # CUDA_VISIBLE_DEVICES=0 VLLM_ATTENTION_BACKEND=FLASHINFER VLLM_WORKER_MULTIPROC_METHOD=spawn UCX_RNDV_SCHEME=get_zcopy UCX_MEMTYPE_CACHE=n UCX_TLS=tcp,cuda_copy,cuda_ipc
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # os.environ["UCX_RNDV_SCHEME"] = "get_zcopy"
    # os.environ["UCX_MEMTYPE_CACHE"] = "n"
    # os.environ["UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"


def handler(signum, frame):
    for process in processes:
        process.terminate()
        process.kill()
    if processes:
        LOGGER.info("exiting")
        LOGGER.info(processes)
        shutil.rmtree(nats_store, ignore_errors=True)
        sys.exit(0)


signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
for sig in signals:
    try:
        signal.signal(sig, handler)
    except Exception:
        pass


def get_visible_devices() -> list[int]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_devices:
        return [int(s) for s in visible_devices.split(",")]
    else:
        output = os.popen("nvidia-smi --list-gpus").read()
        count = len(output.strip().split("\n"))
        return list(range(count))


def _get_artifacts_dir(args):
    if args.baseline_workers is not None and args.baseline_workers > 0:
        artifact_dir = f"{args.artifact_dir}/{datetime.datetime.today().strftime('%Y-%m-%d')}/isl_cached_{args.isl_cached}_isl_uncached_{args.isl_uncached}_osl_{args.osl}/{'chunked_' if args.enable_chunked_prefill else ''}baseline_tp{args.baseline_tp_size}dp{args.baseline_workers}"
    else:
        artifact_dir = f"{args.artifact_dir}/{datetime.datetime.today().strftime('%Y-%m-%d')}/isl_cached_{args.isl_cached}_isl_uncached_{args.isl_uncached}_osl_{args.osl}/context_tp{args.context_tp_size}dp{args.context_workers}_generate_tp{args.generate_tp_size}dp{args.generate_workers}"
    return artifact_dir


def _kill_processes():
    cmd = "ps aux | grep -e 'multi' -e '_worker' -e 'nats' -e '_api' | grep -v grep | awk '{print $2}' | xargs kill -9"

    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        LOGGER.info("Command executed successfully")
        LOGGER.info("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        LOGGER.error("Command failed with error:", e)
        LOGGER.error("Error output:", e.stderr)


def _launch_api_server(args, index):
    api_server_script_path = f"{EXAMPLE_ROOT}/api_server.py"
    address = str(args.api_server_url).split(":")[1][2:]
    port = str(args.api_server_url).split(":")[-1]

    command = [
        api_server_script_path,
        "--api-server-port",
        port,
        "--api-server-host",
        address,
        "--nats-url",
        args.nats_url,
        "--log-level",
        args.log_level,
        "--model-name",
        args.model_name,
        "--tokenizer",
        args.model_ckpt,
    ]

    LOGGER.info(" ".join(command))

    if args.dry_run:
        return

    # with open(f"{LOG_DIR}/api_server_{index}.stdout.log", "wt") as output_:
    #     with open(f"{LOG_DIR}/api_server_{index}.stderr.log", "wt") as output_err:
    process = subprocess.Popen(
        command,
        # stdin=subprocess.DEVNULL, stdout=output_, stderr=output_err
    )
    time.sleep(1)
    return process


def _generate_gpu_assigment(ngpus: int | list[int], task_types, world_size):
    # Create a list of worker assignments to GPUs in world for all tasks
    # There is a single task per node and it must commit all GPUs to a single worker
    # or split them between context and generate workers and baseline workers
    gpu_ids = ngpus if isinstance(ngpus, list) else list(range(ngpus))
    gpuid_index = 0
    gpuid = gpu_ids[gpuid_index]
    task_id = 0
    workers_division = []
    worker_id = 0
    for task_type, workers, tp_size in task_types:
        for _ in range(workers):
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
    for task in workers_division:
        _, task_id_first, __, __ = task[0]
        if len(task) > 1:
            for worker_id, task_id, gpuid, task_type in task[1:]:
                if task_id_first != task_id:
                    raise ValueError(
                        f"Task ID mismatch: {task_id_first} != {task_id} for {task}, workers_division: {workers_division}"
                    )
    task_id_to_runs_dict = {}
    for task in workers_division:
        _, task_id, __, task_type = task[0]

        # Build GPU ids list for each task
        gpus_ids = [gpuid for _, _, gpuid, _ in task]
        if task_id not in task_id_to_runs_dict:
            task_id_to_runs_dict[task_id] = []
        task_id_to_runs_dict[task_id].append((gpus_ids, task_type))
    if len(task_id_to_runs_dict) != world_size:
        raise ValueError(
            f"World size mismatch: {len(task_id_to_runs_dict)} != {world_size} workers_division: {workers_division} task_id_to_runs_dict: {task_id_to_runs_dict}"
        )
    return task_id_to_runs_dict


def parse_slurm_nodelist(env_var="SLURM_STEP_NODELIST"):
    # Get the SLURM_STEP_NODELIST environment variable
    nodelist_str = os.getenv(env_var)

    # Return localhost if the environment variable is not set
    if not nodelist_str:
        return ["localhost"]

    # Function to expand a range like "0045-0047" into a list of numbers
    def expand_range(range_str, padding):
        start, end = range_str.split("-")
        return [str(i).zfill(padding) for i in range(int(start), int(end) + 1)]

    # Regular expression to match the prefix and the node ranges
    prefix_pattern = re.compile(r"([a-zA-Z\-]+)\[?(\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)\]?")
    match = prefix_pattern.match(nodelist_str)

    # If the pattern doesn't match, the nodelist is not in the expected format
    if not match:
        raise ValueError("Invalid nodelist format")

    prefix, node_ranges = match.groups()

    # Split the node ranges by comma and expand them
    nodes = []
    for part in node_ranges.split(","):
        if "-" in part:
            # Find the padding based on the first number in the range
            padding = len(part.split("-")[0])
            nodes.extend([f"{prefix}{node}" for node in expand_range(part, padding)])
        else:
            # Use the length of the part for padding if it's a single node
            padding = len(part)
            nodes.append(f"{prefix}{part.zfill(padding)}")

    return nodes


def generate_common_args(args):
    command_args = [
        "--nats-url",
        args.nats_url,
        "--model-name",
        args.model_ckpt,
        "--kv-cache-dtype",
        "fp8",
        "--dtype",
        "auto",
        "--log-level",
        args.log_level,
        "--worker-name",
        args.model_name,
        "--disable-async-output-proc",
        "--disable-log-stats",
    ]

    if args.max_model_len is not None:
        command_args.extend(["--max-model-len", str(args.max_model_len)])
    return command_args


def generate_profiling_args(args, task_type: str, worker_id: int):
    timestamp = datetime.datetime.today().strftime("%Y%m%d%H%M%S")
    job_id = os.environ.get("SLURM_JOBID", 0)
    artifact_dir = _get_artifacts_dir(args)
    return [
        "nsys",
        "profile",
        # control with cudaProfilerStart/cudaProfilerStop
        "--capture-range=cudaProfilerApi",
        # application will be stopped after the profiling is done
        "--capture-range-end=stop-shutdown",
        "--force-overwrite=true",
        "--sample=none",
        "--cpuctxsw=none",
        "--trace=cuda,nvtx,osrt,ucx,python-gil",
        # https://docs.nvidia.com/nsight-systems/UserGuide/index.html#python-profiling
        "--python-sampling=true",
        "--python-sampling-frequency=100",
        "--trace-fork-before-exec=true",
        "--gpu-metrics-device=all",
        "--cuda-graph-trace=node",
        f"--output={artifact_dir}/prof_job-{job_id}_task-{task_type}_worker-{worker_id}_{timestamp}.nsys-rep",
        "python3",
    ]


def _launch_workers(args):
    env = os.environ.copy()

    context_workers = args.context_workers or 0
    generate_workers = args.generate_workers or 0
    baseline_workers = args.baseline_workers or 0

    task_types = [
        ["context", context_workers, args.context_tp_size],
        ["generate", generate_workers, args.generate_tp_size],
        ["baseline", baseline_workers, args.baseline_tp_size],
    ]

    rank = int(os.environ.get("SLURM_PROCID", 0))
    ntasks = int(os.environ.get("SLURM_NTASKS", 1))

    tasks_asigment = _generate_gpu_assigment(args.visible_devices, task_types, ntasks)

    previous_tasks = 0
    for i in range(rank):
        previous_tasks += len(tasks_asigment[i])

    tasks_asigment = tasks_asigment[rank]

    LOGGER.info(f"Rank: {rank}, previous_tasks: {previous_tasks}")

    commands = []
    logpaths = []
    cuda_devices = []

    for gpus_ids, task_type in tasks_asigment:
        if task_type == "context":
            command_args = generate_common_args(args)
            max_num_seqs = -1
            if args.max_num_seqs != -1:
                max_num_seqs = args.max_num_seqs
            if args.context_max_num_seqs != -1:
                max_num_seqs = args.context_max_num_seqs
            if max_num_seqs != -1:
                command_args.extend(["--max-num-seqs", str(max_num_seqs)])
            command_args.extend(["--max-batch-size", str(args.context_max_batch_size)])
            command_args.extend(
                [
                    "--gpu-memory-utilization",
                    str(args.context_gpu_memory_utilization),
                ]
            )
            if args.enable_prefix_caching:
                command_args.append("--enable-prefix-caching")
            commands.append(
                [f"{EXAMPLE_ROOT}/workers/worker_prefill.py"]
                + command_args
                + [
                    "--context-tp-size",
                    str(args.context_tp_size),
                    "--generate-tp-size",
                    str(args.generate_tp_size),
                ]
            )
            logpaths.append(f"{WORKER_LOG_DIR}/context_{rank}")
            cuda_devices.append(gpus_ids)
        elif task_type == "generate":
            command_args = generate_common_args(args)
            max_num_seqs = -1
            if args.generate_max_num_seqs != -1:
                max_num_seqs = str(args.generate_max_num_seqs)
            if args.max_num_seqs != -1:
                max_num_seqs = str(args.max_num_seqs)
            if max_num_seqs != -1:
                command_args.extend(["--max-num-seqs", str(max_num_seqs)])
            command_args.extend(["--max-batch-size", str(args.generate_max_batch_size)])
            command_args.extend(
                [
                    "--gpu-memory-utilization",
                    str(args.generate_gpu_memory_utilization),
                ]
            )
            # TODO ptarasiewicz: enable prefix caching for generate workers
            # if args.enable_prefix_caching:
            #     command_args.append("--enable-prefix-caching")
            commands.append(
                [f"{EXAMPLE_ROOT}/workers/worker_generate.py"]
                + command_args
                + [
                    "--context-tp-size",
                    str(args.context_tp_size),
                    "--generate-tp-size",
                    str(args.generate_tp_size),
                ]
            )
            logpaths.append(f"{WORKER_LOG_DIR}/generate_{rank}")
            cuda_devices.append(gpus_ids)
        elif task_type == "baseline":
            command_args = generate_common_args(args)
            max_num_seqs = -1
            if args.max_num_seqs != -1:
                max_num_seqs = args.max_num_seqs
            if max_num_seqs != -1:
                command_args.extend(["--max-num-seqs", str(max_num_seqs)])
            command_args.extend(["--max-batch-size", str(args.baseline_max_batch_size)])
            command_args.extend(
                [
                    "--gpu-memory-utilization",
                    str(args.baseline_gpu_memory_utilization),
                ]
            )
            if args.enable_chunked_prefill:
                command_args.append("--enable-chunked-prefill")
            if args.enable_prefix_caching:
                command_args.append("--enable-prefix-caching")
            commands.append(
                [f"{EXAMPLE_ROOT}/workers/worker_baseline.py"]
                + command_args
                + ["--baseline-tp-size", str(args.baseline_tp_size)]
            )
            logpaths.append(f"{WORKER_LOG_DIR}/baseline_{rank}")
            cuda_devices.append(gpus_ids)
        else:
            raise ValueError(f"Invalid task type: {task_type}")

        if args.profile_workers:
            worker_id = len(commands) - 1
            commands[-1] = (
                generate_profiling_args(args, task_type, worker_id + previous_tasks)
                + commands[-1]
            )
            env["RUN_PROFILING"] = "1"

    processes = []
    if not args.dry_run:
        shutil.rmtree(WORKER_LOG_DIR, ignore_errors=True)
        os.makedirs(WORKER_LOG_DIR, exist_ok=True)

    env["VLLM_TORCH_HOST"] = args.host
    env["VLLM_TORCH_PORT"] = "36183"
    env["VLLM_BASELINE_WORKERS"] = str(baseline_workers)
    env["VLLM_CONTEXT_WORKERS"] = str(context_workers)
    env["VLLM_GENERATE_WORKERS"] = str(generate_workers)
    env["VLLM_BASELINE_TP_SIZE"] = str(args.baseline_tp_size)
    env["VLLM_CONTEXT_TP_SIZE"] = str(args.context_tp_size)
    env["VLLM_GENERATE_TP_SIZE"] = str(args.generate_tp_size)
    env["VLLM_LOGGING_LEVEL"] = args.log_level
    env["VLLM_DATA_PLANE_BACKEND"] = args.data_plane_backend
    env["PYTHONUNBUFFERED"] = "1"

    for worker_id, (command, logpath, devices) in enumerate(
        zip(commands, logpaths, cuda_devices)
    ):
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
        env["VLLM_WORKER_ID"] = str(worker_id + previous_tasks)

        LOGGER.info(
            f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} VLLM_WORKER_ID={env['VLLM_WORKER_ID']} {' '.join(command)}"
        )

        if args.dry_run:
            continue

        # with open(f"{logpath}.stdout.log", "wt") as output_:
        #     with open(f"{logpath}.stderr.log", "wt") as output_err:
        process = subprocess.Popen(
            command,
            env=env,
            # stdin=subprocess.DEVNULL,
            # stdout=output_,
            # stderr=output_err,
        )
        time.sleep(1)
        processes.append(process)

    return processes


def _launch_nats_server(args):
    # nats://127.0.0.1:4223
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
        return

    shutil.rmtree(args.nats_store, ignore_errors=True)

    # with open(f"{LOG_DIR}/nats_server.stdout.log", "wt") as output_:
    #     with open(f"{LOG_DIR}/nats_server.stderr.log", "wt") as output_err:
    process = subprocess.Popen(
        command,
        # stdin=subprocess.DEVNULL, stdout=output_, stderr=output_err
    )
    time.sleep(1)
    return process


def run_benchmark(args):
    LOGGER.info(f"RUNNING BENCHMARKS {args.load_value}")
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
            LOGGER.info(" ".join(benchmark_command))
        else:
            LOGGER.info(f"RUNNING BENCHMARK FOR {load_value}")
            LOGGER.info(" ".join(benchmark_command))
            subprocess.run(benchmark_command, check=True)


def wait_for_server(url, timeout=300):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "llama",
        "messages": [
            {
                "role": "system",
                "content": "What is the capital of France?",
            }
        ],
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
                print(response.content)
                return True
            else:
                if time.time() - start > timeout:
                    raise ValueError(
                        f"Server is not responding: {response.status_code}"
                    )
                LOGGER.warning(f"Server is not responding: {response.status_code}")
                time.sleep(5)
        except requests.exceptions.RequestException as e:
            LOGGER.warning(f"Server is not responding: {e}")
            time.sleep(5)
            if time.time() - start > timeout:
                raise ValueError(f"Server is not responding: {e}")


def launch_ucx_perftest(rank, args):
    # CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=mlx5_0:1 UCX_TLS=rc,cuda_copy ucx_perftest -t tag_bw -m cuda -s 100000000 -n 10 -p 9999 -c 0 dgx13

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    cmd = [
        "ucx_perftest",
        "-t",
        "tag_bw",
        "-m",
        "cuda",
        "-s",
        "100000000",
        "-n",
        "10",
        "-p",
        "9999",
        "-c",
        "0",
    ]
    if rank != 0:
        cmd.extend([args.host])

    LOGGER.info(" ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def launch_ucp_test(rank, args):
    env = os.environ.copy()
    cmd = [
        "python3",
        f"{EXAMPLE_ROOT}/scripts/ucp_test.py",
        "--rank",
        str(rank),
        "--host",
        args.host,
        "--device",
        str(random.randint(0, 7)),
    ]
    cmd1 = cmd + ["--port", "13337"]
    cmd2 = cmd + ["--port", "13338"]
    LOGGER.info(" ".join(cmd1))
    LOGGER.info(" ".join(cmd2))
    test1 = subprocess.Popen(cmd1, env=env)
    test2 = subprocess.Popen(cmd2, env=env)
    test1.wait()
    test2.wait()


def _parse_args():
    global nats_store
    parser = argparse.ArgumentParser(description="Launch Servers")
    parser.add_argument(
        "--model-ckpt",
        type=str,
        default="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
        help="model",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="llama",
        help="model",
    )

    parser.add_argument(
        "--log-dir",
        default=LOG_DIR,
    )

    parser.add_argument("--dry-run", action="store_true", default=False)

    parser.add_argument("--context-tp-size", type=int, default=1)

    parser.add_argument("--generate-tp-size", type=int, default=1)

    parser.add_argument("--baseline-tp-size", type=int, default=1)

    parser.add_argument("--context-max-batch-size", type=int, default=10000)

    parser.add_argument("--generate-max-batch-size", type=int, default=10000)

    parser.add_argument("--baseline-max-batch-size", type=int, default=10000)

    parser.add_argument("--log-level", type=str, default="INFO")

    hosts = parse_slurm_nodelist()
    host = hosts[0]

    parser.add_argument("--nats-url", type=str, default=f"nats://{host}:4223")

    parser.add_argument("--nats-store", type=str, default="/tmp/nats/triton-3-demo")

    parser.add_argument("--nats-debug", action="store_true", default=False)

    parser.add_argument("--context-workers", type=int)

    parser.add_argument(
        "--generate-workers",
        type=int,
        help="Number of generate workers, only one or zero is currently supported.",
    )

    parser.add_argument("--baseline-workers", type=int)

    parser.add_argument("--api-server-url", type=str, default=f"http://{host}:8005")

    parser.add_argument("--workers-only", action="store_true", default=False)

    parser.add_argument("--max-model-len", type=int, default=None)

    parser.add_argument("--max-num-seqs", type=int, default=-1)

    parser.add_argument("--context-max-num-seqs", type=int, default=-1)

    parser.add_argument("--generate-max-num-seqs", type=int, default=-1)

    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Enable benchmarking mode.",
    )
    parser.add_argument(
        "--benchmark-timeout",
        type=int,
        default=300,
        help="Timeout for model to be ready for benchmarking, in seconds.",
    )
    parser.add_argument(
        "--isl-cached",
        type=int,
        default=0,
        help="Input prefix sequence length for benchmarking.",
    )
    parser.add_argument(
        "--isl-uncached",
        type=int,
        default=2048,
        help="Input sequence length for benchmarking.",
    )
    parser.add_argument(
        "--osl", type=int, default=128, help="Output sequence length for benchmarking."
    )
    parser.add_argument(
        "--load-type",
        type=str,
        default="concurrency",
        choices=["concurrency", "rps"],
        help="Type of load for benchmarking.",
    )
    parser.add_argument(
        "--load-value",
        type=int,
        nargs="+",
        default=[32],
        help="Values of load for benchmarking.",
    )
    parser.add_argument(
        "--request-count-per-load-value",
        type=int,
        default=100,
        help="Number of requests per load value for benchmarking.",
    )

    parser.add_argument(
        "--min-request-count",
        type=int,
        default=None,
        help="Minimum number of requests for benchmarking.",
    )

    parser.add_argument(
        "--data-plane-backend",
        type=str,
        default="nccl",
        choices=["nccl", "ucx"],
        help="Data plane backend for vLLM kv cache transfer.",
    )

    parser.add_argument(
        "--enable-chunked-prefill",
        action="store_true",
        default=False,
        help="Enable chunked prefill for baseline workers.",
    )

    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        default=False,
        help="Enable prefix caching for baseline workers.",
    )

    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="artifacts",
        help="Directory to store benchmark artifacts.",
    )

    parser.add_argument(
        "--baseline-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for baseline workers.",
    )

    parser.add_argument(
        "--context-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for context workers.",
    )

    parser.add_argument(
        "--generate-gpu-memory-utilization",
        type=float,
        default=0.5,
        help="GPU memory utilization for generate workers.",
    )

    parser.add_argument(
        "--profile-workers",
        action="store_true",
        help="Enable nsight profiling for workers.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(filename)s: "
        "%(levelname)s: "
        "%(funcName)s(): "
        "%(lineno)d:\t"
        "%(message)s",
    )

    visible_devices = get_visible_devices()
    args.visible_devices = [int(nr) for nr in visible_devices]
    args.host = host

    nats_store = args.nats_store

    using_context_workers = (
        args.context_workers is not None and args.context_workers > 0
    )
    using_generate_workers = (
        args.generate_workers is not None and args.generate_workers > 0
    )
    using_baseline_workers = (
        args.baseline_workers is not None and args.baseline_workers > 0
    )

    if using_context_workers != using_generate_workers:
        parser.error("--context-workers and --generate-workers must be used together")

    if not using_context_workers and not using_baseline_workers:
        parser.error(
            "You must specify either --context-workers and --generate-workers or --baseline-workers"
        )

    if using_context_workers == using_baseline_workers:
        parser.error(
            "--context-workers and --generate-workers are mutually exclusive with --baseline-workers"
        )

    # If we reach here, the input is valid
    if args.baseline_workers is not None:
        LOGGER.info(f"Using baseline workers: {args.baseline_workers}")
    else:
        LOGGER.info(
            f"Using context workers: {args.context_workers} and generate workers: {args.generate_workers}"
        )

    if args.generate_workers is not None and args.generate_workers > 1:
        parser.error("Only one generate worker is currently supported")

    return args


def main():
    try:
        args = _parse_args()
        LOGGER.info(args)
        LOGGER.info(f"Example root: {EXAMPLE_ROOT}")
        processes = []

        update_env()

        os.makedirs(LOG_DIR, exist_ok=True)

        rank = int(os.environ.get("SLURM_PROCID", 0))

        if not args.workers_only:
            if rank == 0:
                processes.append(_launch_nats_server(args))
                processes.append(_launch_api_server(args, 0))
        processes.extend(_launch_workers(args))

        if rank == 0:
            if args.benchmark:
                if wait_for_server(
                    args.api_server_url + "/v1/chat/completions", args.benchmark_timeout
                ):
                    LOGGER.info("Server is working, starting benchmark.")
                    run_benchmark(args)
                    LOGGER.info("Benchmark finished.")
                    return

        for process in processes:
            if process:
                LOGGER.info(f"waiting {process}")
                process.wait()
    except Exception as e:
        LOGGER.error(e, exc_info=True)
    finally:
        _kill_processes()


if __name__ == "__main__":
    main()
