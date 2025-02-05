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

import argparse
import os
import shutil
import subprocess
from string import Template

from gpu_info import get_gpu_product_name
from huggingface_hub import snapshot_download
from known_models import KNOWN_MODELS

TARGET_DIR = "/workspace/examples/llm/tensorrtllm/operators"

TENSORRTLLM_EXAMPLE_DIR = "/tensorrtllm_backend/tensorrt_llm/examples"

TENSORRTLLM_BACKEND_DIR = "/tensorrtllm_backend"


def _prepare(args):
    templates = KNOWN_MODELS[args.model]["templates"]
    template_arguments = KNOWN_MODELS[args.model]["template_arguments"]

    model_name = (
        KNOWN_MODELS[args.model]["model_repo_name"]
        if "model_repo_name" in KNOWN_MODELS[args.model]
        else None
    )

    _existing_dir(
        args,
        "tensorrtllm_model",
        args.force_model_repo,
        "model repo",
        suffix=[args.hw_name, f"TP_{args.tp_size}"],
        model_name=model_name,
    )

    for argument, value in template_arguments.items():
        template_arguments[argument] = value.format(args=args)
    template_arguments["request_stats_max_iterations"] = 1000
    print(template_arguments)

    for template in templates:
        if isinstance(template, tuple):
            template_basename = template[1]
            template = template[0]
        else:
            template_basename = os.path.basename(template)
        template_path = os.path.join(
            TENSORRTLLM_BACKEND_DIR,
            "all_models",
            "inflight_batcher_llm",
            template,
            "config.pbtxt",
        )
        if template == "ensemble":
            target_path = os.path.join(
                args.tensorrtllm_model, args.model, "config.pbtxt"
            )
        else:
            target_path = os.path.join(
                args.tensorrtllm_model, template_basename, "config.pbtxt"
            )

        if not args.force_model_repo and os.path.exists(target_path):
            continue

        print(template_path, os.path.exists(template_path), target_path)

        with open(template_path) as f:
            pbtxt_template = Template(f.read())

        pbtxt = pbtxt_template.safe_substitute(template_arguments)

        pbtxt = pbtxt.replace(f'name: "{os.path.basename(template)}"', "")

        if not args.dry_run:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "w") as f:
                f.write(pbtxt)
            model_asset_path = os.path.join(os.path.dirname(template_path), "1")
            if os.path.exists(model_asset_path):
                shutil.copytree(
                    model_asset_path,
                    os.path.join(
                        os.path.dirname(target_path), os.path.basename(model_asset_path)
                    ),
                )


def _call(args, command):
    print(" ".join(command))
    if args.dry_run:
        return 0
    else:
        return subprocess.call(command)


def _existing_dir(args, directory_type, force, command, suffix=[], model_name=None):
    model_name = args.model if model_name is None else model_name
    target_dir = os.path.join(
        args.target_dir, directory_type + "s", model_name, *suffix
    )

    setattr(args, directory_type, target_dir)
    if force:
        if not args.dry_run:
            shutil.rmtree(target_dir, ignore_errors=True)
    if os.path.exists(target_dir):
        print(f"Skipping {command} Found {target_dir}")
        return True

    if not args.dry_run:
        os.makedirs(target_dir, exist_ok=True)

    return False


def _download(args):
    if "hf_id" not in KNOWN_MODELS[args.model]:
        print("Skipping Download")
        return

    if "download_patterns" in KNOWN_MODELS[args.model]:
        patterns = KNOWN_MODELS[args.model]["download_patterns"]
    else:
        patterns = ["*.safetensors", "*.json"]

    model_name = (
        KNOWN_MODELS[args.model]["download_model_name"]
        if "download_model_name" in KNOWN_MODELS[args.model]
        else None
    )

    if _existing_dir(
        args, "hf_download", args.force_download, "download", model_name=model_name
    ):
        return

    print(f"Downloading {KNOWN_MODELS[args.model]['hf_id']} to {args.hf_download}")

    if args.dry_run:
        return

    snapshot_download(
        KNOWN_MODELS[args.model]["hf_id"],
        allow_patterns=patterns,
        token=True,
        local_dir=args.hf_download,
    )


def _convert(args):
    if "convert" not in KNOWN_MODELS[args.model]:
        return

    if _existing_dir(
        args,
        "tensorrtllm_checkpoint",
        args.force_convert,
        "convert",
        suffix=[args.gpu_name, f"TP_{args.tp_size}"],
    ):
        return

    convert_command = ["python3"]

    convert_command.extend(KNOWN_MODELS[args.model]["convert"])

    convert_command[1] = os.path.join(args.tensorrtllm_example_dir, convert_command[1])

    convert_command.extend(["--model_dir", "{args.hf_download}"])
    convert_command.extend(["--output_dir", "{args.tensorrtllm_checkpoint}"])
    convert_command.extend(["--tp_size", "{args.tp_size}"])

    convert_command = [x.format(args=args) for x in convert_command]

    _call(args, convert_command)


def _build(args):
    if "build" not in KNOWN_MODELS[args.model]:
        return

    if _existing_dir(
        args,
        "tensorrtllm_engine",
        args.force_build,
        "build",
        suffix=[args.gpu_name, f"TP_{args.tp_size}"],
    ):
        return

    build_command = [
        "python3",
        "-m",
        "tensorrt_llm.commands.build",
        "--checkpoint_dir",
        "{args.tensorrtllm_checkpoint}",
        "--output_dir",
        "{args.tensorrtllm_engine}",
        "--max_batch_size",
        args.max_batch_size,
        "--max_num_tokens",
        args.max_num_tokens,
    ]

    build_command.extend(KNOWN_MODELS[args.model]["build"])

    build_command = [x.format(args=args) for x in build_command]

    _call(args, build_command)


def _parse_args():
    parser = argparse.ArgumentParser(description="Prepare Models")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(KNOWN_MODELS.keys()),
        default="llama-3.1-8b-instruct",
        help="model",
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--force-build",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--force-model-repo",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--force-convert",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--target_dir",
        default=TARGET_DIR,
    )

    parser.add_argument(
        "--tensorrtllm_example_dir",
        default=TENSORRTLLM_EXAMPLE_DIR,
    )

    parser.add_argument("--reduce_fusion", default=None, choices=["enable", "disable"])

    parser.add_argument(
        "--enable_chunked_context", default="true", choices=["true", "false"]
    )

    parser.add_argument("--dry-run", action="store_true", default=False)

    parser.add_argument("--tp-size", type=int, default=1)

    parser.add_argument("--max-batch-size", type=int, default=None)

    parser.add_argument("--max-num-tokens", type=int, default=None)

    parser.add_argument("--postprocessing-instance-count", type=int, default=10)

    parser.add_argument("--preprocessing-instance-count", type=int, default=1)

    args = parser.parse_args()

    args.gpu_name = get_gpu_product_name()

    args.hw_name = args.gpu_name
    if args.hw_name is None:
        args.hw_name = "CPU"

    max_batch_size = (
        str(KNOWN_MODELS[args.model]["max_batch_size"])
        if not args.max_batch_size
        else str(args.max_batch_size)
    )

    args.max_batch_size = max_batch_size

    max_num_tokens = (
        str(KNOWN_MODELS[args.model]["max_num_tokens"])
        if not args.max_num_tokens
        else str(args.max_num_tokens)
    )

    args.max_num_tokens = max_num_tokens

    args.participant_ids = ",".join([str(index) for index in range(args.tp_size)])

    if args.reduce_fusion is None:
        args.reduce_fusion = "enable" if args.tp_size > 1 else "disable"

    # args.participant_ids = ""

    return args


if __name__ == "__main__":
    args = _parse_args()
    print(args)

    _download(args)
    _convert(args)
    _build(args)
    _prepare(args)

    print("Your models under GPU type: ", args.gpu_name)
