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


KNOWN_MODELS = {
    "mock": {
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "download_patterns": ["*.json"],
        "max_num_tokens": 2048,
        "max_batch_size": 512,
        "templates": [
            "preprocessing",
            "postprocessing",
            "ensemble",
            (
                "/workspace/examples/python/llm/tensorrtllm/operators/triton_core_models/mock",
                "context",
            ),
            (
                "/workspace/examples/python/llm/tensorrtllm/operators/triton_core_models/mock",
                "generate",
            ),
            (
                "/workspace/examples/python/llm/tensorrtllm/operators/triton_core_models/mock",
                "tensorrt_llm",
            ),
        ],
        "template_arguments": {
            "tokenizer_dir": "{args.hf_download}",
            "triton_max_batch_size": "{args.max_batch_size}",
            "preprocessing_instance_count": "{args.preprocessing_instance_count}",
            "postprocessing_instance_count": "{args.postprocessing_instance_count}",
            "context_token_latency_ms": "0.1",
            "generate_token_latency_ms": "0.5",
        },
    },
    "llama-3.1-70b-instruct": {
        "hf_id": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "download_model_name": "llama-3.1-70b-instruct",
        "convert": [
            "quantization/quantize.py",
            "--dtype",
            "float16",
            "--qformat",
            "fp8",
            "--calib_size",
            "512",
            "--kv_cache_dtype",
            "fp8",
        ],
        "build": [
            "--gpt_attention_plugin",
            "float16",
            "--max_seq_len",
            "131072",
            "--use_fused_mlp",
            "enable",
            "--reduce_fusion",
            "disable",
            "--multiple_profiles",
            "enable",
            "--use_paged_context_fmha",
            "enable",
        ],
        "max_num_tokens": 2048,
        "max_batch_size": 512,
        "templates": [
            "preprocessing",
            "postprocessing",
            "ensemble",
            ("tensorrt_llm", "context"),
            ("tensorrt_llm", "generate"),
            "tensorrt_llm",
        ],
        "template_arguments": {
            "triton_max_batch_size": "{args.max_batch_size}",
            "decoupled_mode": "True",
            "preprocessing_instance_count": "{args.preprocessing_instance_count}",
            "postprocessing_instance_count": "{args.postprocessing_instance_count}",
            "triton_backend": "tensorrtllm",
            "enable_chunked_context": "{args.enable_chunked_context}",
            "max_beam_width": "1",
            "engine_dir": "{args.tensorrtllm_engine}",
            "exclude_input_in_output": "True",
            "enable_kv_cache_reuse": "True",
            "batching_strategy": "inflight_fused_batching",
            "max_queue_delay_microseconds": "0",
            "max_queue_size": "0",
            "participant_ids": "{args.participant_ids}",
            "tokenizer_dir": "{args.hf_download}",
            "encoder_input_features_data_type": "TYPE_FP16",
        },
    },
    "llama-3.1-8b-instruct": {
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "download_model_name": "llama-3.1-8b-instruct",
        "convert": ["llama/convert_checkpoint.py", "--dtype", "float16"],
        "build": [
            "--remove_input_padding",
            "enable",
            "--gpt_attention_plugin",
            "float16",
            "--context_fmha",
            "enable",
            "--gemm_plugin",
            "float16",
            "--paged_kv_cache",
            "enable",
            "--use_paged_context_fmha",
            "enable",
        ],
        "max_num_tokens": 16384,
        "max_batch_size": 64,
        "templates": [
            "preprocessing",
            "postprocessing",
            "ensemble",
            ("tensorrt_llm", "context"),
            ("tensorrt_llm", "generate"),
            "tensorrt_llm",
        ],
        "template_arguments": {
            "triton_max_batch_size": "{args.max_batch_size}",
            "decoupled_mode": "True",
            "preprocessing_instance_count": "{args.preprocessing_instance_count}",
            "postprocessing_instance_count": "{args.postprocessing_instance_count}",
            "triton_backend": "tensorrtllm",
            "max_beam_width": "1",
            "engine_dir": "{args.tensorrtllm_engine}",
            "exclude_input_in_output": "True",
            "enable_kv_cache_reuse": "True",
            "batching_strategy": "inflight_fused_batching",
            "max_queue_delay_microseconds": "0",
            "max_queue_size": "0",
            "participant_ids": "0",
            "tokenizer_dir": "{args.hf_download}",
            "encoder_input_features_data_type": "TYPE_FP16",
        },
    },
    "llama-3-8b-instruct-generate": {
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "max_batch_size": 256,
        "model_repo_name": "llama-3-8b-instruct-disaggregated",
        "download_model_name": "llama-3-8b-instruct",
        "convert": [
            "quantization/quantize.py",
            "--dtype",
            "float16",
            "--qformat",
            "fp8",
            "--calib_size",
            "512",
            "--kv_cache_dtype",
            "fp8",
        ],
        "build": [
            "--gpt_attention_plugin",
            "float16",
            "--workers",
            "{args.tp_size}",
            "--max_seq_len",
            "1024",
            "--use_fused_mlp",
            "enable",
            "--multiple_profiles",
            "enable",
            "--use_paged_context_fmha",
            "enable",
        ],
        "max_num_tokens": 256,
        "templates": [
            ("tensorrt_llm", "generate"),
            "postprocessing",
        ],
        "template_arguments": {
            "triton_max_batch_size": "{args.max_batch_size}",
            "decoupled_mode": "True",
            "preprocessing_instance_count": "{args.preprocessing_instance_count}",
            "postprocessing_instance_count": "{args.postprocessing_instance_count}",
            "triton_backend": "tensorrtllm",
            "max_beam_width": "1",
            "engine_dir": "{args.tensorrtllm_engine}",
            "exclude_input_in_output": "True",
            "enable_kv_cache_reuse": "True",
            "batching_strategy": "inflight_fused_batching",
            "max_queue_delay_microseconds": "0",
            "max_queue_size": "0",
            "participant_ids": "0",
            "tokenizer_dir": "{args.hf_download}",
            "encoder_input_features_data_type": "TYPE_FP16",
        },
    },
    "llama-3-8b-instruct-context": {
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "max_batch_size": 256,
        "model_repo_name": "llama-3-8b-instruct-disaggregated",
        "download_model_name": "llama-3-8b-instruct",
        "convert": [
            "quantization/quantize.py",
            "--dtype",
            "float16",
            "--qformat",
            "fp8",
            "--calib_size",
            "512",
            "--kv_cache_dtype",
            "fp8",
        ],
        "build": [
            "--gpt_attention_plugin",
            "float16",
            "--workers",
            "{args.tp_size}",
            "--max_seq_len",
            "8192",
            "--use_fused_mlp",
            "enable",
            "--multiple_profiles",
            "enable",
            "--use_paged_context_fmha",
            "enable",
        ],
        "max_num_tokens": 8192,
        "templates": [
            "/workspace/examples/disaggregated_serving/tensorrtllm_templates/context",
            "preprocessing",
        ],
        "template_arguments": {
            "triton_max_batch_size": "{args.max_batch_size}",
            "decoupled_mode": "False",
            "preprocessing_instance_count": "{args.preprocessing_instance_count}",
            "postprocessing_instance_count": "{args.postprocessing_instance_count}",
            "triton_backend": "tensorrtllm",
            "max_beam_width": "1",
            "engine_dir": "{args.tensorrtllm_engine}",
            "exclude_input_in_output": "True",
            "enable_kv_cache_reuse": "True",
            "batching_strategy": "inflight_fused_batching",
            "max_queue_delay_microseconds": "0",
            "max_queue_size": "0",
            "participant_ids": "0",
            "tokenizer_dir": "{args.hf_download}",
            "encoder_input_features_data_type": "TYPE_FP16",
        },
    },
    "llama-3-8b-instruct": {
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "convert": [
            "quantization/quantize.py",
            "--dtype",
            "float16",
            "--qformat",
            "fp8",
            "--calib_size",
            "512",
            "--kv_cache_dtype",
            "fp8",
        ],
        "build": [
            "--gpt_attention_plugin",
            "float16",
            "--workers",
            "{args.tp_size}",
            "--max_seq_len",
            "8192",
            "--use_fused_mlp",
            "enable",
            "--multiple_profiles",
            "enable",
            "--reduce_fusion",
            "{args.reduce_fusion}",
            "--use_paged_context_fmha",
            "enable",
        ],
        "max_num_tokens": 16384,
        "max_batch_size": 512,
        "templates": [
            "preprocessing",
            "postprocessing",
            "ensemble",
            ("tensorrt_llm", "context"),
            ("tensorrt_llm", "generate"),
            "tensorrt_llm",
        ],
        "template_arguments": {
            "triton_max_batch_size": "{args.max_batch_size}",
            "decoupled_mode": "True",
            "preprocessing_instance_count": "{args.preprocessing_instance_count}",
            "postprocessing_instance_count": "{args.postprocessing_instance_count}",
            "triton_backend": "tensorrtllm",
            "max_beam_width": "1",
            "engine_dir": "{args.tensorrtllm_engine}",
            "exclude_input_in_output": "True",
            "enable_kv_cache_reuse": "True",
            "batching_strategy": "inflight_fused_batching",
            "max_queue_delay_microseconds": "0",
            "max_queue_size": "0",
            "participant_ids": "0",
            "tokenizer_dir": "{args.hf_download}",
            "encoder_input_features_data_type": "TYPE_FP16",
        },
    },
    "llama-3-8b-instruct-default": {
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "download_model_name": "llama-3-8b-instruct",
        "convert": ["llama/convert_checkpoint.py", "--dtype", "float16"],
        "build": [
            "--remove_input_padding",
            "enable",
            "--gpt_attention_plugin",
            "float16",
            "--context_fmha",
            "enable",
            "--gemm_plugin",
            "float16",
            "--paged_kv_cache",
            "enable",
            "--use_paged_context_fmha",
            "enable",
        ],
        "max_batch_size": 64,
        "templates": [
            "preprocessing",
            "postprocessing",
            "ensemble",
            ("tensorrt_llm", "context"),
            ("tensorrt_llm", "generate"),
            "tensorrt_llm",
        ],
        "template_arguments": {
            "triton_max_batch_size": "{args.max_batch_size}",
            "decoupled_mode": "True",
            "preprocessing_instance_count": "{args.preprocessing_instance_count}",
            "postprocessing_instance_count": "{args.postprocessing_instance_count}",
            "triton_backend": "tensorrtllm",
            "max_beam_width": "1",
            "engine_dir": "{args.tensorrtllm_engine}",
            "exclude_input_in_output": "True",
            "enable_kv_cache_reuse": "True",
            "batching_strategy": "inflight_fused_batching",
            "max_queue_delay_microseconds": "0",
            "max_queue_size": "0",
            "participant_ids": "0",
            "tokenizer_dir": "{args.hf_download}",
            "encoder_input_features_data_type": "TYPE_FP16",
        },
    },
    "llama-3-70b-instruct-context": {
        "hf_id": "meta-llama/Meta-Llama-3-70B-Instruct",
        "download_model_name": "llama-3-70b-instruct",
        "model_repo_name": "llama-3-70b-disaggegated",
        "max_batch_size": 128,
        "convert": [
            "quantization/quantize.py",
            "--dtype",
            "float16",
            "--qformat",
            "fp8",
            "--calib_size",
            "512",
            "--kv_cache_dtype",
            "fp8",
        ],
        "build": [
            "--gpt_attention_plugin",
            "float16",
            "--workers",
            "{args.tp_size}",
            "--max_seq_len",
            "8192",
            "--use_fused_mlp",
            "enable",
            "--reduce_fusion",
            "{args.reduce_fusion}",
            "--multiple_profiles",
            "enable",
            "--use_paged_context_fmha",
            "enable",
        ],
        "max_num_tokens": 8192,
        "templates": [
            "preprocessing",
            "/workspace/examples/disaggregated_serving/tensorrtllm_templates/context",
        ],
        "template_arguments": {
            "triton_max_batch_size": "{args.max_batch_size}",
            "decoupled_mode": "True",
            "preprocessing_instance_count": "{args.preprocessing_instance_count}",
            "postprocessing_instance_count": "{args.postprocessing_instance_count}",
            "triton_backend": "tensorrtllm",
            "max_beam_width": "1",
            "engine_dir": "{args.tensorrtllm_engine}",
            "exclude_input_in_output": "True",
            "enable_kv_cache_reuse": "True",
            "batching_strategy": "inflight_fused_batching",
            "max_queue_delay_microseconds": "0",
            "max_queue_size": "0",
            "participant_ids": "{args.participant_ids}",
            "tokenizer_dir": "{args.hf_download}",
            "encoder_input_features_data_type": "TYPE_FP16",
        },
    },
    "llama-3-70b-instruct-generate": {
        "hf_id": "meta-llama/Meta-Llama-3-70B-Instruct",
        "download_model_name": "llama-3-70b-instruct",
        "model_repo_name": "llama-3-70b-disaggegated",
        "max_batch_size": 128,
        "convert": [
            "quantization/quantize.py",
            "--dtype",
            "float16",
            "--qformat",
            "fp8",
            "--calib_size",
            "512",
            "--kv_cache_dtype",
            "fp8",
        ],
        "build": [
            "--gpt_attention_plugin",
            "float16",
            "--workers",
            "{args.tp_size}",
            "--max_seq_len",
            "1024",
            "--use_fused_mlp",
            "enable",
            "--reduce_fusion",
            "{args.reduce_fusion}",
            "--multiple_profiles",
            "enable",
            "--use_paged_context_fmha",
            "enable",
        ],
        "max_num_tokens": 128,
        "templates": [
            "postprocessing",
            "/workspace/examples/disaggregated_serving/tensorrtllm_templates/generate",
        ],
        "template_arguments": {
            "triton_max_batch_size": "{args.max_batch_size}",
            "decoupled_mode": "True",
            "preprocessing_instance_count": "{args.preprocessing_instance_count}",
            "postprocessing_instance_count": "{args.postprocessing_instance_count}",
            "triton_backend": "tensorrtllm",
            "max_beam_width": "1",
            "engine_dir": "{args.tensorrtllm_engine}",
            "exclude_input_in_output": "True",
            "enable_kv_cache_reuse": "True",
            "batching_strategy": "inflight_fused_batching",
            "max_queue_delay_microseconds": "0",
            "max_queue_size": "0",
            "participant_ids": "{args.participant_ids}",
            "tokenizer_dir": "{args.hf_download}",
            "encoder_input_features_data_type": "TYPE_FP16",
        },
    },
    "llama-3-70b-instruct": {
        "hf_id": "meta-llama/Meta-Llama-3-70B-Instruct",
        "max_batch_size": 512,
        "convert": [
            "quantization/quantize.py",
            "--dtype",
            "float16",
            "--qformat",
            "fp8",
            "--calib_size",
            "512",
            "--kv_cache_dtype",
            "fp8",
        ],
        "build": [
            "--gpt_attention_plugin",
            "float16",
            "--workers",
            "{args.tp_size}",
            "--max_seq_len",
            "8192",
            "--use_fused_mlp",
            "enable",
            "--reduce_fusion",
            "{args.reduce_fusion}",
            "--multiple_profiles",
            "enable",
            "--use_paged_context_fmha",
            "enable",
        ],
        "max_num_tokens": 16384,
        "templates": [
            "preprocessing",
            "postprocessing",
            "ensemble",
            "tensorrt_llm",
        ],
        "template_arguments": {
            "triton_max_batch_size": "{args.max_batch_size}",
            "decoupled_mode": "True",
            "preprocessing_instance_count": "{args.preprocessing_instance_count}",
            "postprocessing_instance_count": "{args.postprocessing_instance_count}",
            "triton_backend": "tensorrtllm",
            "max_beam_width": "1",
            "engine_dir": "{args.tensorrtllm_engine}",
            "exclude_input_in_output": "True",
            "enable_kv_cache_reuse": "True",
            "batching_strategy": "inflight_fused_batching",
            "max_queue_delay_microseconds": "0",
            "max_queue_size": "0",
            "participant_ids": "{args.participant_ids}",
            "tokenizer_dir": "{args.hf_download}",
            "encoder_input_features_data_type": "TYPE_FP16",
        },
    },
}
