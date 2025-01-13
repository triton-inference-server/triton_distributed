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

# A script to download a Python wheel, patch it, copy additional files,
# repackage it, and optionally install the new wheel.

import typing

if typing.TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
    from vllm.attention.backends.abstract import AttentionMetadata

import hashlib
import uuid
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed

from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.attention.backends.flashinfer import FlashInferMetadata, FlashInferBackend
from vllm.logger import init_logger
from vllm.distributed.parallel_state import get_tp_group, get_store
import vllm.envs as envs
from vllm import _custom_ops as ops

from vllm.distributed.data_plane import NcclDataPlane, UcpDataPlane

logger = init_logger(__name__)


_kv_cache_handler = None


def get_kv_cache_handler():
    global _kv_cache_handler
    if _kv_cache_handler is None:
        _kv_cache_handler = KVCacheHandler()
    return _kv_cache_handler


class KVCacheHandler:

    def __init__(self):
        if _kv_cache_handler is not None:
            raise ValueError("KVCacheHandler is a singleton")
        
        self._data_plane_backend = envs.VLLM_DATA_PLANE_BACKEND
        if self._data_plane_backend == "nccl":
            self._data_plane = NcclDataPlane()
            self._store = get_store()
            logger.info("Store set up")
            self._store.set(f"worker_{envs.VLLM_WORKER_ID}_rank_{get_tp_group().local_rank}", f"{self._data_plane._hostname}:{self._data_plane._port}")
        elif self._data_plane_backend == "ucx":
            self._data_plane = UcpDataPlane(keep_endpoints_open=True)
            self._data_plane.connect()
            rank = torch.distributed.get_rank()
            is_master = envs.VLLM_WORKER_ID == 0 and rank == 0
            self._store = torch.distributed.TCPStore(envs.VLLM_TORCH_HOST, envs.VLLM_TORCH_PORT, is_master=is_master)
            self._store.set(f"worker_{envs.VLLM_WORKER_ID}_rank_{rank}", f"{self._data_plane.hostname}:{self._data_plane.port}")
        else:
            raise ValueError(f"Unknown data plane backend {self._data_plane_backend}")
        self._local_store = {}

        self.transport_thread = ThreadPoolExecutor(max_workers=1)
        logger.info("KVCacheHandler initialized")                
    
    def send(
        self,
        model: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: typing.List[torch.Tensor],
    ):  
        with torch.cuda.nvtx.range("KV send"):
            self._send(
                model,
                model_input,
                kv_caches,
            )             

    def _send(
        self,
        model: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: typing.List[torch.Tensor],
    ):

        seq_lens = model_input.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
        start_layer = model.start_layer
        end_layer = model.end_layer
        request_ids = list(model_input.request_ids_to_seq_ids.keys())

        _, _, block_size, num_heads, _ = kv_caches[0].shape
        attention_backend = _get_attention_backend(model_input.attn_metadata)

        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            request_id = request_ids[idx]

            keys, values = [], []

            logger.debug(f"seq_len {slen}, start_pos {start_pos}, end_pos {end_pos}, slot_mapping_flat {slot_mapping_flat.shape}")
            current_slot_mapping = slot_mapping_flat[start_pos:end_pos]
            if len(model_input.attn_metadata.block_tables[idx]) > 0:
                block_inds = torch.range(0, block_size - 1, device=current_slot_mapping.device, dtype=current_slot_mapping.dtype)
                additional_inds = torch.cat([block_inds + block_size * i for i in model_input.attn_metadata.block_tables[idx]]).to(current_slot_mapping.device).to(current_slot_mapping.dtype)
                logger.debug(f"additional_inds: {additional_inds.shape}")
                logger.debug(f"current_slot_mapping: {current_slot_mapping.shape}")
                current_slot_mapping = torch.cat([additional_inds, current_slot_mapping])
            logger.debug(f"new current_slot_mapping: {current_slot_mapping.shape}")

            current_slot_mapping_quotient = current_slot_mapping // block_size
            current_slot_mapping_remainder = current_slot_mapping % block_size

            logger.debug("kv_caches shape: %s", kv_caches[0].shape)

            for layer_id in range(start_layer, end_layer):
                kv_cache = kv_caches[layer_id - start_layer]
                if attention_backend == "flash_attn":
                    key_cache = kv_cache[
                        0, current_slot_mapping_quotient, current_slot_mapping_remainder
                    ]
                    value_cache = kv_cache[
                        1, current_slot_mapping_quotient, current_slot_mapping_remainder
                    ]
                elif attention_backend == "flash_infer":
                    key_cache = kv_cache[
                        current_slot_mapping_quotient, 0, current_slot_mapping_remainder
                    ]
                    value_cache = kv_cache[
                        current_slot_mapping_quotient, 1, current_slot_mapping_remainder
                    ]
                else:
                    raise ValueError(f"Unknown attention backend {attention_backend}")
                keys.append(key_cache)
                values.append(value_cache)

            keys = torch.stack(keys, dim=0)
            values = torch.stack(values, dim=0)


            tp_multipler = envs.VLLM_GENERATE_TP_SIZE // envs.VLLM_CONTEXT_TP_SIZE
            first_rank = envs.VLLM_CONTEXT_WORKERS * envs.VLLM_CONTEXT_TP_SIZE
            for i in range(tp_multipler):
                num_heads_per_generate_rank = num_heads // tp_multipler
                first_head = i * num_heads_per_generate_rank
                partial_keys = keys[
                    :, :, first_head : first_head + num_heads_per_generate_rank, :
                ].clone()
                partial_values = values[
                    :, :, first_head : first_head + num_heads_per_generate_rank, :
                ].clone()
                target_local_rank = get_tp_group().local_rank * tp_multipler + i
                target_rank = target_local_rank + first_rank
                # torch.cuda.synchronize()
                self._send_tensors(request_id, target_rank, target_local_rank, partial_keys, partial_values)
            logger.debug("Tensors sent")

        logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())


    def recv(
        self,
        model: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: typing.List[torch.Tensor],
    ):
        with torch.cuda.nvtx.range("KV recv"):
            self._recv(
                model.start_layer,
                model.end_layer,
                [model.layers[i].self_attn.attn.kv_cache_dtype for i in range(model.start_layer, model.end_layer)],
                [model.layers[i].self_attn.attn._k_scale for i in range(model.start_layer, model.end_layer)],
                [model.layers[i].self_attn.attn._v_scale for i in range(model.start_layer, model.end_layer)],
                model_input,
                kv_caches,
            )


    def _recv(
        self,
        start_layer: int,
        end_layer: int,
        kv_cache_dtypes: typing.List[torch.dtype],
        k_scales: typing.List[float],
        v_scales: typing.List[float],
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: typing.List[torch.Tensor],
    ):
        seq_lens = model_input.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
        request_ids = list(model_input.request_ids_to_seq_ids.keys())

        _, _, block_size, num_heads, head_dim = kv_caches[0].shape
        attention_backend = _get_attention_backend(model_input.attn_metadata)   

        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            request_id = request_ids[idx]
            num_tokens = slen

            base_request_id, context_worker_id = request_id.split("___")
            context_worker_id = int(context_worker_id)

            keys, values = self._recv_tensors(base_request_id, context_worker_id, num_tokens, end_layer - start_layer, num_heads, head_dim)

            logger.debug(f"Received tensors for request_id {request_id}")

            if kv_caches[0].dtype == torch.uint8:
                torch_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer("fp8_e4m3")
                keys = keys.view(torch_dtype).to(torch.bfloat16)
                values = values.view(torch_dtype).to(torch.bfloat16)
                logger.debug(f"Converted caches to torch.blfoat16")

            for i in range(start_layer, end_layer):
                kv_cache = kv_caches[i - start_layer]
                key, value = keys[i], values[i]

                if attention_backend == "flash_attn":
                    key_cache, value_cache = kv_cache[0], kv_cache[1]
                elif attention_backend == "flash_infer":
                    key_cache, value_cache = kv_cache[:, 0], kv_cache[:, 1]
                else:
                    raise ValueError(f"Unknown attention backend {attention_backend}")
                ops.reshape_and_cache_flash(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    slot_mapping_flat[start_pos:end_pos],
                    kv_cache_dtypes[i],
                    k_scales[i],
                    v_scales[i],
                )

        logger.debug(f"KV receive DONE for rank {torch.distributed.get_rank()}")

    def _send_tensors(self, request_id, target_rank, target_local_rank, keys, values):
        logger.debug(f"Sending tensors for request_id {request_id} to rank {target_rank}")
        logger.debug(f"Tensor shapes: keys {keys.shape}, values {values.shape}")
        logger.debug(f"Tensor dtypes: keys {keys.dtype}, values {values.dtype}")
        
        if self._data_plane_backend == "nccl":
            self._send_tensors_nccl(request_id, target_rank, target_local_rank, keys, values)
        elif self._data_plane_backend == "ucx":
            self._send_tensors_ucx(request_id, target_local_rank, keys, values)

    def _send_tensors_nccl(self, request_id, target_rank, target_local_rank, keys, values):
        generate_worker_id = envs.VLLM_CONTEXT_WORKERS
        target_addr = self._store.get(f"worker_{generate_worker_id}_rank_{target_local_rank}").decode()
        self._data_plane.put_output_tensor(keys, rank=target_rank, tensor_id=f"{request_id}_keys_rank{target_local_rank}", remote_address=target_addr)
        self._data_plane.put_output_tensor(values, rank=target_rank, tensor_id=f"{request_id}_values_rank{target_local_rank}", remote_address=target_addr)

    def _send_tensors_ucx(self, request_id, target_local_rank, keys, values):
        torch.cuda.synchronize()
        self._data_plane.put_output_tensor(
            keys, _create_id_from_str(f"{request_id}_keys_rank{target_local_rank}"),
        )
        self._data_plane.put_output_tensor(
            values, _create_id_from_str(f"{request_id}_values_rank{target_local_rank}"),
        )

    def _recv_tensors(self, request_id, context_worker_id, num_tokens, num_layers, num_heads, head_dim):
        logger.debug(f"Receiving tensors for request_id {request_id} from worker {context_worker_id}")
        
        if self._data_plane_backend == "nccl":
            return self._recv_tensors_nccl(request_id, context_worker_id, num_tokens, num_layers, num_heads, head_dim)
        elif self._data_plane_backend == "ucx":
            return self._recv_tensors_ucx(request_id, context_worker_id, num_tokens, num_layers, num_heads, head_dim)

    def _recv_tensors_nccl(self, request_id, context_worker_id, num_tokens, num_layers, num_heads, head_dim):
        tp_rank = get_tp_group().local_rank
        tp_multipler = envs.VLLM_GENERATE_TP_SIZE // envs.VLLM_CONTEXT_TP_SIZE
        source_tp_rank = tp_rank // tp_multipler
        source_rank = context_worker_id * envs.VLLM_CONTEXT_TP_SIZE + source_tp_rank
        worker_key = f"worker_{context_worker_id}_rank_{source_tp_rank}"
        source_addr = self._local_store.get(worker_key)
        if source_addr is None:
            logger.info("Fetching source address for worker %d by key %s", context_worker_id, worker_key)
            source_addr = self._store.get(worker_key).decode()
            self._local_store[worker_key] = source_addr
        keys = self._data_plane.get_tensor(rank=source_rank, tensor_id=f"{request_id}_keys_rank{tp_rank}", remote_address=source_addr)
        values = self._data_plane.get_tensor(rank=source_rank, tensor_id=f"{request_id}_values_rank{tp_rank}", remote_address=source_addr)
        return keys, values

    def _recv_tensors_ucx(self, request_id, context_worker_id, num_tokens, num_layers, num_heads, head_dim):
        local_rank = get_tp_group().local_rank
        tp_rank = get_tp_group().local_rank
        tp_multipler = envs.VLLM_GENERATE_TP_SIZE // envs.VLLM_CONTEXT_TP_SIZE
        source_tp_rank = tp_rank // tp_multipler
        source_addr = self._store.get(f"worker_{context_worker_id}_rank_{source_tp_rank}").decode()

        keys_id = _create_id_from_str(f"{request_id}_keys_rank{local_rank}")
        keys_uri = f"ucp://{source_addr}/{keys_id}"
        keys = self._data_plane.get_tensor(
            keys_uri,
            (num_layers, num_tokens, num_heads, head_dim),
            torch.uint8,
            device_id=local_rank,
        )

        values_id = _create_id_from_str(f"{request_id}_values_rank{local_rank}")
        values_uri = f"ucp://{source_addr}/{values_id}"
        values = self._data_plane.get_tensor(
            values_uri,
            (num_layers, num_tokens, num_heads, head_dim),
            torch.uint8,
            device_id=local_rank,
        )

        return keys, values


def _get_attention_backend(attn_metadata: "AttentionMetadata") -> str:
    if isinstance(attn_metadata, FlashAttentionMetadata):
        return "flash_attn"
    elif isinstance(attn_metadata, FlashInferMetadata):
        return "flash_infer"
    else:
        raise ValueError(
            f"Unknown attention metadata type {type(attn_metadata)}. Only FlashAttentionMetadata and FlashInferMetadata are supported."
        )


def _create_id_from_str(
    str_id: str
) -> uuid.UUID:
    # Create a hash of the str string using SHA-1
    hashed_key = hashlib.sha1(str_id.encode("utf-8"))
    # Generate a UUID from the hash, ensuring it's in the correct format
    hash_hex = hashed_key.hexdigest()[:32]  # Get first 32 characters
    uuid_generated = uuid.UUID(hash_hex)
    return uuid_generated
