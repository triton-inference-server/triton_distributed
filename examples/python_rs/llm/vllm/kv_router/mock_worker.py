# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import asyncio
import uuid

import uvloop
import vllm
from common.parser import parse_vllm_args
from common.protocol import Request, Response
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import logger as vllm_logger
from triton_distributed_rs import (
    DistributedRuntime,
    KvRouter,
    KvMetricsPublisher,
    triton_endpoint,
    triton_worker,
)

import ctypes
from ctypes import c_char_p, c_uint32, c_void_p, c_size_t, c_int64

class TritonResult:
    OK = 0
    ERR = 1


class MockEngine:
    """
    Request handler for the generate endpoint
    """

    def __init__(self, metrics_publisher, worker_id):
        self.worker_id = worker_id
        # KV events
        self.lib = ctypes.CDLL("/opt/triton/llm_binding/lib/libtriton_llm_capi.so")
        self.lib.triton_llm_init.argtypes = [c_char_p, c_char_p, c_int64]
        self.lib.triton_llm_init.restype = c_uint32
        result = self.lib.triton_llm_init("triton-init".encode(), "vllm".encode(), worker_id)
        if result == TritonResult.OK:
            vllm_logger.info("KVCacheEventManager initialized successfully. Ready to publish KV Cache Events")
        else:
            vllm_logger.info("KVCacheEventManager initialization failed!")
        self.lib.triton_kv_event_publish_stored.argtypes = [
            ctypes.c_uint64,                    # event_id
            ctypes.POINTER(ctypes.c_uint32),    # token_ids
            ctypes.POINTER(ctypes.c_size_t),    # num_block_tokens
            ctypes.POINTER(ctypes.c_uint64),    # block_ids
            ctypes.c_size_t,                    # num_blocks
            ctypes.POINTER(ctypes.c_uint64),    # parent_hash
            ctypes.c_uint64,                    # lora_id
        ]
        self.lib.triton_kv_event_publish_stored.restype = ctypes.c_uint32  # triton_llm_result_t

        self.lib.triton_kv_event_publish_removed.argtypes = [
            ctypes.c_uint64,                    # event_id
            ctypes.POINTER(ctypes.c_uint64),    # block_ids
            ctypes.c_size_t,                    # num_blocks
        ]
        self.lib.triton_kv_event_publish_removed.restype = ctypes.c_uint32  # triton_llm_result_t

        # KV metrics
        self.metrics_publisher = metrics_publisher

        self.request_active_slots = 0
        self.request_total_slots = 4
        self.kv_active_block = 0
        self.kv_total_blocks = 4
        # [NOTE] Now that the component must has proper metrics reported
        # to be properly selected by the router
        self.metrics_publisher.publish(self.request_active_slots, self.request_total_slots, self.kv_active_block, self.kv_total_blocks, worker_id)


    @triton_endpoint(Request, Response)
    async def generate(self, request):
        print(f"Received request: {request}")
        self.request_active_slots = min(self.request_active_slots + 1, self.request_total_slots)
        self.kv_active_block = min(self.kv_active_block + 1, self.kv_total_blocks)
        self.metrics_publisher.publish(self.request_active_slots, self.request_total_slots, self.kv_active_block, self.kv_total_blocks, self.worker_id)
        yield "Hello, World!"

    async def cooldown(self):
        while True:
            await asyncio.sleep(5)
            self.request_active_slots = max(0, self.request_active_slots - 1)
            self.kv_active_block = max(0, self.kv_active_block - 1)
            # [FIXME] worker_id is not needed as endpoint collection
            # contains the (lease) id
            self.metrics_publisher.publish(self.request_active_slots, self.request_total_slots, self.kv_active_block, self.kv_total_blocks, self.worker_id)

@triton_worker()
async def worker(runtime: DistributedRuntime):
    """
    Instantiate a `backend` component and serve the `generate` endpoint
    A `Component` can serve multiple endpoints
    """
    component = runtime.namespace("triton-init").component("vllm")
    metrics_publisher = KvMetricsPublisher()
    await metrics_publisher.create_service(component)

    endpoint = component.endpoint("generate")
    engine = MockEngine(metrics_publisher, endpoint.lease_id())
    await asyncio.gather(
        engine.cooldown(),
        endpoint.serve_endpoint(engine.generate),
    )


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
