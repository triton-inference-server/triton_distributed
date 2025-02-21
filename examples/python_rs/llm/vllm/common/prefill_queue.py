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


from typing import Optional

from common.protocol import PrefillRequest
from triton_distributed_rs.nats_queue import NATSQueue


class PrefillQueue(NATSQueue):
    """
    A wrapper of NATSQueue for PrefillRequest.
    The stream name is forced to be "prefill_queue".
    """

    def __init__(
        self,
        stream_name="prefill_queue",
        nats_server: str = "nats://localhost:4222",
        dequeue_timeout: float = 1,
    ):
        super().__init__(
            # force set stream_name to "prefill_queue"
            stream_name="prefill_queue",
            nats_server=nats_server,
            dequeue_timeout=dequeue_timeout,
        )

    async def enqueue_prefill_request(self, prefill_request: PrefillRequest) -> None:
        raw_request = prefill_request.model_dump_json()
        await self.enqueue_task(raw_request)

    async def dequeue_prefill_request(self) -> Optional[PrefillRequest]:
        raw_request = await self.dequeue_task()
        if raw_request is not None:
            prefill_request = PrefillRequest.model_validate_json(raw_request)
            return prefill_request
        else:
            return None
