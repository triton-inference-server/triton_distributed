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
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Type

from pydantic import BaseModel, ValidationError
from triton_distributed_rs._core import DistributedRuntime
from triton_distributed_rs._core import KvRouter as KvRouter
from triton_distributed_rs._core import KvMetricsPublisher as KvMetricsPublisher


def triton_worker():
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_running_loop()
            runtime = DistributedRuntime(loop)

            await func(runtime, *args, **kwargs)

            # # wait for one of
            # # 1. the task to complete
            # # 2. the task to be cancelled

            # done, pending = await asyncio.wait({task, cancelled}, return_when=asyncio.FIRST_COMPLETED)

            # # i want to catch a SIGINT or SIGTERM or a cancellation event here

            # try:
            #     # Call the actual function
            #     return await func(runtime, *args, **kwargs)
            # finally:
            #     print("Decorator: Cleaning up runtime resources")
            #     # Perform cleanup actions here

        return wrapper

    return decorator


def triton_endpoint(
    request_model: Type[BaseModel], response_model: Type[BaseModel]
) -> Callable:
    def decorator(
        func: Callable[..., AsyncGenerator[Any, None]]
    ) -> Callable[..., AsyncGenerator[Any, None]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> AsyncGenerator[Any, None]:
            # Validate the request
            try:
                if len(args) in [1, 2]:
                    args = list(args)
                    if isinstance(args[-1], str):
                        args[-1] = request_model.parse_raw(args[-1])
                    elif isinstance(args[-1], dict):
                        args[-1] = request_model.parse_obj(args[-1])
                    else:
                        raise ValueError(f"Invalid request: {args[-1]}")
            except ValidationError as e:
                raise ValueError(f"Invalid request: {e}")

            # Wrap the async generator
            async for item in func(*args, **kwargs):
                # Validate the response
                # TODO: Validate the response
                try:
                    yield item
                except ValidationError as e:
                    raise ValueError(f"Invalid response: {e}")

        return wrapper

    return decorator
