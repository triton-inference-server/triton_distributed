#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from compoundai import api
from compoundai.sdk.decorators import nova_endpoint
from compoundai.sdk.dependency import depends
from compoundai.sdk.service import service

"""
Pipeline Architecture:

Users/Clients (HTTP)
      │
      ▼
┌─────────────┐
│  Frontend   │  HTTP API endpoint (/generate)
└─────────────┘
      │ nova/distributed-runtime
      ▼
┌─────────────┐
│   Middle    │
└─────────────┘
      │ nova/distributed-runtime
      ▼
┌─────────────┐
│  Backend    │
└─────────────┘
"""


@service(
    resources={"cpu": "2"},
    traffic={"timeout": 30},
    nova={
        "enabled": True,
        "namespace": "inference",
    }
)
class Backend:
    def __init__(self) -> None:
        print("Starting backend")

    @nova_endpoint()
    async def generate(self, text: str):
        """Generate tokens."""
        text = f"{text}-back"
        print(f"Backend received: {text}")
        for token in text.split():
            yield f"Backend: {token}"


@service(
    resources={"cpu": "2"},
    traffic={"timeout": 30},
    nova={
        "enabled": True,
        "namespace": "inference"
    }
)
class Middle:
    backend = depends(Backend)

    def __init__(self) -> None:
        print("Starting middle")

    @nova_endpoint()
    async def generate(self, text: str):
        """Forward requests to backend."""
        text = f"{text}-mid"
        print(f"Middle received: {text}")
        async for response in self.backend.generate(text):
            print(f"Middle received response: {response}")
            yield f"Middle: {response}"


@service(
    resources={"cpu": "1"},
    traffic={"timeout": 60}  # Regular HTTP API
)
class Frontend:
    middle = depends(Middle)

    def __init__(self) -> None:
        print("Starting frontend")

    @api
    async def generate(self, text: str):
        """Stream results from the pipeline."""
        print(f"Frontend received: {text}")
        async for response in self.middle.generate(text):
            yield f"Frontend: {response}"
