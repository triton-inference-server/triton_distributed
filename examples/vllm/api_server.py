#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import asyncio
import logging

import uvicorn
from fastapi import FastAPI
from triton_api_server.open_ai.server import create_app

from triton_distributed.worker.remote_model_connector import RemoteModelConnector

LOGGER = logging.getLogger(__name__)

app = FastAPI()

# TODO: how to pass default sampling parameters to the endpoint
# TODO: how to handle different message formats despite changing of endpoint type
# TODO: ensure errors are transmitted to the client
# TODO: pass request_id to the InferenceRequest

parser = argparse.ArgumentParser(
    description="Run the Triton API server with OpenAI endpoint"
)

parser.add_argument(
    "--nats-url",
    type=str,
    required=False,
    default="nats://localhost:4222",
    help="URL of NATS server",
)

parser.add_argument(
    "--data-plane-host",
    type=str,
    required=False,
    default=None,
    help="Data plane host",
)

parser.add_argument(
    "--data-plane-port",
    type=int,
    required=False,
    default=0,
    help="Data plane port. (default: 0 means the system will choose a port)",
)

parser.add_argument(
    "--model-name",
    type=str,
    required=False,
    default="prefill",
    help="Model name",
)

parser.add_argument(
    "--fastapi-host",
    type=str,
    required=False,
    default="127.0.0.1",
    help="FastAPI host",
)

parser.add_argument(
    "--fastapi-port",
    type=int,
    required=False,
    default=8000,
    help="FastAPI port",
)

parser.add_argument(
    "--log-level",
    type=str,
    required=False,
    default="info",
    help="Logging level (e.g., debug, info, warning, error, critical)",
)

parser.add_argument(
    "--log-format",
    type=str,
    required=False,
    default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    help="Logging format",
)


async def start_http_server(args):
    logging.basicConfig(level=args.log_level.upper(), format=args.log_format)

    triton_connector = RemoteModelConnector(
        nats_url=args.nats_url,
        data_plane_host=args.data_plane_host,
        data_plane_port=args.data_plane_port,
        model_name=args.model_name,
        keep_dataplane_endpoints_open=True,
    )
    LOGGER.info(f"Start API server {args}")
    async with triton_connector:
        config = uvicorn.Config(
            app,
            host=args.fastapi_host,
            port=args.fastapi_port,
            log_level=args.log_level.lower(),
        )
        server = uvicorn.Server(config)
        create_app(triton_connector, app)
        LOGGER.info("Start HTTP endpoint")
        await server.serve()
        LOGGER.info("End HTTP endpoint")
    LOGGER.info("Triton connector closed")


if __name__ == "__main__":
    args = parser.parse_args()
    asyncio.run(start_http_server(args))
