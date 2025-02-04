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

from typing import Optional

from triton_distributed.icp.nats_request_plane import NatsRequestPlane
from triton_distributed.icp.ucp_data_plane import UcpDataPlane

# UCP data plane causes deadlocks when used more than once, so we use a singleton
_g_singletonic_data_plane = None
_g_singletonic_data_plane_connection_count = 0

_g_actual_host = None
_g_actual_port = None


def set_actual_host_port(host, port):
    global _g_actual_host
    global _g_actual_port
    if _g_singletonic_data_plane is not None:
        raise Exception("Cannot set actual host and port after data plane is created")
    _g_actual_host = host
    _g_actual_port = port


def set_data_plane(data_plane):
    global _g_singletonic_data_plane
    global _g_singletonic_data_plane_connection_count
    _g_singletonic_data_plane_connection_count = 1
    _g_singletonic_data_plane = data_plane


class RemoteConnector:
    """Handle connection to both request and data planes."""

    def __init__(
        self,
        nats_url: str,
        data_plane_host: Optional[str] = None,
        data_plane_port: int = 0,
        keep_dataplane_endpoints_open: bool = False,
    ):
        """Initialize RemoteConnector.

        Args:
            nats_url (str): URL of NATS server.
        """
        global _g_singletonic_data_plane
        global _g_actual_port
        global _g_actual_host
        self._nats_url = nats_url
        self._request_plane = NatsRequestPlane(nats_url)
        if _g_singletonic_data_plane is None:
            if _g_actual_host is not None:
                data_plane_host = _g_actual_host
            if _g_actual_port is not None:
                data_plane_port = _g_actual_port
            _g_singletonic_data_plane = UcpDataPlane(
                hostname=data_plane_host,
                port=data_plane_port,
                keep_endpoints_open=keep_dataplane_endpoints_open,
            )
        self._connected = False
        self._data_plane = _g_singletonic_data_plane

    async def connect(self):
        """Connect to both request and data planes."""
        global _g_singletonic_data_plane
        global _g_singletonic_data_plane_connection_count

        assert _g_singletonic_data_plane is not None

        await self._request_plane.connect()
        if _g_singletonic_data_plane_connection_count == 0:
            _g_singletonic_data_plane.connect()
        _g_singletonic_data_plane_connection_count += 1
        self._connected = True

    async def close(self):
        """Disconnect from both request and data planes."""
        global _g_singletonic_data_plane
        global _g_singletonic_data_plane_connection_count

        assert _g_singletonic_data_plane is not None

        await self._request_plane.close()
        _g_singletonic_data_plane_connection_count -= 1
        if _g_singletonic_data_plane_connection_count == 0:
            _g_singletonic_data_plane.close()
            _g_singletonic_data_plane = None
        self._data_plane.close()
        self._connected = False

    async def __aenter__(self):
        """Enter context manager."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit context manager."""
        await self.close()
