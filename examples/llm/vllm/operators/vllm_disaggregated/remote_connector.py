# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from typing import Optional


from triton_distributed.icp.request_plane import RequestPlane
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
        request_plane: RequestPlane,
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
        self._request_plane = request_plane
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
        assert _g_singletonic_data_plane
        if _g_singletonic_data_plane_connection_count == 0:
            _g_singletonic_data_plane.connect()
        _g_singletonic_data_plane_connection_count += 1
        self._connected = True

    async def close(self):
        """Disconnect from both request and data planes."""
        global _g_singletonic_data_plane
        global _g_singletonic_data_plane_connection_count
        assert _g_singletonic_data_plane
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
