<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Triton Distributed Python Bindings

Python bindings for the Triton distributed runtime system, enabling distributed computing capabilities for machine learning workloads.

## 🚀 Quick Start

1. Install `uv`: https://docs.astral.sh/uv/#getting-started
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install `protoc` protobuf compiler: https://grpc.io/docs/protoc-installation/.

For example on an Ubuntu/Debian system:
```
apt install protobuf-compiler
```

3. Setup a virtualenv
```
cd python-wheels/triton-distributed
uv venv
source .venv/bin/activate
uv pip install maturin
```

4. Build and install triton_distributed wheel
```
maturin develop --uv
```

# Run Examples

## Pre-requisite

See [README.md](../README.md).

## Hello World Example

1. Start 3 separate shells, and activate the virtual environment in each
```
cd python-wheels/triton-distributed
source .venv/bin/activate
```

2. In one shell (shell 1), run example server the instance-1
```
python3 ./examples/hello_world/server.py
```

3. (Optional) In another shell (shell 2), run example the server instance-2
```
python3 ./examples/hello_world/server.py
```

4. In the last shell (shell 3), run the example client:
```
python3 ./examples/hello_world/client.py
```

If you run the example client in rapid succession, and you started more than
one server instance above, you should see the requests from the client being
distributed across the server instances in each server's output. If only one
server instance is started, you should see the requests go to that server
each time.
