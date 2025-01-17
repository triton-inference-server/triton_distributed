<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Triton Distributed

<h4> A Datacenter Scale Distributed Inference Serving Framework </h4>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Triton Distributed is a flexible, component based, data center scale
inference serving framework designed to leverage the strengths of the
standalone Triton Inference Server while expanding its capabilities
to meet the demands of complex use cases including those of Generative
AI. It is designed to enable developers to implement and customize
routing, load balancing, scaling and workflow definitions at the data
center scale without sacrificing performance or ease of use.

> [!NOTE]
> This project is currently in the alpha / experimental /
> rapid-prototyping stage and we are actively looking for feedback and
> collaborators.

## Building Triton Distributed

Triton Distributed development and examples are container based.

You can build the Triton Distributed container using the build scripts
in `container/` (or directly with `docker build`).

We provide 3 types of builds:

1. `STANDARD` which includes our default set of backends (onnx, openvino...)
2. `TENSORRTLLM` which includes our TRT-LLM backend
3. `VLLM` which includes our VLLM backend

For example, if you want to build a container for the `VLLM` backend you can run

`./container/build.sh --framework VLLM`

Please see the instructions in the corresponding example for specific build instructions.

## Running Triton Distributed for Local Testing and Development

You can run the Triton Distributed container using the run scripts in
`container/` (or directly with `docker run`).

The run script offers a few common workflows:

1. Running a command in a container and exiting.

```
./container/run.sh -- python3 -c "import triton_distributed.icp.protos.icp_pb2 as icp_proto; print(icp_proto); print(dir(icp_proto));"
```

2. Starting an interactive shell.
```
./container/run.sh -it
```

3. Mounting the local workspace and Starting an interactive shell.

```
./container/run.sh -it --mount-workspace
```

The last command also passes common environment variables ( ```-e
HF_TOKEN```) and mounts common directories such as ```/tmp:/tmp```,
```/mnt:/mnt```.

Please see the instructions in the corresponding example for specific
deployment instructions.



## 1. Big Picture
Triton Distributed extends the standard [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) model-serving paradigm with additional “planes” that distribute data and requests across multiple processes or machines. Conceptually, you still write a Triton **Model**, but your inference requests and data transfers can be routed through:

1. **Request Plane**: Sends model-inference requests among nodes or processes.
2. **Data Plane**: Moves tensor data and references between processes or GPUs.

This architecture allows you to build large, multi-process or multi-node solutions for AI model inference without manually managing transport and synchronization. You can share GPU memory references, shift them among distributed processes, etc.

---


## 2. Key Components
The repo has four major logical layers:

1. **ICP (Inter-Component Protocol)**:
   - Python modules under `triton_distributed/icp/…`
   - Encodes how the data and requests get serialized/transported.
   - Implements **NatsRequestPlane** and **UcpDataPlane**, which are concrete transport/connection classes for requests/data.
     - **NatsRequestPlane** uses NATS for distributing requests.
     - **UcpDataPlane** uses UCX (libucp) for transferring tensor data, possibly GPU-to-GPU.

2. **Worker**:
   - Python modules under `triton_distributed/worker/…`
   - Exposes the concept of an **Operator** (a processing node that can serve one or more Triton models or custom logic).
   - Runs the main loop that pulls requests from a Request Plane, processes them, and returns responses.
   - Contains a Python “mini” server (the `Worker`) that spawns or manages multiple Operators.

3. **Integration Tests & Examples**:
   - A directory structure with unit tests and integration tests showing how to compose multiple workers.
   - The “hello world” example is under `examples/hello_world/`.

4. **Triton Python Models**:
   - Under various directories like `.../operators/triton_core_models/...` or `icp/src/python/triton_distributed/icp/...`
   - Typical Triton `model.py` files that define custom Python logic behind each “model.”


## 4. ICP Planes & Worker Internals

### 4.1 Request Plane (NATS)
`NatsRequestPlane` handles distributing requests among processes. Under the hood, it:

- Connects to a NATS server (which might run in local Docker or remote).
- Creates “streams” in NATS for each operator or for direct routing.
- On the “client” side (where you call `post_request`), it publishes request messages to the right NATS subjects.
- On the “server” side (the Worker), it “pulls” requests from NATS subscriptions.

### 4.2 Data Plane (UCX)
`UcpDataPlane` references UCX-Py (libucp) to exchange actual tensor data. By default:

- When you “put” a tensor, the data plane either:
  1. Embeds small data directly in the message (the “contents” approach), or
  2. If large, stores a reference (GPU or CPU memory) in the local `_tensor_store`, then sends a small “URI” like `ucp://hostname:port/<uuid>` to the remote side.
- The remote side can do “get_tensor” by connecting to `ucp://hostname:port` and pulling the data.

This allows distributed GPU memory references with minimal overhead.

### 4.3 Worker
A `Worker` runs in a separate process. It:

- Starts or registers Triton model(s).
- Connects to the chosen request plane (NATS) and data plane (UCX).
- Enters a loop:
  1. `pull_requests` from NATS,
  2. routes them to the correct Operator,
  3. gets the results,
  4. returns them to the request plane.

In the “hello world,” you see three Worker processes—each hosting either the encoder, decoder, or aggregator operator.



<!--

## Goals

## Concepts

## Examples

-->
