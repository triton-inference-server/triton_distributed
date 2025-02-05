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
[![GitHub Release](https://img.shields.io/github/v/release/triton-inference-server/triton_distributed)](https://github.com/triton-inference-server/triton_distributed/releases/latest)


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

## Pre-requisites

Before building the Triton Distributed containers, ensure the following:

- Docker with [buildx plugin](https://github.com/docker/buildx) is installed on the system. For more information, refer to the Docker documentation.
- Access to the Docker daemon is available from the system.
- GPU Functionality: Run `nvidia-smi` to verify GPU functionality.
- You can get NVIDIA optimized Docker images for Python frameworks from the [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/containers).
- To run model inference on NVIDIA GPU using the Docker runtime, we recommend that you
install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html), which enables GPU acceleration for containers.


## Building Triton Distributed

Triton Distributed development and examples are container based.

You can build the Triton Distributed container using the build scripts
in `container/` (or directly with `docker build`).

We provide 3 types of builds:

1. `STANDARD` which includes our default set of backends (onnx, openvino...)
2. `TENSORRTLLM` which includes our TRT-LLM backend
3. `VLLM` which includes our VLLM backend

For example, if you want to build a container for the `STANDARD` backends you can run

`./container/build.sh`

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

## Hello World

[Hello World](./examples/hello_world)

A basic example demonstrating the new interfaces and concepts of
Triton Distributed. In the Hello World example, you can deploy a set
of simple workers to load balance requests from a local work queue.

## LLM

[TENSORRTLLM](./examples/llm/tensorrtllm)

An intermediate example expanding further on the concepts indroduced
in the Hello World example. In this example, we demonstrate
[Disaggregated Serving](https://arxiv.org/abs/2401.09670)
as an application of the components defined in Triton Distributed.

[VLLM](./examples/llm/vllm)

An intermediate example expanding further on the concepts indroduced
in the Hello World example. In this example, we demonstrate
[Disaggregated Serving](https://arxiv.org/abs/2401.09670)
as an application of the components defined in Triton Distributed.


# Disclaimers

> [!NOTE]
> This project is currently in the alpha / experimental /
> rapid-prototyping stage and we will be adding new features incrementally.

1. The `TENSORRTLLM` and `VLLM` containers are WIP and not expected to
   work out of the box.

2. Testing has primarily been on single node systems with processes
   launched within a single container.
