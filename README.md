<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
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

You can build the Triton Distributed container using the build scripts in `container/`. 

We provide 3 types of builds: 

1. `STANDARD` which includes our default set of backends (onnx, openvino...)
2. `TENSORRTLLM` which includes our TRT-LLM backend
3. `VLLM` which includes our VLLM backend

For example, if you want to build a container for the `VLLM` backend you can run 

`./container/build.sh --framework VLLM`

Please see the instructions in the corresponding example for specific build instructions.

<!--

## Goals

## Concepts

## Examples

-->
