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

# Hello World

A basic example demonstrating the new interfaces and concepts of
triton distributed. In the hello world example, you can deploy a set
of simple workers to load balance requests from a local work queue.

The example demonstrates:

1. How to incorporate an existing Triton Core Model into a triton distributed worker.
1. How to incorporate a standalone python class into a triton distributed worker.
1. How deploy a set of workers
1. How to send requests to the triton distributed deployment


## Building the Hello World Environment

The hello world example is designed to be deployed in a containerized
environment and to work with and without GPU support.

To get started build the "STANDARD" triton distributed development
environment.

Note: "STANDARD" is the default framework

```
./containers/build.sh
```


## Starting the Deployment

```
./containers/run.sh -it -- python3 -m hello_world.deploy --initialize-request-plane
```

#### Expected Output


```
Starting Workers
17:17:09 deployment.py:115[triton_distributed.worker.deployment] INFO:

Starting Worker:

	Config:
	WorkerConfig(request_plane=<class 'triton_distributed.icp.nats_request_plane.NatsRequestPlane'>,
             data_plane=<function UcpDataPlane at 0x7f477eb5d580>,
             request_plane_args=([], {}),
             data_plane_args=([], {}),
             log_level=1,
             operators=[OperatorConfig(name='encoder',
                                       implementation=<class 'triton_distributed.worker.triton_core_operator.TritonCoreOperator'>,
                                       repository='/workspace/examples/hello_world/operators/triton_core_models',
                                       version=1,
                                       max_inflight_requests=1,
                                       parameters={'config': {'instance_group': [{'count': 1,
                                                                                  'kind': 'KIND_CPU'}],
                                                              'parameters': {'delay': {'string_value': '0'},
                                                                             'input_copies': {'string_value': '1'}}}},
                                       log_level=None)],
             triton_log_path=None,
             name='encoder.0',
             log_dir='/workspace/examples/hello_world/logs',
             metrics_port=50000)
	<SpawnProcess name='encoder.0' parent=1 initial>

17:17:09 deployment.py:115[triton_distributed.worker.deployment] INFO:

Starting Worker:

	Config:
	WorkerConfig(request_plane=<class 'triton_distributed.icp.nats_request_plane.NatsRequestPlane'>,
             data_plane=<function UcpDataPlane at 0x7f477eb5d580>,
             request_plane_args=([], {}),
             data_plane_args=([], {}),
             log_level=1,
             operators=[OperatorConfig(name='decoder',
                                       implementation=<class 'triton_distributed.worker.triton_core_operator.TritonCoreOperator'>,
                                       repository='/workspace/examples/hello_world/operators/triton_core_models',
                                       version=1,
                                       max_inflight_requests=1,
                                       parameters={'config': {'instance_group': [{'count': 1,
                                                                                  'kind': 'KIND_CPU'}],
                                                              'parameters': {'delay': {'string_value': '0'},
                                                                             'input_copies': {'string_value': '1'}}}},
                                       log_level=None)],
             triton_log_path=None,
             name='decoder.0',
             log_dir='/workspace/examples/hello_world/logs',
             metrics_port=50001)
	<SpawnProcess name='decoder.0' parent=1 initial>

17:17:09 deployment.py:115[triton_distributed.worker.deployment] INFO:

Starting Worker:

	Config:
	WorkerConfig(request_plane=<class 'triton_distributed.icp.nats_request_plane.NatsRequestPlane'>,
             data_plane=<function UcpDataPlane at 0x7f477eb5d580>,
             request_plane_args=([], {}),
             data_plane_args=([], {}),
             log_level=1,
             operators=[OperatorConfig(name='encoder_decoder',
                                       implementation='EncodeDecodeOperator',
                                       repository='/workspace/examples/hello_world/operators',
                                       version=1,
                                       max_inflight_requests=1,
                                       parameters={},
                                       log_level=None)],
             triton_log_path=None,
             name='encoder_decoder.0',
             log_dir='/workspace/examples/hello_world/logs',
             metrics_port=50002)
	<SpawnProcess name='encoder_decoder.0' parent=1 initial>

Workers started ... press Ctrl-C to Exit
```

## Sending Requests

From a separate terminal run the sample client.

```
./containers/run.sh -it -- python3 -m hello_world.client
```

#### Expected Output

```

Client: 0 Received Response: 42 From: 39491f06-d4f7-11ef-be96-047bcba9020e Error: None:  43%|███████▋          | 43/100 [00:04<00:05,  9.83request/s]

Throughput: 9.10294484748811 Total Time: 10.985455989837646
Clients Stopped Exit Code 0


```

## Behind the Scenes

The hello world example is designed to demonstrate and allow
experimenting with different mixtures of compute and memory loads and
different numbers of workers for different parts of the hello world
workflow.

### Hello World Workflow

The hello world workflow is a simple two stage pipeline with an
encoding stage and a decoding stage plus an encoder-decoder stage to
orchestrate the overall workflow.

```
client <-> encoder_decoder <-> encoder
                      |
					  -----<-> decoder
```


#### Encoder

The encoder follows the simple procedure:

1. copy the input x times (x is configurable via parameter)
2. invert the input
3. delay * size of output

#### Decoder

The decoder follows the simple procedure:

1. remove the extra copies
2. invert the input
3. delay * size of output

#### Encoder - Decoder

The encoder-decoder operator controls the overall workflow.

It first sends a request for an encoder. Once it receives the response
it sends the output from the encoder as an input to the decoder. Note
in this step memory is transferred directly between the encoder and
decoder workers - and does not pass through the encoder-decoder.

### Operators

Operators are responsible for actually doing work and responding to
requests. Operators are supported in two main flavors and are hosted
by a common Worker class.

#### Triton Core Operator

The triton core operator makes a triton model (following the standard
model repo and backend structure of the tritonserver) available on the
request plane. Both the encoder and decoder are implemented as triton
python backend models.

#### Standalone Operator

The encoder-decoder operator is a standalone python class that
implements the Operator interface. Internally it makes remote requests
to other workers. Generally a standalone operator can make use of
other operators for its work but isn't required to.

### Workers

Workers host one or more operators and pull requests from the request
plane and forward them to a local operator.

### Request Plane

The current triton distributed framework leverages a distributed work
queue for its request plane implementation. The request plane ensures
that requests for operators are forwarded and serviced by a single
worker.

### Data Plane

The triton distributed framework leverages point to point data
transfers using the UCX library to provide optimized primitives for
device to device transfers.

Data sent over the data plane is only pulled by the worker that needs
to perform work on it. Requests themselves contain data descriptors
and can be referenced and shared with other workers.

Note: there is also a provision for sending data in the request
contents when the message size is small enough that UCX transfer is
not needed.

### Components

Any process which communicates with one or more of the request or data
planes is considered a "component". While this example only uses
"Workers" future examples will also include api servers, routers, and
other types of components.

### Deployment

The final piece is a deployment. A deployment is a set of components
deployed across a cluster. Components may be added and removed from
deployments.


## Limitations and Caveats

The example is a rapidly evolving prototype and shouldn't be used in
production. Limited testing has been done and it is meant to help
flesh out the triton distributed concepts, architecture, and
interfaces.

1. No multi-node testing / support has been done

2. No performance tuning / measurement has been done

