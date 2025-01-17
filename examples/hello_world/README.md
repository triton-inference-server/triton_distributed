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
pipeline.

### Hello World Pipeline

The hello world pipeline is a simple two stage pipeline with an
encoding stage and a decoding stage plus a


<!--

```
examples/
└── hello_world
    ├── README.md
    ├── api_server
    ├── client (optional)
    ├── deploy
    │   └── __main__.py (should it contain all workers, the example have here also API server-like logic to publish requests from users)
    ├── docs
    ├── operators
    │   └── triton_core_models (optional)
    │       ├── decoder
    │       │   ├── 1
    │       │   │   └── model.py
    │       │   └── config.pbtxt
    │       └── encoder
    │           ├── 1
    │           │   └── model.py
    │           └── config.pbtxt
    ├── router (optional)
    ├── scripts (What should be here?)
    ├── single_file.py
    └── tests
```

Review plans for deploy cli / client cli
```
deploy --encoder workers:instances:device --decoder workers:instances:device --encoder-decoder workers
  in future
  deploy --api-server <kserve>
  deploy --request-plane nginx  (would need to convert encode decode into bls?)
```


Below is a high-level overview of how Triton Distributed is organized, with special attention to the “hello world” example that demonstrates how the system’s pieces fit together.

---


## 3. “Hello World” Layout
In `examples/hello_world/`, you see a minimal demonstration of how to:

1. Create a few Triton models (the “encoder” and “decoder”).
2. Start a small distributed deployment with these models.
3. Send requests in parallel and demonstrate data-plane usage.

### 3.1 Directory Structure

```
examples/hello_world/
  deploy/
    __main__.py     # Entry point that starts the “hello world” deployment
  operators/
    triton_core_models/
      encoder/1/model.py    # Python model code for an “encoder” step
      decoder/1/model.py    # Python model code for a “decoder” step
```

#### (a) The `__main__.py` (Deploy Script)
This file spins up everything end-to-end:

- Creates a local NATS server object (`nats_server`) so that requests can be published and consumed.
- Defines **OperatorConfig** objects for the two Triton models, `encoder` and `decoder`. Each references a local path to the Python model code and custom parameters (e.g., instance group, concurrency, etc.).
- Defines a custom “orchestrator” operator named `encoder_decoder` (`EncodeDecodeOperator` in the code) that chains calls to the `encoder` and `decoder`.
- Creates three WorkerConfig entries:
  1. Worker that hosts the `encoder` model  **(REMOVE in Python)**
  2. Worker that hosts the `decoder` model  **(REMOVE in Python)**
  3. Worker that hosts the aggregator operator (`encoder_decoder`) (Python HERE)
- Launches all three processes with a `Deployment` object. (We need separate entry points for API server)
- Sends test requests to `encoder_decoder` operator (which calls `encoder` then `decoder`) and verifies the results. (this will run vLLM)


#### (c) The `EncodeDecodeOperator`
This is a custom operator (in `deploy/__main__.py` as a short class, or sometimes in a separate file) that demonstrates how to chain calls:

```python
for request in requests:
  # 1. Send "input" to the "encoder" model
  encoded_responses = await self._encoder.async_infer(inputs={"input": request.inputs["input"]})

  # 2. When the encoder finishes, read "input_copies" from the response
  #    then call “decoder” with the “encoded” output
  decoded_responses = await self._decoder.async_infer(
      inputs={"input": encoded_response.outputs["output"]},
      parameters={"input_copies": input_copies},
  )

  # 3. Return the result back to the user
  await request.response_sender().send(final=True, outputs={"output": decoded_response.outputs["output"]})
```

Hence, the aggregator itself is just a normal Python class implementing the `Operator` interface, but inside it calls **RemoteOperator** objects for actual inference.

## 5. How the Hello World Example Flows
1. **`main()`** in `examples/hello_world/deploy/__main__.py` starts:
   - A local NATS server for request-plane traffic.
   - Worker processes for “encoder,” “decoder,” and “encoder_decoder.”
2. Each Worker loads a Python model or an Operator class:
   - The `encoder` Worker loads the model code from `encoder/1/model.py`.
   - The `decoder` Worker does the same for `decoder/1/model.py`.
   - The `encoder_decoder` Worker instantiates the `EncodeDecodeOperator` Python class, which calls `encoder` and `decoder` remotely.
3. The script then calls `send_requests(nats_server_url)`:
   - It uses a **RemoteOperator** for `encoder_decoder` and does something like:
     ```python
     remote_operator: RemoteOperator = RemoteOperator("encoder_decoder", 1, request_plane, data_plane)
     await remote_operator.async_infer(inputs={"input": some_numpy_array})
     ```
4. The `async_infer()` method publishes a request to the “encoder_decoder” Worker (via `NatsRequestPlane`) and references data (via `UcpDataPlane`).
5. The aggregator Worker receives the request, calls `_encoder.async_infer()`, which sends a second request to the “encoder” Worker:
   - The “encoder” Worker runs the simple tile/invert logic.
   - Once done, it returns the result to the aggregator Worker.
6. The aggregator Worker then calls `_decoder.async_infer()`, which calls the “decoder” Worker’s model, which re-inverts and slices the data, returning it back.
7. Finally, the aggregator Worker returns the final “decoded” data to the original caller in `send_requests`.

-->
