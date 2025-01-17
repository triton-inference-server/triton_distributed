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
