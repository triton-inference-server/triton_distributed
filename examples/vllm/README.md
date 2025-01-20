<!--
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# vLLM

vLLM disaggregated serving example.

## Setup environment

Start in the root of this repository:
```
cd <repo_root>
```

### Docker repository

The easiest way to run the example is to pull and run the prebuild docker image from the repository:

```bash
./container/run.sh --image gitlab-master.nvidia.com/dl/triton/triton-3:triton_with_vllm
```

### Manual build

#### Build docker

```bash
export GITLAB_TOKEN=<your gitlab token>
./container/build.sh --framework VLLM
```

#### Run docker

```bash
./container/run.sh --framework VLLM
```

#### vLLM fork
In the docker vLLM is installed from the vLLM fork. You can also install it manually:
```bash
git clone https://gitlab-master.nvidia.com/dl/vllm/vllm.git
cd vllm
git checkout ptarasiewicz/disaggregated_inference
pip install .
```
This fork contains changes required for running this example.

## Run example

Go to the example directory:
```bash
cd examples/vllm
```

When model is not cached locally it will be downloaded from the HuggingFace Hub. Make sure to make the access token available by setting the environment variable:

```bash
export HF_TOKEN=<HF token>
```

Run the disaggregated server:
```bash
./scripts/launch_workers.py --model-ckpt neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 --model-name llama --context-tp-size 2 --context-workers 2 --generate-tp-size 4 --generate-workers 1
```

or baseline (aggregated):
```bash
./scripts/launch_workers.py --model-ckpt neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 --model-name llama --baseline-tp-size 4 --baseline-workers 2
```

> [!NOTE]
> The FP8 variant of the Llama 3.1 model `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8`
> is chosen above to reduce the size of the data transfer in KV Cache movements.
> Only GPUs with Compute Capability >= 8.9 can use FP8. For older GPUs, try
> using a standard BF16/FP16 precision variant of the model such as
> `meta-llama/Meta-Llama-3.1-8B-Instruct`.


Run client:

```bash
# localhost:8005 is the default, but should be changed based on server deployment args.
curl localhost:8005/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [
      {"role": "system", "content": "What is the capital of France?"}
    ],
    "temperature": 0, "top_p": 0.95, "max_tokens": 25, "stream": true, "n": 1, "frequency_penalty": 0.0, "stop": []
  }'
```

## Benchmark

To run benchmarking script run:
```bash
./scripts/run_benchmark.py --model llama --url localhost:8005 --isl 3000 --osl 150 --load-type concurrency --load-value 1 --request-count 10
```


## Known limitations
- Only one generate worker is supported.
- Number of workers must be fixed when launching the server.
- Llama3.1 tokenizer is hard coded in the API server.
- Streaming currently only returns 2 responses: The 1st response has the 1st token
  from prefill, and the 2nd response contains the remaining N-1 generated tokens.
