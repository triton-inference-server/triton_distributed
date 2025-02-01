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

# Tuning and Benchmarking Disaggregated Serving

**Disaggregated Serving** [^1] enables developers and teams deploying
LLMs to tune their deployment based on input and output sequence
lengths to achieve a targeted SLA with the right mix of context and
generation workers. In particular disaggregated serving enables teams
the ability to choose different parallelization strategies for each
phase and balance throughput (tokens / sec / gpu) and latency (tokens
/ sec / user).

## Example:

### 50 tokens per sec SLA with Input (3000) / Output (150)  Sequence Length Tuning

To determine the best mix of context and generate workers for a
targeted latency and input and output sequence length generally we
perform "sweeps" comparing different strategies to find the best
throughput within the SLA.

For example for input sequence length 3000 and output sequence length
150 after sweeping different tensor parallellism strategies on an two
H100 x 8 GPU nodes we've found that using 2 instances of TP 4 for
context (on one node) and using 1 instance of TP 8 for generate (on
the second node) gives the best throughput at a latency target of 50
tokens per sec per user.

At that latency target, in our early measurements disaggregated
serving outperforms traditional aggregated LLM serving by more than 1.5x
(with throughput normalized per GPU).

### Reproducing Results

To reproduce similar results on a 2 node H100 x 8 GPU system we
provide sample scripts.

### Launch Context Workers on First Node

On first (head) node:

```
bash deploy_llama_70b_context_tp2dp4.sh --head-url <head url>
```

### Launch Generate Worker on Second Node

On second node:

```
bash deploy_llama_70b_generate_tp8dp1.sh --head-url <head url>
```

### Benchmark

The following `genai-perf` command simulates traffic with 3000 input and 150 output sequence lengths.

```
genai-perf profile \
  -m llama \
  --url <api server url> \
  --endpoint-type chat \
  --streaming \
  --num-dataset-entries 100 \
  --service-kind openai \
  --endpoint v1/chat/completions \
  --warmup-request-count 10 \
  --random-seed 123 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-stddev 0 \
  --tokenizer neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
  --synthetic-input-tokens-mean 3000 \
  --output-tokens-mean 150 \
  --extra-inputs seed:100 \
  --extra-inputs min_tokens:150 \
  --extra-inputs max_tokens:150 \
  --profile-export-file my_profile_export.json \
  --artifact-dir artifacts/ \
  --concurrency < N > \
  --request-count < 10 * N > \
  -- -v \
  --async
```

### Example Results

The following results are given as an example, are not fully
optimized, and do not indicate what you may get locally.

| label    | configuration                  | concurrency | output_token_throughput_per_request | output_token_throughput_per_gpu | time_to_first_token | inter_token_latency |
|----------|--------------------------------|-------------|-------------------------------------|---------------------------------|---------------------|---------------------|
| disagg   | context_tp2dp4_generate_tp8dp1 |          48 |                    49.18197330348195      |        87.55798331              |       1157.4852116520833    |       15.935926391666667  |
| baseline | baseline_tp4dp1                |           4 |                         50.27116554062172 |                     56.26445983 |         709.2506074249999 |         15.265875249999999 |


###  Baseline Comparison

On a single node you can run a comparison. With aggregated workers we
found the best throughput at the target SLA and input and output
sequence lengths 2 two instances of tensor parallelism 4.

```
bash deploy_llama_70b_baseline_tp4dp2.sh --head-url <head url>
```

To see the results use the same `genai-perf` command used to benchmark
the disaggregated setup.


### Stopping deployment

```
pkill -SIGINT -f python3
pkill -SIGINT -f nats
```

## Known issue

Sometimes during the first run there there are nats errors. In that case just restart the deployment.

## References

[^1]: Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao
Zhang. Distserve: Disaggregating prefill and decoding for goodput-optimized large language
model serving. *arXiv:2401.09670v3 [cs.DC]*, 2024.
