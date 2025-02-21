# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import asyncio
import json

import uvloop
from triton_distributed_rs import DistributedRuntime, triton_worker

from .protocol import Request


@triton_worker()
async def worker(
    runtime: DistributedRuntime, 
    prompts: list[str], 
    timestamps: list[float],
    max_tokens_list: int, 
    temperature: float,
    gamma: float,
    balance_threshold: float
):
    """
    Instantiate a `backend` client and call the `generate` endpoint asynchronously
    based on provided prompts and timestamps
    """
    endpoint = runtime.namespace("triton-init").component("preprocess").endpoint("generate")
    client = await endpoint.client()

    async def process_prompt(prompt: str, delay: float, max_tokens: int):
        if delay > 0:
            await asyncio.sleep(delay)
        
        req_start_time = asyncio.get_event_loop().time()
        first_token_time = None
        token_times = []
        last_token_time = req_start_time

        stream = await client.generate(
            Request(
                prompt=prompt,
                sampling_params={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "ignore_eos": True, 
                },
            ).model_dump_json()
        )
        
        async for resp in stream:
            current_time = asyncio.get_event_loop().time()
            if first_token_time is None:
                first_token_time = current_time - req_start_time
            else:
                token_times.append(current_time - last_token_time)
            last_token_time = current_time
        
        avg_token_time = sum(token_times) / len(token_times) if token_times else 0
        return {
            "time_to_first_token": first_token_time,
            "avg_time_between_tokens": avg_token_time
        }

    # Create tasks with their respective delays
    tasks = []
    start_time = asyncio.get_event_loop().time()  # Add global start time here
    for prompt, timestamp, max_tokens in zip(prompts, timestamps, max_tokens_list):
        tasks.append(process_prompt(prompt, timestamp, max_tokens))

    # Run all tasks concurrently and collect timing results
    results = await asyncio.gather(*tasks)
    
    # Calculate average metrics across all prompts
    avg_first_token = sum(r["time_to_first_token"] for r in results) / len(results)
    avg_between_tokens = sum(r["avg_time_between_tokens"] for r in results) / len(results)
    
    # Calculate total time from start to finish
    total_time = asyncio.get_event_loop().time() - start_time
    
    # Save results to JSON file
    results_data = {
        "average_time_to_first_token": avg_first_token,
        "average_time_between_tokens": avg_between_tokens,
        "total_time": total_time,
        "gamma": gamma,
        "balance_threshold": balance_threshold
    }
    
    # Append results to JSONL file
    with open('benchmark_results.jsonl', 'a') as f:
        f.write(json.dumps(results_data) + '\n')

    print(f"\nAverage time to first token: {avg_first_token:.3f} seconds")
    print(f"Average time between tokens: {avg_between_tokens:.3f} seconds")
    print(f"Total time for all requests: {total_time:.3f} seconds")


if __name__ == "__main__":
    uvloop.install()
    
    import json
    
    max_count = 1000
    
    with open('/workspace/examples/datasets/mooncake_trace_synthesized_processed_10000samples.json', "r") as f:
        prompts = json.load(f)
    with open('/workspace/examples/datasets/mooncake_trace_synthesized.jsonl', "r") as f:
        data = [json.loads(line) for line in f]
    prompts, data = prompts[:max_count], data[:max_count]
    
    # Filter out samples where combined length exceeds 16000
    valid_indices = [i for i, d in enumerate(data) 
                    if float(d['input_length']) + float(d['output_length']) <= 16000]
    
    prompts = [prompts[i] for i in valid_indices]
    timestamps = [float(data[i]['timestamp']) / 2000 for i in valid_indices]
    max_tokens_list = [int(data[i]['output_length']) for i in valid_indices] 
    
    # Normalize timestamps to start from 0
    base_time = min(timestamps)
    timestamps = [t - base_time for t in timestamps]

    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--balance_threshold', type=float, default=0.1)
    args = parser.parse_args()

    asyncio.run(worker(prompts, timestamps, max_tokens_list, 0.5, args.gamma, args.balance_threshold))