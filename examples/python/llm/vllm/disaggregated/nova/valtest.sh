#!/bin/bash

# Initial configuration
model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
p_tensor_parallel_size=1  # for prefill workers
d_tensor_parallel_size=2  # for decode workers
max_model_len=16384
max_num_batched_tokens=16384
max_num_seqs=1024
gpu_memory_utilization=0.9
enable_chunked_prefill=False
num_p=1  # number of prefill workers
num_d=1  # number of decode workers
total_rank=$((num_p + num_d))
curr_rank=0
curr_kv_rank=0
kv_ip="localhost"  # example value
kv_port=8888      # example value

echo "=== Initial Configuration ==="
echo "Model: ${model}"
echo "Total rank: ${total_rank}"
echo "Prefill tensor parallel size: ${p_tensor_parallel_size}"
echo "Decode tensor parallel size: ${d_tensor_parallel_size}"
echo ""

echo "=== Prefill Workers ==="
for (( i=1; i<=num_p; i++ )); do
    cuda_devices=$(seq $curr_rank $(($curr_rank + $p_tensor_parallel_size - 1)))
    cuda_devices=$(echo $cuda_devices | tr ' ' ',')
    
    echo "Prefill Worker $i:"
    echo "  CUDA_VISIBLE_DEVICES: ${cuda_devices}"
    echo "  Current Rank: ${curr_rank}"
    echo "  Current KV Rank: ${curr_kv_rank}"
    echo "  KV Config:"
    echo "    Connector: PyNcclConnector"
    echo "    Role: kv_producer"
    echo "    KV Rank: ${curr_kv_rank}"
    echo "    KV Parallel Size: ${total_rank}"
    echo "    KV IP: ${kv_ip}"
    echo "    KV Port: ${kv_port}"
    echo ""
    
    curr_rank=$((curr_rank + p_tensor_parallel_size))
    curr_kv_rank=$((curr_kv_rank + 1))
done

echo "=== Decode Workers ==="
for (( i=1; i<=num_d; i++ )); do
    cuda_devices=$(seq $curr_rank $(($curr_rank + $d_tensor_parallel_size - 1)))
    cuda_devices=$(echo $cuda_devices | tr ' ' ',')
    
    echo "Decode Worker $i:"
    echo "  CUDA_VISIBLE_DEVICES: ${cuda_devices}"
    echo "  Current Rank: ${curr_rank}"
    echo "  Current KV Rank: ${curr_kv_rank}"
    echo "  KV Config:"
    echo "    Connector: PyNcclConnector"
    echo "    Role: kv_consumer"
    echo "    KV Rank: ${curr_kv_rank}"
    echo "    KV Parallel Size: ${total_rank}"
    echo "    KV IP: ${kv_ip}"
    echo "    KV Port: ${kv_port}"
    echo ""
    
    curr_rank=$((curr_rank + d_tensor_parallel_size))
    curr_kv_rank=$((curr_kv_rank + 1))
done