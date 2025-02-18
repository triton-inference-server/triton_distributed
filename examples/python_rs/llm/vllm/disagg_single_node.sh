#!/bin/bash

# default values
model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
p_tensor_parallel_size=1
d_tensor_parallel_size=1
max_model_len=16384
max_num_batched_tokens=16384
max_num_seqs=1024
gpu_memory_utilization=0.9
enable_chunked_prefill=False
kv_ip=127.0.0.1
kv_port=14579

num_p=1
num_d=1
total_rank=$((num_p + num_d))
curr_kv_rank=0

# Function to display usage
usage() {
    echo "Usage: $0 [--model <model>] [--p_tensor_parallel_size <size>] [--d_tensor_parallel_size <size>] [--max_model_len <len>] [--max_num_batched_tokens <tokens>] [--max_num_seqs <seqs>] [--gpu_memory_utilization <utilization>] [--enable_chunked_prefill <True/False>] [--kv_ip <ip>] [--kv_port <port>] [--num_p <p>] [--num_d <d>]"
    exit 1
}

# Parse the command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            model="$2"
            shift 2
            ;;
        --p_tensor_parallel_size)
            p_tensor_parallel_size="$2"
            shift 2
            ;;
        --d_tensor_parallel_size)
            d_tensor_parallel_size="$2"
            shift 2
            ;;
        --max_model_len)
            max_model_len="$2"
            shift 2
            ;;
        --max_num_batched_tokens)
            max_num_batched_tokens="$2"
            shift 2
            ;;
        --max_num_seqs)
            max_num_seqs="$2"
            shift 2
            ;;
        --gpu_memory_utilization)
            gpu_memory_utilization="$2"
            shift 2
            ;;
        --enable_chunked_prefill)
            enable_chunked_prefill="$2"
            shift 2
            ;;
        --kv_ip)
            kv_ip="$2"
            shift 2
            ;;
        --kv_port)
            kv_port="$2"
            shift 2
            ;;
        --num_p)
            num_p="$2"
            shift 2
            ;;
        --num_d)
            num_d="$2"
            shift 2
            ;;
        --total_rank)
            total_rank="$2"
            shift 2
            ;;
        --curr_kv_rank)
            curr_kv_rank="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

# rank here is GPU rank
curr_rank=0

echo "total rank: "${total_rank}

for (( i=1; i<=num_p; i++ )); do
    cuda_devices=$(seq $curr_rank $(($curr_rank + $p_tensor_parallel_size - 1)))
    cuda_devices=$(echo $cuda_devices | tr ' ' ',')
    echo "starting gpu rank "${cuda_devices}" (prefill)"


    VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=${cuda_devices} python3 -u -m disaggregated.prefill_worker \
    --model ${model} \
    --max-model-len ${max_model_len} \
    --max-num-batched-tokens ${max_num_batched_tokens} \
    --enable-chunked-prefill ${enable_chunked_prefill} \
    --gpu-memory-utilization ${gpu_memory_utilization} \
    --enforce-eager \
    --tensor-parallel-size ${p_tensor_parallel_size} \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":'${curr_kv_rank}',"kv_parallel_size":'${total_rank}',"kv_ip":"'${kv_ip}'","kv_port":'${kv_port}'}' & 
    disown
    curr_rank=$((curr_rank + p_tensor_parallel_size))
    curr_kv_rank=$((curr_kv_rank + 1))
done

for (( i=1; i<=num_d; i++ )); do
    cuda_devices=$(seq $curr_rank $(($curr_rank + $d_tensor_parallel_size - 1)))
    cuda_devices=$(echo $cuda_devices | tr ' ' ',')
    echo "starting gpu rank "${cuda_devices}" (decode)"

    VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=${cuda_devices} python3 -u -m disaggregated.decode_worker \
    --model ${model} \
    --max-model-len ${max_model_len} \
    --max-num-batched-tokens ${max_num_batched_tokens} \
    --enable-chunked-prefill ${enable_chunked_prefill} \
    --gpu-memory-utilization ${gpu_memory_utilization} \
    --enforce-eager \
    --tensor-parallel-size ${d_tensor_parallel_size} \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":'${curr_kv_rank}',"kv_parallel_size":'${total_rank}',"kv_ip":"'${kv_ip}'","kv_port":'${kv_port}'}' & 
    disown
    curr_rank=$((curr_rank + d_tensor_parallel_size))
    curr_kv_rank=$((curr_kv_rank + 1))
done