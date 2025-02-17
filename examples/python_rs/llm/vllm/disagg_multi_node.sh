#!/bin/bash

local_ip="216.86.169.7"
remote_ips=("216.86.169.29")

# num_p/d_engines in each remote node, length should be remote_ips[@] + 1
num_p=(1 0)
num_d=(0 1)

local_triton_distribution_dir="/ephemeral/hzhou/triton_distributed/"
remote_working_dir="/ephemeral/hzhou/"

model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
p_tensor_parallel_size=1
d_tensor_parallel_size=1
max_model_len=16384
max_num_batched_tokens=16384
max_num_seqs=1024
gpu_memory_utilization=0.9
enable_chunked_prefill=False

# Docker image details
image_name="triton-distributed:latest-vllm"
image_tar="triton-distributed.tar"

# Get local Image ID
local_image_id=$(docker images --no-trunc --format "{{.ID}}" "$image_name" 2>/dev/null)

# Check if the local image exists
if [ -z "$local_image_id" ]; then
    echo "❌ Error: Local image '$image_name' not found!"
    exit 1
fi

echo "🖥️ Local Image ID: $local_image_id"

# Flag to check if we need to save the image
need_to_save=false

# Loop through each remote IP
for remote_ip in "${remote_ips[@]}"; do
    echo "🔍 Checking remote machine: $remote_ip"

    # Get remote Image ID
    remote_image_id=$(ssh "$remote_ip" "docker images --no-trunc --format \"{{.ID}}\" \"$image_name\" 2>/dev/null")

    echo "🔄 Remote Image ID on $remote_ip: ${remote_image_id:-'Not Found'}"

    # Compare Image IDs
    if [ "$local_image_id" != "$remote_image_id" ] || [ -z "$remote_image_id" ]; then
        echo "⚠️ Image mismatch or not found on $remote_ip. It needs the image."
        need_to_save=true
        break
    fi
done

# Save the image **only if needed**
if [ "$need_to_save" = true ]; then
    echo "💾 Saving image to tar file..."
    docker save -o "$image_tar" "$image_name"
fi

# Loop through each remote IP
for remote_ip in "${remote_ips[@]}"; do
    echo "🔄 Sync remote working dir"
    rsync -avz --progress ${local_triton_distribution_dir} ${remote_ip}:${remote_working_dir}/triton_distributed/

    echo "🔍 Checking remote machine: $remote_ip"

    # Get remote Image ID
    remote_image_id=$(ssh "$remote_ip" "docker images --no-trunc --format \"{{.ID}}\" \"$image_name\" 2>/dev/null")

    echo "🔄 Remote Image ID on $remote_ip: $remote_image_id"

    # Compare Image IDs
    if [ "$local_image_id" == "$remote_image_id" ] && [ -n "$remote_image_id" ]; then
        echo "✅ $remote_ip already has the correct image."
    else
        echo "⚠️ Image mismatch or not found on $remote_ip. Transferring..."
        
        # Transfer the image tar file
        scp "$image_tar" "$remote_ip:${remote_working_dir}/"

        # Load the image on the remote machine
        ssh "$remote_ip" "docker load -i ${remote_working_dir}/$image_tar && rm ${remote_working_dir}/$image_tar"

        echo "✅ Image successfully transferred and loaded on $remote_ip."
    fi

    echo "-----------------------------------"
done

# Cleanup: remove the tar file after distribution
rm -f "$image_tar"

echo "🎉 Image synchronization complete for all machines!"

docker stop $(docker ps -q --filter "ancestor=nats")
docker stop $(docker ps -q --filter "ancestor=bitnami/etcd")
docker compose -f ${local_triton_distribution_dir}/runtime/rust/docker-compose.yml up -d 

nats_server="nats://${local_ip}:4222"
etcd_endpoints="http://${local_ip}:2379"

# start nats and etcd services
for remote_ip in "${remote_ips[@]}"; do
    echo "🔄 starting etcd and nats service on $remote_ip"
    ssh "$remote_ip" "docker stop \$(docker ps -q --filter "ancestor=nats")"
    ssh "$remote_ip" "docker stop \$(docker ps -q --filter "ancestor=bitnami/etcd")"
    ssh "$remote_ip" "docker compose -f ${remote_working_dir}/triton_distributed/runtime/rust/docker-compose.yml up -d"

    etcd_endpoints="${etcd_endpoints},http://${remote_ip}:2379"
done

total_rank=0
for num in "${num_p[@]}"; do
    total_rank=$((total_rank + num))
done
for num in "${num_d[@]}"; do
    total_rank=$((total_rank + num))
done
echo "🔍 total_rank: $total_rank"

# start workers locally
node_rank=0
echo "🔄 starting workers locally"
docker stop $(docker ps -q --filter "ancestor=triton-distributed:latest-vllm")
${local_triton_distribution_dir}/container/run.sh --framework VLLM -v ${local_triton_distribution_dir}:/triton_distributed_repo -d --command "tail -f /dev/null"
sleep 5 # make sure docker start up correctly
docker exec $(docker ps -q --filter "ancestor=triton-distributed:latest-vllm") bash -c \
"
source /opt/triton/venv/bin/activate && \
export NATS_SERVER=\"${nats_server}\" && \
export ETCD_ENDPOINTS=\"${etcd_endpoints}\" && \
cd /triton_distributed_repo/examples/python_rs/llm/vllm && \
./disagg_single_node.sh \
    --model ${model} \
    --p_tensor_parallel_size ${p_tensor_parallel_size} \
    --d_tensor_parallel_size ${d_tensor_parallel_size} \
    --max_model_len ${max_model_len} \
    --max_num_batched_tokens ${max_num_batched_tokens} \
    --max_num_seqs ${max_num_seqs} \
    --gpu_memory_utilization ${gpu_memory_utilization} \
    --enable_chunked_prefill ${enable_chunked_prefill} \
    --kv_ip ${local_ip} \
    --kv_port 14579 \
    --num_p ${num_p[$node_rank]} \
    --num_d ${num_d[$node_rank]} \
    --total_rank ${total_rank} \
    --curr_kv_rank 0 \
"
curr_kv_rank=$((curr_kv_rank + num_p[node_rank]))
curr_kv_rank=$((curr_kv_rank + num_d[node_rank]))
node_rank=$((node_rank + 1))

# start remote workers
for remote_ip in "${remote_ips[@]}"; do
    echo "🔄 starting workers on $remote_ip"
    ssh ${remote_ip} << EOF
docker stop \$(docker ps -q --filter "ancestor=triton-distributed:latest-vllm")
${remote_working_dir}/triton_distributed/container/run.sh --framework VLLM -v ${remote_working_dir}/triton_distributed/:/triton_distributed_repo -d --command "tail -f /dev/null"
sleep 5 # make sure docker start up correctly
docker exec \$(docker ps -q --filter "ancestor=triton-distributed:latest-vllm") bash -c "
source /opt/triton/venv/bin/activate && \
export NATS_SERVER=\"${nats_server}\" && \
export ETCD_ENDPOINTS=\"${etcd_endpoints}\" && \
cd /triton_distributed_repo/examples/python_rs/llm/vllm && \
./disagg_single_node.sh \
    --model ${model} \
    --p_tensor_parallel_size ${p_tensor_parallel_size} \
    --d_tensor_parallel_size ${d_tensor_parallel_size} \
    --max_model_len ${max_model_len} \
    --max_num_batched_tokens ${max_num_batched_tokens} \
    --max_num_seqs ${max_num_seqs} \
    --gpu_memory_utilization ${gpu_memory_utilization} \
    --enable_chunked_prefill ${enable_chunked_prefill} \
    --kv_ip ${local_ip} \
    --kv_port 14579 \
    --num_p ${num_p[${node_rank}]} \
    --num_d ${num_d[${node_rank}]} \
    --total_rank ${total_rank} \
    --curr_kv_rank ${curr_kv_rank}
"
EOF
    curr_kv_rank=$((curr_kv_rank + num_p[node_rank]))
    curr_kv_rank=$((curr_kv_rank + num_d[node_rank]))
    node_rank=$((node_rank + 1))
done