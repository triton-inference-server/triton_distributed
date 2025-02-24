#!/bin/sh

nats-server \
    -js \
    --trace \
    -p 4222 \
    -m 8222 &

sleep 15

etcd \
    --listen-peer-urls http://localhost:2380 \
    --listen-client-urls http://localhost:2379 \
    --advertise-client-urls http://localhost:2379 \
    --initial-cluster-token etcd-cluster-1 \
    --initial-cluster etcd-server=http://localhost:2380 \
    --initial-cluster-state new &

sleep 15

http &

sleep 15

llmctl http add chat-models deepseek-ai/DeepSeek-R1-Distill-Llama-8B triton-init.vllm.generate&

sleep 15

source /opt/triton/venv/bin/activate
cd /workspace/examples/python_rs/llm/vllm
VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0,1 python3 -m disaggregated.prefill_worker \
            --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
            --max-model-len 100 \
            --gpu-memory-utilization 0.8 \
            --enforce-eager \
            --tensor-parallel-size 2 \
            --kv-transfer-config \
            '{"kv_connector":"TritonNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'

VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=2,3 python3 -m disaggregated.prefill_worker \
            --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
            --max-model-len 100 \
            --gpu-memory-utilization 0.8 \
            --enforce-eager \
            --tensor-parallel-size 2 \
            --kv-transfer-config \
            '{"kv_connector":"TritonNcclConnector","kv_role":"kv_producer","kv_rank":1,"kv_parallel_size":3}'



VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=1 python3 -m disaggregated.decode_worker \
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --max-model-len 100 \
        --gpu-memory-utilization 0.8 \
        --enforce-eager \
        --tensor-parallel-size 4 \
        --kv-transfer-config \
        '{"kv_connector":"TritonNcclConnector","kv_role":"kv_consumer","kv_rank":2,"kv_parallel_size":3}'

