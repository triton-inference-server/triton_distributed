# vLLM Example

This example demonstrates how to use the Triton Distributed to serve the vLLM engine.

## Pre-requisite

Please refer to the [README](../../README.md) for the pre-requisite and virtual environment setup.

### vLLM installation

```
uv pip install setuptools vllm==0.7.0
```

## Run the monolith example

In the first shell, run the server:

```
python monolith_worker.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --max-model-len 100
```


In the second shell, run the client:

```
python client.py
```

## Run the disaggregated example

In the first shell, run the prefill worker:

```
python prefill_worker.py \
    --model facebook/opt-125m \
    --max-model-len 100 \
    --gpu-memory-utilization 0.4 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'
```

In the second shell, run the decode worker:

```
python decode_worker.py \
    --model facebook/opt-125m \
    --max-model-len 100 \
    --gpu-memory-utilization 0.4 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}'
```

In the third shell, run the client:

```
python client.py
```
