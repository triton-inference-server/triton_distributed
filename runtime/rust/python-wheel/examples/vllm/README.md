# vLLM Example

This example demonstrates how to use the Triton Distributed to serve the vLLM engine.

## Pre-requisite

Please refer to the [README](../../README.md) for the pre-requisite and virtual environment setup.

### vLLM installation

```
uv pip install setuptools vllm==0.7.0
```

## Run the example

In the first shell, run the server:

```
python vllm_worker.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --max-model-len 100
```


In the second shell, run the client:

```
python client.py
```

