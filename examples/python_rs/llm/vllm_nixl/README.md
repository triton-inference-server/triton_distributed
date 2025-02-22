## Build docker

```
./container/build.sh --framework VLLM --target dev --build-context nixl=<path to nixl repo @ main>
```

## Run container

```
./container/run.sh --framework VLLM --target dev -it
```

## Install vllm patch

```
VLLM_USE_PRECOMPILED=1 uv pip install -e "vllm  @ <path to vllm repo @ ptarasiewicz/nixl-disagg>"
```


## Run example

Run in container. In terminal 1:

```
cd /workspace/examples/python_rs/llm/vllm_nixl
CUDA_VISIBLE_DEVICES=0 python prefill.py
```

In terminal 2:

```
cd /workspace/examples/python_rs/llm/vllm_nixl
CUDA_VISIBLE_DEVICES=1 python decode.py
```




