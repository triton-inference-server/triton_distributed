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
CUDA_VISIBLE_DEVICES=0,1 python prefill.py
```

In terminal 2:

```
cd /workspace/examples/python_rs/llm/vllm_nixl
CUDA_VISIBLE_DEVICES=2,3 python decode.py
```



In terminal 3:
```
cd /workspace/examples/python_rs/llm/vllm_nixl
python client.py --prompt "A B C D E" --remote-prefill
python client.py --prompt "A B C D E"
```

## TODO

- [x] Manual nixl example with tp1
- [x] Zero copy
- [x] Conditional remote prefill
- [x] Manual example with tp > 1
- [x] Run on triton distributed runtime
- [ ] [Piotr] add oai http endpoint
- [ ] [Piotr] Sample only on decode, do note return remote prefill response
- [ ] [Neelay] Add etcd for discovery
- [ ] [Alec] Enable chunked prefill
- [ ] Support mixed tp
- [ ] Enable async output processing - could be working
- [ ] Process many remote prefill in one iteration
- [ ] Support recompute preemption
- [ ] Make sure decode does not preempt blocks before xfer finishes
- [ ] Layer wise transfer
- [ ] Non blocking send in prefill (cache manager should check xfer status)
- [ ] Test under load
- [ ] Check if all transfers finished before moving to decode
- [ ] Support pp > 1
