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

Run in container. Clean up metadata files:
```
rm -r /tmp/nixl
```

In terminal 0:
```
llmctl http add chat-models deepseek-ai/DeepSeek-R1-Distill-Llama-8B test-nixl.process.chat/completions
TRT_LOG=DEBUG http --port 8181
```


In terminal 1:

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
curl localhost:8181/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

## Run GAP

```
genai-perf profile \
  -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --url localhost:8181 \
  --endpoint-type chat \
  --streaming \
  --service-kind openai \
  --endpoint v1/chat/completions \
  --warmup-request-count 10 \
  --random-seed 123 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-stddev 0 \
  --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --synthetic-input-tokens-mean 80 \
  --output-tokens-mean 10 \
  --extra-inputs min_tokens:10 \
  --extra-inputs max_tokens:10 \
  --profile-export-file my_profile_export.json \
  --artifact-dir artifacts/ \
  --concurrency 1 \
  --request-count 10 \
  -- -v \
  --async
```

## TODOs, limitations, known issues

- [ ] Add etcd for discovery
- [ ] Enable chunked prefill
- [ ] Support mixed tp
- [ ] Enable async output processing - could be working
- [ ] Process many remote prefill in one iteration
- [ ] Support recompute preemption
- [ ] Make sure decode does not preempt blocks before xfer finishes
- [ ] Layer wise transfer
- [ ] Non blocking send in prefill (cache manager should check xfer status)
- [ ] Test under load
- [ ] Support pp > 1
- [ ] Check why adding extra seed input is crashing vllm with remote prefill
- [ ] Concurrency > 2 is not working
- [ ] Require sending two parallel requests to start decode for the first time
- [ ] Parse cmdline args
- [x] Manual nixl example with tp1
- [x] Zero copy
- [x] Conditional remote prefill
- [x] Manual example with tp > 1
- [x] Run on triton distributed runtime
- [x] add oai http endpoint
- [x] Sample only on decode, do note return remote prefill response
- [x] Check if all transfers finished before moving to decode
