

## Run baseline

```
bash deploy_llama_70b_baseline_tp4dp2.sh --head-url <head url>
```

## Run disagg

On head node:
```
bash deploy_llama_70b_context_tp2dp4.sh --head-url <head url>
```

on a second node:
```
bash deploy_llama_70b_generate_tp8dp1.sh --head-url <head url>
```


## Benchmark disagg

```
genai-perf profile \
  -m llama \
  --url <api server url> \
  --endpoint-type chat \
  --streaming \
  --num-dataset-entries 100 \
  --service-kind openai \
  --endpoint v1/chat/completions \
  --warmup-request-count 10 \
  --random-seed 123 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-stddev 0 \
  --tokenizer neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
  --synthetic-input-tokens-mean 3000 \
  --output-tokens-mean 150 \
  --extra-inputs seed:100 \
  --extra-inputs min_tokens:150 \
  --extra-inputs max_tokens:150 \
  --profile-export-file my_profile_export.json \
  --artifact-dir artifacts/ \
  --concurrency < N > \
  --request-count < 10 * N > \
  -- -v \
  --async
```

## Results (placeholder)

| label    | configuration                  | concurrency | output_token_throughput_per_request | output_token_throughput_per_gpu | time_to_first_token | inter_token_latency |
|----------|--------------------------------|-------------|-------------------------------------|---------------------------------|---------------------|---------------------|
| disagg   | context_tp4dp1_generate_tp4dp1 |          16 |                         45.70218424 |                     86.75790947 |         719.2443455 |         17.47573939 |
| baseline | baseline_tp4dp1                |           4 |                         49.60947526 |                     49.60134097 |         500.9769517 |         16.93055177 |



## Stopping deployment

```
pkill -9 -f python3
pkill -9 -f nats
```


## Known issue

Sometimes during the first run there there are nats errors. In that case just restart the deployment.