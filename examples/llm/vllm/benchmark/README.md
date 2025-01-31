

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


## Benchmark

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