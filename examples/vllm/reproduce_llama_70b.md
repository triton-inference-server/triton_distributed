### Setup
```
Model: neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8
Attention: FlashInfer
HW: EOS H100
```

### Results

![0% cached](images/pareto_plot_uncached.png)
![99% cached](images/pareto_plot_cached.png)

### Reproduction

 - sbatch scripts are stand alone, source code is included in the docker image, it is not necessary to clone the repo.
 - in the graphs above:
   - `tokens/s/gpu` is `output_token_throughput` metric in the genai-perf results normalized by the number of GPUs used (16 in the sbatch scripts)
   - `tokens/s/user` is `output_token_throughput_per_request` metric in the genai-perf results

To start a Triton 3 server on EOS run for disagg (default context tp2dp4 generate tp8dp1):
```
sbatch scripts/run_llama_70b_disagg.sh
```

for baseline (default tp4dp4):

```
sbatch scripts/run_llama_70b_baseline.sh
```


To run genai-perf for 0 cached / 3000 context / 150 generated and concurrency 32:

```
genai-perf profile \
-m llama \
--url <first slurm node for the job running server>:8005 \
--endpoint-type chat \
--streaming \
--num-dataset-entries 1000 \
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
--concurrency 32 \
--request-count 320 \
-- -v \
--async
```


To run genai-perf for 2970 cached / 30 context / 150 generated and concurrency 32:

```
genai-perf profile \
-m llama \
--url <first slurm node for the job running server>:8005 \
--endpoint-type chat \
--streaming \
--num-dataset-entries 1000 \
--service-kind openai \
--endpoint v1/chat/completions \
--warmup-request-count 10 \
--random-seed 123 \
--synthetic-input-tokens-stddev 0 \
--output-tokens-stddev 0 \
--tokenizer neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
--synthetic-input-tokens-mean 30 \
--output-tokens-mean 150 \
--extra-inputs seed:100 \
--extra-inputs min_tokens:150 \
--extra-inputs max_tokens:150 \
--profile-export-file my_profile_export.json \
--artifact-dir artifacts/ \
--num-prefix-prompts 1 \
--prefix-prompt-length 2970 \
--concurrency 32 \
--request-count 320 \
-- -v \
--async
```
